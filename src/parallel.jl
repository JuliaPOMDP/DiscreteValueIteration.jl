"""
    ParallelValueIterationSolver

Parallel asynchronous (Gauss-Seidel) Value Iteration solver. It allows to benefits from multiple CPUs to speed up Value Iteration.
# Fields:
- `max_iterations::Int64`: number of iterations to run VI, default=100
- `belres::Float64`: stop when the Bellman residual is lower than this value, default=1e-3
- `n_procs::Int64`: number of processes to use, default=`Sys.CPU_CORES` (maximum available on the machine)
- `include_Q::Bool`: if set to true, returns the state action values as well, default=true
- `state_order::Vector{Tuple{Int64, Int64}}`: provide a decomposition of the state space to treat serially. 
Each element of state_order is the start and end index of a chunk of states in the state space. Each chunk is solved in parallel.
"""
@with_kw mutable struct ParallelValueIterationSolver <: Solver
    max_iterations::Int64 = 100 # max number of iterations 
    belres::Float64 = 1e-3 # the Bellman Residual
    n_procs::Int64 = Sys.CPU_CORES# number of processors to use
    include_Q::Bool = true
    state_order::Vector{Tuple{Int64, Int64}} = Tuple{Int64, Int64}[] # contains chunks of state indices to process serially, each element is the start and end idx of a chunk
end

function solve(solver::ParallelValueIterationSolver, mdp::Union{MDP,POMDP},
               policy::ValueIterationPolicy=ValueIterationPolicy(mdp, include_Q=true);
               verbose::Bool=false)

    @warn_requirements solve(solver, mdp)
        
    # processor restriction checks    
    n_procs   = solver.n_procs
    n_procs < 2 ? error("Less than 2 processors not allowed") : nothing
    n_procs > Sys.CPU_CORES ? error("Requested too many processors") : nothing
   
    # check if a default state ordering is needed
    order = solver.state_order
    if isempty(order)
        ns = n_states(mdp)
        solver.state_order = [(1,ns)]
    end
    verbose ? println("Starting parallel Gauss Seidel Value Iteration with $n_procs cores") : nothing
    verbose ? flush(STDOUT) : nothing
    gauss_seidel(solver, mdp, policy, verbose=verbose)
end

function gauss_seidel(solver::ParallelValueIterationSolver, mdp::Union{MDP, POMDP},
                      policy::ValueIterationPolicy;
                      verbose::Bool=false)
    # gauss-seidel does not check tolerance
    # always runs for max iterations

    ns = n_states(mdp)
    na = n_actions(mdp)

    max_iterations = solver.max_iterations
    n_procs  = solver.n_procs

    # number of chunks in each DP iteration
    n_chunks = length(solver.state_order)
    order   = solver.state_order

    # state chunks split between each worker
    chunks = chunk_ordering(n_procs, order)
    
    # create an ordered list of states for fast iteration
    verbose ? println("ordering states ...") : nothing
    states_ = ordered_states(mdp)
    verbose ? println("done ordering states") : nothing
    verbose ? flush(STDOUT) : nothing

    # shared utility function and Q-matrix
    util = SharedArray{Float64}(ns, init = S -> S[Base.localindexes(S)] = 0., pids=1:n_procs)
    q_mat  = SharedArray{Float64}((ns, na), init = S -> S[Base.localindexes(S)] = 0., pids=1:n_procs)
    pol = SharedArray{Int64}(ns, init = S -> S[Base.localindexes(S)] = 0., pids=1:n_procs)
    residual = SharedArray{Float64}(1, init = S -> S[Base.localindexes(S)] = 0., pids=1:n_procs)
    S = state_type(mdp)
    states = SharedArray{S}(ns, pids=1:n_procs)
    workers = WorkerPool(collect(1:n_procs))
    
    # init
    states[:] = states_[:] #XXX find a nicer way to initialize the shared array
    util[:] = policy.util[:]
    q_mat[:] = policy.qmat[:]
    pol[:] = policy.policy[:]

    iter_time  = 0.0
    total_time = 0.0

    for i = 1:max_iterations
        tic()
        residual[1] = 0.
        for c = 1:n_chunks
            state_indices = chunks[c]
            results = pmap(workers, 
                           x -> solve_chunk(mdp, states, util, pol, q_mat, solver.include_Q, residual, x), 
                           state_indices)
        end # chunk loop 

        iter_time = toq();
        total_time += iter_time
        verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual[1], iter_time*1000.0, total_time) : nothing
        verbose ? flush(STDOUT) : nothing
        residual[1] < solver.belres ? break : nothing
    end # main iteration loop
    return ValueIterationPolicy(mdp, q_mat, util, pol)
end

# updates shared utility and Q-Matrix using gauss-seidel value iteration (asynchronous)
function solve_chunk(mdp::M, 
                    states::SharedArray{S, 1}, 
                    util::SharedArray{Float64, 1}, 
                    pol::SharedArray{Int64, 1}, 
                    qmat::SharedArray{Float64, 2}, 
                    include_Q::Bool,
                    residual::SharedArray{Float64, 1},
                    state_indices::Tuple{Int64, Int64}
                    ) where {M <: Union{MDP, POMDP}, S}

    discount_factor = discount(mdp)
    for istate=state_indices[1]:state_indices[2]
        s = states[istate]
        sub_aspace = actions(mdp, s)
        if isterminal(mdp, s)
            util[istate] = 0.0
            pol[istate] = 1
        else
            old_util = util[istate] # for residual
            max_util = -Inf
            for a in iterator(sub_aspace)
                iaction = action_index(mdp, a)
                dist = transition(mdp, s, a) # creates distribution over neighbors
                u = 0.0
                for (sp, p) in weighted_iterator(dist)
                    p == 0.0 ? continue : nothing # skip if zero prob
                    r = reward(mdp, s, a, sp)
                    isp = state_index(mdp, sp)
                    u += p * (r + discount_factor * util[isp])
                end
                new_util = u
                if new_util > max_util
                    max_util = new_util
                    pol[istate] = iaction
                end
                include_Q ? (qmat[istate, iaction] = new_util) : nothing
            end # action
            util[istate] = max_util
            diff = abs(max_util - old_util)
            diff > residual[1] ? (residual[1] = diff) : nothing
            # update the value array
        end
    end # state loop
    return 
end

# returns an array of start and end indices for each chunk
function chunk_ordering(n_procs::Int64, order::Vector{Tuple{Int64, Int64}})
    n_chunks = length(order)
    # start and end indices
    chunks = Vector{Vector{Tuple{Int64, Int64}}}(n_chunks) 
    for i = 1:n_chunks
        co = order[i]
        start_idx = co[1]
        end_idx = co[2]
        n_states_per_chunk = end_idx - start_idx
        # divide the work among the processors
        stride = div(n_states_per_chunk, (n_procs-1))
        temp = Vector{Tuple{Int64, Int64}}()
        for j = 0:(n_procs-2)
            si = j * stride + start_idx
            ei = si + stride - 1
            if j == (n_procs-2) 
                ei = end_idx
            end
            push!(temp, (si ,ei))
        end
        chunks[i] = temp
    end
    return chunks
end

@POMDP_require solve(solver::ParallelValueIterationSolver, mdp::Union{MDP,POMDP}) begin
    vi_solver = ValueIterationSolver(solver.max_iterations, solver.belres)
    @subreq solve(vi_solver, mdp)
end