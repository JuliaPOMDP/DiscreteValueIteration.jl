"""
    ParallelValueIterationSolver
Parallel asynchronous (Gauss-Seidel) Value Iteration solver. It allows to benefits from multiple CPUs to speed up Value Iteration.
# Fields:
- `max_iterations::Int64`: number of iterations to run VI, default=100
- `belres::Float64`: stop when the Bellman residual is lower than this value, default=1e-3
- `n_procs::Int64`: number of processors to use, default=`Sys.CPU_CORES` (maximum available on the machine)
- `include_Q::Bool`: if set to true, returns the state action values as well, default=true
- `state_order::Vector{Tuple{Int64, Int64}}`: provide a decomposition of the state space to treat serially. 
Each element of state_order is the start and end index of a chunk of states in the state space. Each chunk is solved in parallel.
"""
mutable struct ParallelValueIterationSolver <: Solver
    max_iterations::Int64 
    belres::Float64 
    n_procs::Int64 
    verbose::Bool
    include_Q::Bool
    init_util::Vector{Float64}
    state_order::Vector{Tuple{Int64, Int64}}
end

function ParallelValueIterationSolver(;max_iterations::Int64 = 100,
                                       belres::Float64 = 1e-3,
                                       n_procs::Int64 = Sys.CPU_CORES,
                                       verbose::Bool = true,        
                                       include_Q::Bool = true,
                                       init_util::Vector{Float64}=Vector{Float64}(undef, 0),
                                       state_order::Vector{Tuple{Int64, Int64}} = Tuple{Int64, Int64}[])
    return ParallelValueIterationSolver(max_iterations, belres, n_procs, verbose, include_Q, init_util, state_order)
end

function ind2state end

function solve(solver::ParallelValueIterationSolver, mdp::Union{MDP,POMDP})

    # check if a default state ordering is needed
    order = solver.state_order
    if isempty(order)
        ns = n_states(mdp)
        solver.state_order = [(1,ns)]
    end
    solver.verbose ? println("Starting parallel Gauss Seidel Value Iteration with $(solver.n_procs) cores") : nothing
    solver.verbose ? flush(stdout) : nothing
    gauss_seidel(solver, mdp)
end

function gauss_seidel(solver::ParallelValueIterationSolver, mdp::Union{MDP, POMDP})

    max_iterations = solver.max_iterations
    n_procs = solver.n_procs
    verbose = solver.verbose

    # number of chunks in each DP iteration
    n_chunks = length(solver.state_order)
    order   = solver.state_order

    # state chunks split between each worker
    chunks = chunk_ordering(n_procs, order)
    
    # create an ordered list of states for fast iteration
    # verbose ? println("ordering states ...") : nothing
    # states_ = ordered_states(mdp)
    # verbose ? println("done ordering states") : nothing
    # verbose ? flush(STDOUT) : nothing



    # init shared utility function and Q-matrix    
    ns = n_states(mdp)
    na = length(actions(mdp))
    
    # intialize the utility and Q-matrix
    if !isempty(solver.init_util)
        @assert length(solver.init_util) == ns "Input utility dimension mismatch"
        init_util = solver.init_util
    else
        init_util = zeros(ns)
    end
    include_Q = solver.include_Q
    if include_Q
        init_qmat = zeros(ns, na)
    end
    init_pol = zeros(Int64, ns)
    qmat = init_qmat
    # shared_array_file_root = "~/tmp/"
    util = SharedArray{Float64}(ns, init = S -> S[localindices(S)] = init_util[localindices(S)])
    qmat  = SharedArray{Float64}((ns, na), init = S -> S[localindices(S)] = init_qmat[localindices(S)])
    pol = SharedArray{Int64}(ns, init = S -> S[localindices(S)] = init_pol[localindices(S)])
    residual = SharedArray{Float64}(1, init = S -> S[localindices(S)] .= 0.)
    # S = state_type(mdp)
    # states = SharedArray{S}("/scratch/boutonm/states.bin", (ns,), init = S -> S[localindices(S)] = states_[localindices(S)],  pids=collect(1:n_procs))
    # workers = WorkerPool(collect(1:n_procs))
    println("shared array initialized")
    flush(stdout)

    
    iter_time  = 0.0
    total_time = 0.0
    state_indices = chunks[1]
    for i = 1:max_iterations
        iter_time = @elapsed begin

        residual[1] = 0.
        state_indices = chunks[1]
        results = pmap(x -> solve_chunk(mdp, util, pol, residual, qmat, include_Q, x), state_indices)
        
        end
        total_time += iter_time
        verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual[1], iter_time*1000.0, total_time) : nothing
        verbose ? flush(stdout) : nothing
        # residual[1] < solver.belres ? break : nothing
    end # main iteration loop
    return ValueIterationPolicy(mdp, convert(Array{Float64, 2}, qmat), convert(Array{Float64, 1} , util), convert(Array{Int64, 1}, pol))
end

# updates shared utility and Q-Matrix using gauss-seidel value iteration (asynchronous)
function solve_chunk(mdp::M,
                    util::SharedArray{Float64, 1}, 
                    pol::SharedArray{Int64, 1}, 
                    residual::SharedArray{Float64, 1},
                    qmat::SharedArray{Float64, 2},
                    include_Q::Bool,
                    state_indices::Tuple{Int64, Int64}
                    ) where {M <: Union{MDP, POMDP}}

    discount_factor = discount(mdp)
    println("Working on states $(state_indices[1]) to $(state_indices[2])")
    flush(stdout)
    worker_time = @elapsed begin
    for istate=state_indices[1]:state_indices[2]
        s = ind2state(mdp, istate)
        # s = states[istate]
        sub_aspace = actions(mdp, s)
        if isterminal(mdp, s)
            util[istate] = 0.0
            pol[istate] = 1
        else
            old_util = util[istate] # for residual
            max_util = -Inf
            for a in sub_aspace
                iaction = actionindex(mdp, a)
                dist = transition(mdp, s, a) # creates distribution over neighbors
                u = 0.0
                for (sp, p) in weighted_iterator(dist)
                    p == 0.0 ? continue : nothing # skip if zero prob
                    r = reward(mdp, s, a, sp)
                    isp = stateindex(mdp, sp)
                    u += p * (r + discount_factor * util[isp])
                end
                new_util = u
                if new_util > max_util
                    max_util = new_util
                    pol[istate] = iaction
                end
                include_Q ? (qmat[istate, iaction] = new_util) : nothing
            end # action
            # update the value array
            util[istate] = max_util
            diff = abs(max_util - old_util)
            diff > residual[1] ? (residual[1] = diff) : nothing
        end
    end # state loop
    end # @elapsed
    println("Done $worker_time")
    flush(stdout)
    return 
end

# returns an array of start and end indices for each chunk
function chunk_ordering(n_procs::Int64, order::Vector{Tuple{Int64, Int64}})
    n_chunks = length(order)
    # start and end indices
    chunks = Vector{Vector{Tuple{Int64, Int64}}}(undef, n_chunks) 
    for i = 1:n_chunks
        co = order[i]
        start_idx = co[1]
        end_idx = co[2]
        n_states_per_chunk = end_idx - start_idx
        # divide the work among the processors
        stride = div(n_states_per_chunk, (n_procs-1))
        temp = Vector{Tuple{Int64, Int64}}(undef, n_procs-1)
        for j = 0:(n_procs-2)
            si = j * stride + start_idx
            ei = si + stride - 1
            if j == (n_procs-2) 
                ei = end_idx
            end
            temp[j+1] =  (si ,ei)
        end
        chunks[i] = temp
    end
    return chunks
end

@POMDP_require solve(solver::ParallelValueIterationSolver, mdp::Union{MDP,POMDP}) begin
    vi_solver = ValueIterationSolver(max_iterations=solver.max_iterations, 
                                     belres=solver.belres,
                                     include_Q=solver.include_Q,
                                     init_util=solver.init_util)
    @subreq solve(vi_solver, mdp)
end
