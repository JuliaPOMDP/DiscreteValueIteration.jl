"""
    ParallelValueIterationNotSharedSolver

Parallel asynchronous (Gauss-Seidel) Value Iteration solver. It allows to benefits from multiple CPUs to speed up Value Iteration.
# Fields:
- `max_iterations::Int64`: number of iterations to run VI, default=100
- `belres::Float64`: stop when the Bellman residual is lower than this value, default=1e-3
- `n_procs::Int64`: number of processors to use, default=`Sys.CPU_CORES` (maximum available on the machine)
- `include_Q::Bool`: if set to true, returns the state action values as well, default=true
- `state_order::Vector{Tuple{Int64, Int64}}`: provide a decomposition of the state space to treat serially. 
Each element of state_order is the start and end index of a chunk of states in the state space. Each chunk is solved in parallel.
"""
mutable struct ParallelValueIterationNotSharedSolver <: Solver
    max_iterations::Int64 
    belres::Float64 
    n_procs::Int64 
    verbose::Bool
    include_Q::Bool
    init_util::Vector{Float64}
    state_order::Vector{Tuple{Int64, Int64}}
end

function ParallelValueIterationNotSharedSolver(;max_iterations::Int64 = 100,
                                       belres::Float64 = 1e-3,
                                       n_procs::Int64 = Sys.CPU_CORES,
                                       verbose::Bool = true,        
                                       include_Q::Bool = true,
                                       init_util::Vector{Float64}=Vector{Float64}(0),
                                       state_order::Vector{Tuple{Int64, Int64}} = Tuple{Int64, Int64}[])
    return ParallelValueIterationNotSharedSolver(max_iterations, belres, n_procs, verbose, include_Q, init_util, state_order)
end

function solve(solver::ParallelValueIterationNotSharedSolver, mdp::Union{MDP,POMDP}; kwargs...)

    # deprecation warning - can be removed when Julia 1.0 is adopted
    if !isempty(kwargs)
        warn("Keyword args for solve(::ParallelValueIterationNotSharedSolver, ::MDP) are no longer supported. For verbose output, use the verbose option in the ValueIterationSolver")
    end

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
    solver.verbose ? println("Starting parallel Gauss Seidel Value Iteration with $n_procs cores") : nothing
    solver.verbose ? flush(STDOUT) : nothing
    gauss_seidel(solver, mdp)
end

function gauss_seidel(solver::ParallelValueIterationNotSharedSolver, mdp::Union{MDP, POMDP})

    max_iterations = solver.max_iterations
    n_procs = solver.n_procs
    verbose = solver.verbose

    # number of chunks in each DP iteration
    n_chunks = length(solver.state_order)
    order   = solver.state_order

    # state chunks split between each worker
    chunks = chunk_ordering(n_procs, order)
    
    # create an ordered list of states for fast iteration
    verbose ? println("ordering states ...") : nothing
    states = ordered_states(mdp)
    verbose ? println("done ordering states") : nothing
    verbose ? flush(STDOUT) : nothing

    # init shared utility function and Q-matrix    
    ns = n_states(mdp)
    na = n_actions(mdp)
    
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
    
    util = SharedArray{Float64}(ns, init = S -> S[localindexes(S)] = init_util[localindexes(S)], pids=1:n_procs)
    qmat  = SharedArray{Float64}((ns, na), init = S -> S[Base.localindexes(S)] = init_qmat[localindexes(S)], pids=1:n_procs)
    pol = SharedArray{Int64}(ns, init = S -> S[Base.localindexes(S)] = init_pol[localindexes(S)], pids=1:n_procs)
    residual = SharedArray{Float64}(1, init = S -> S[Base.localindexes(S)] = 0., pids=1:n_procs)
    workers = WorkerPool(collect(1:n_procs))

    iter_time  = 0.0
    total_time = 0.0

    for i = 1:max_iterations
        tic()
        residual[1] = 0.
        for c = 1:n_chunks
            state_indices = chunks[c]
            results = pmap(workers, 
                           x -> solve_chunk(mdp, states, util, pol, qmat, solver.include_Q, residual, x), 
                           state_indices)
        end # chunk loop 

        iter_time = toq();
        total_time += iter_time
        verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual[1], iter_time*1000.0, total_time) : nothing
        verbose ? flush(STDOUT) : nothing
        residual[1] < solver.belres ? break : nothing
    end # main iteration loop
    return ValueIterationPolicy(mdp, convert(Array{Float64, 2}, qmat), convert(Array{Float64, 1} , util), convert(Array{Int64, 1}, pol))
end

# updates shared utility and Q-Matrix using gauss-seidel value iteration (asynchronous)
function solve_chunk(mdp::M, 
                    states::Array{S, 1}, 
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

@POMDP_require solve(solver::ParallelValueIterationNotSharedSolver, mdp::Union{MDP,POMDP}) begin
    vi_solver = ValueIterationSolver(max_iterations=solver.max_iterations, 
                                     belres=solver.belres,
                                     include_Q=solver.include_Q,
                                     init_util=solver.init_util)
    @subreq solve(vi_solver, mdp)
end
