# The solver type
""" 
ParallelValueIterationSolver

    - max_iterations::Int64, the maximum number of iterations value iteration runs for (default 100)
    - belres::Float64, the Bellman residual (default 1e-3)
    - verbose::Bool, if set to true, the bellman residual and the time per iteration will be printed to STDOUT (default false)
    - include_Q::Bool, if set to true, the solver outputs the Q values in addition to the utility and the policy (default true)
    - init_util::Vector{Float64}, provides a custom initialization of the utility vector. (initializes utility to 0 by default)
    - gauss_seidel::Bool, if true, runs asynchronous value iteration (one copy of the utility vector)
""" 
mutable struct ParallelValueIterationSolver <: Solver
    max_iterations::Int64 # max number of iterations
    belres::Float64 # the Bellman Residual
    verbose::Bool 
    include_Q::Bool
    init_util::Vector{Float64}
    asynchronous::Bool
    n_procs::Int64
end
function ParallelValueIterationSolver(;max_iterations::Int64 = 100, 
                               belres::Float64 = 1e-3,
                               verbose::Bool = false,
                               include_Q::Bool = true,
                               init_util::Vector{Float64}=Vector{Float64}(undef, 0),
                               asynchronous::Bool = true,
                               n_procs::Int64 = nprocs() - 1)    
    return ParallelValueIterationSolver(max_iterations, belres, verbose, include_Q, init_util, asynchronous, n_procs)
end

function solve(solver::ParallelValueIterationSolver, mdp::Union{MDP,POMDP}; kwargs...)

    # deprecation warning - can be removed when Julia 1.0 is adopted
    if !isempty(kwargs)
        @warn("Keyword args for solve(::ParallelValueIterationSolver, ::MDP) are no longer supported. For verbose output, use the verbose option in the ParallelValueIterationSolver")
    end

    @warn_requirements solve(solver, mdp)

    solver.verbose ? println("Starting parallel Asynchronous Value Iteration with $(solver.n_procs) cores") : nothing
    solver.verbose ? flush(stdout) : nothing
    if solver.asynchronous
        policy = asynchronous_parallel_value_iteration(solver, mdp)
    else
        policy = synchronous_parallel_value_iteration(solver, mdp)
    end
    return policy
end

function synchronous_parallel_value_iteration(solver::ParallelValueIterationSolver, mdp::Union{MDP, POMDP})
    throw("ParallelValueIterationError: synchronous parallel value iteration is not supported yet")
end

function asynchronous_parallel_value_iteration(solver::ParallelValueIterationSolver, mdp::Union{MDP, POMDP})

    max_iterations = solver.max_iterations
    n_procs = solver.n_procs
    verbose = solver.verbose


    # init shared utility function and Q-matrix    
    ns = n_states(mdp)
    na = n_actions(mdp)

    state_chunks = split_states(ns, n_procs)
    
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
        qmat  = SharedArray{Float64}((ns, na), init = S -> S[localindices(S)] = init_qmat[localindices(S)])
    else
        qmat = nothing 
    end
    init_pol = zeros(Int64, ns)
    
    S = statetype(mdp)
    state_space = collect(states(mdp))
    shared_states = SharedArray{S}(ns, init = S -> S[localindices(S)] = state_space[localindices(S)])
    util = SharedArray{Float64}(ns, init = S -> S[localindices(S)] = init_util[localindices(S)])    
    pol = SharedArray{Int64}(ns, init = S -> S[localindices(S)] = init_pol[localindices(S)])
    residual = SharedArray{Float64}(1, init = S -> S[localindices(S)] .= 0.)
    pool = CachingPool(workers())
    iter_time  = 0.0
    total_time = 0.0
    for i = 1:max_iterations
        iter_time = @elapsed begin
            residual[1] = 0.
            results = pmap(x -> solve_chunk(mdp, shared_states, util, pol, qmat, include_Q, residual, x), pool, state_chunks)
        end
        total_time += iter_time
        verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual[1], iter_time*1000.0, total_time) : nothing
        verbose ? flush(stdout) : nothing
        residual[1] < solver.belres ? break : nothing
    end # main iteration loop
    if include_Q
        return ValueIterationPolicy(mdp, convert(Array{Float64, 2}, qmat), convert(Array{Float64, 1} , util), convert(Array{Int64, 1}, pol))
    else
        return ValueIterationPolicy(mdp, utility=convert(Array{Float64, 1} , util), policy=convert(Array{Int64, 1}, pol), include_Q=false)
    end
end

# updates shared utility and Q-Matrix using gauss-seidel value iteration (asynchronous)
function solve_chunk(mdp::M, 
                    states::SharedArray{S, 1}, 
                    util::SharedArray{Float64, 1}, 
                    pol::Union{Nothing, SharedArray{Int64, 1}}, 
                    qmat::SharedArray{Float64, 2}, 
                    include_Q::Bool,
                    residual::SharedArray{Float64, 1},
                    state_indices::UnitRange{Int64}
                    ) where {M <: Union{MDP, POMDP}, S}

    discount_factor = discount(mdp)
    for istate=state_indices
        s = states[istate]
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
        temp = Vector{Tuple{Int64, Int64}}(n_procs-1)
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

function split_states(ns::Int64, n_procs::Int64)
    state_chunks = Vector{UnitRange{Int64}}(undef, n_procs)
    stride = div(ns, n_procs)
    for i=0:n_procs-1
        si = i*stride + 1
        ei = (i + 1)*stride
        if i == n_procs-1
            ei = ns
        end
        state_chunks[i+1] = si:ei
    end 
    return state_chunks
end

@POMDP_require solve(solver::ParallelValueIterationSolver, mdp::Union{MDP,POMDP}) begin
    vi_solver = ValueIterationSolver(max_iterations=solver.max_iterations, 
                                     belres=solver.belres,
                                     include_Q=solver.include_Q,
                                     init_util=solver.init_util)
    @subreq solve(vi_solver, mdp)
end