"""
    ParallelSynchronousValueIterationSolver

Parallel asynchronous (Gauss-Seidel) Value Iteration solver. It allows to benefits from multiple CPUs to speed up Value Iteration.
# Fields:
- `max_iterations::Int64`: number of iterations to run VI, default=100
- `belres::Float64`: stop when the Bellman residual is lower than this value, default=1e-3
- `n_procs::Int64`: number of processors to use, default=`Sys.CPU_CORES` (maximum available on the machine)
- `include_Q::Bool`: if set to true, returns the state action values as well, default=true
- `state_order::Vector{Tuple{Int64, Int64}}`: provide a decomposition of the state space to treat serially. 
Each element of state_order is the start and end index of a chunk of states in the state space. Each chunk is solved in parallel.
"""
mutable struct ParallelSynchronousValueIterationSolver <: Solver
    max_iterations::Int64 
    belres::Float64 
    n_procs::Int64 
    verbose::Bool
    include_Q::Bool
    init_util::Vector{Float64}
end

function ParallelSynchronousValueIterationSolver(;max_iterations::Int64 = 100,
                                       belres::Float64 = 1e-3,
                                       n_procs::Int64 = Sys.CPU_CORES,
                                       verbose::Bool = true,        
                                       include_Q::Bool = true,
                                       init_util::Vector{Float64}=Vector{Float64}(0))
    return ParallelSynchronousValueIterationSolver(max_iterations, belres, n_procs, verbose, include_Q, init_util)
end

function solve(solver::ParallelSynchronousValueIterationSolver, mdp::Union{MDP,POMDP}; kwargs...)

    # deprecation warning - can be removed when Julia 1.0 is adopted
    if !isempty(kwargs)
        warn("Keyword args for solve(::ParallelSynchronousValueIterationSolver, ::MDP) are no longer supported. For verbose output, use the verbose option in the ValueIterationSolver")
    end

    @warn_requirements solve(solver, mdp)
        
    # processor restriction checks    
    n_procs   = solver.n_procs
    n_procs < 2 ? error("Less than 2 processors not allowed") : nothing
    n_procs > Sys.CPU_CORES ? error("Requested too many processors") : nothing
   
    solver.verbose ? println("Starting parallel synchronous Value Iteration with $n_procs cores") : nothing
    solver.verbose ? flush(STDOUT) : nothing
    synchronous_value_iteration(solver, mdp)
end

function synchronous_value_iteration(solver::ParallelSynchronousValueIterationSolver, mdp::Union{MDP, POMDP})

    max_iterations = solver.max_iterations
    n_procs = solver.n_procs
    verbose = solver.verbose

    # create an ordered list of states for fast iteration
    verbose ? println("ordering states ...") : nothing
    states = ordered_states(mdp)
    states_chunks = split_states(n_procs, states)
    # for s in states_chunks; println(length(s)); end
    # println(sum(length(s) for s in states_chunks))
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
    
    workers = WorkerPool(collect(1:n_procs))

    iter_time  = 0.0
    total_time = 0.0
    old_util = init_util
    for i = 1:max_iterations
        tic()
        residual = 0.
        util_chunks = pmap(workers, x -> solve_chunk(mdp, x, old_util), states_chunks)
        new_util = vcat(util_chunks...)
        residual = norm(old_util - new_util, Inf)
        old_util = new_util
        iter_time = toq();
        total_time += iter_time
        verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
        verbose ? flush(STDOUT) : nothing
        residual < solver.belres ? break : nothing
    end # main iteration loop
    # return ValueIterationPolicy(mdp, convert(Array{Float64, 2}, qmat), convert(Array{Float64, 1} , util), convert(Array{Int64, 1}, pol))
end

# updates shared utility and Q-Matrix using gauss-seidel value iteration (asynchronous)
function solve_chunk(mdp::M, 
                    states_chunk::Array{S, 1}, 
                    old_util::Array{Float64, 1}
                    ) where {M <: Union{MDP, POMDP}, S}
    util = zeros(length(states_chunk))
    discount_factor = discount(mdp)
    for (i, s) in enumerate(states_chunk)
        sub_aspace = actions(mdp, s)
        if isterminal(mdp, s)
            util[i] = 0.0
        else
            max_util = -Inf
            for a in iterator(sub_aspace)
                iaction = action_index(mdp, a)
                dist = transition(mdp, s, a) # creates distribution over neighbors
                u = 0.0
                for (sp, p) in weighted_iterator(dist)
                    p == 0.0 ? continue : nothing # skip if zero prob
                    r = reward(mdp, s, a, sp)
                    isp = state_index(mdp, sp)
                    u += p * (r + discount_factor * old_util[isp])
                end
                new_util = u
                if new_util > max_util
                    max_util = new_util
                end
            end # action
            util[i] = max_util
        end
    end # state loop
    return util
end

# returns an array of start and end indices for each chunk
function split_states(n_procs::Int64, states::Vector{S}) where S 
    stride = div(length(states), n_procs)
    states_chunks = Vector{Vector{S}}(n_procs)
    for i=0:n_procs-1
        si = i*stride + 1
        ei = (i + 1)*stride
        if i == n_procs-1
            ei = length(states)
        end
        states_chunks[i+1] = states[si:ei]
    end 
    return states_chunks
end


@POMDP_require solve(solver::ParallelSynchronousValueIterationSolver, mdp::Union{MDP,POMDP}) begin
    vi_solver = ValueIterationSolver(max_iterations=solver.max_iterations, 
                                     belres=solver.belres,
                                     include_Q=solver.include_Q,
                                     init_util=solver.init_util)
    @subreq solve(vi_solver, mdp)
end
