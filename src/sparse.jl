struct SparseValueIterationSolver <: Solver
    max_iterations::Int64
    belres::Float64 # the Bellman Residual
    include_Q::Bool
    verbose::Bool
    init_util::Vector{Float64}
end

function SparseValueIterationSolver(;max_iterations=500,
                                    belres::Float64=1e-3,
                                    include_Q::Bool=true,
                                    verbose::Bool=false,
                                    init_util::Vector{Float64}=Vector{Float64}(undef, 0))
    return SparseValueIterationSolver(max_iterations, belres, include_Q, verbose, init_util)
end

@POMDP_require solve(solver::SparseValueIterationSolver, mdp::MDP) begin
    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req discount(::P)
    @subreq ordered_states(mdp)
    @subreq ordered_actions(mdp)
    @req transition(::P,::S,::A)
    @req reward(::P,::S,::A,::S)
    @req stateindex(::P,::S)
    @req actionindex(::P, ::A)
    @req actions(::P, ::S)
    as = actions(mdp)
    ss = states(mdp)
    @req length(::typeof(ss))
    @req length(::typeof(as))
    a = first(as)
    s = first(ss)
    dist = transition(mdp, s, a)
    D = typeof(dist)
    @req support(::D)
    @req pdf(::D,::S)
    @subreq SparseTabularMDP(mdp)
end

function qvalue!(m::Union{MDP,POMDP}, transition_A_S_S2, reward_S_A::AbstractMatrix{F}, value_S::AbstractVector{F}, out_qvals_S_A) where {F}
    @assert size(out_qvals_S_A) == (length(states(m)), length(actions(m)))
    for a in 1:length(actions(m))
        out_qvals_S_A[:, a] = view(reward_S_A, :, a) + discount(m) * transition_A_S_S2[a] * value_S
    end
end

function qvalue!(m::MDP, transition_A_S_S2, reward_S_A, value_S, out_qvals_S_A, _mul_cache)
    @assert size(out_qvals_S_A) == (length(states(m)), length(actions(m)))
    γ = discount(m)
    for a in 1:length(actions(m))
        Vp = mul!(_mul_cache, transition_A_S_S2[a], value_S)
        out_qvals_S_A[:, a] .= view(reward_S_A, :, a) .+ γ .* Vp
    end
end

function _value!(V, q_vals_S_A)
    δ_max = 0.0
    for i ∈ 1:size(q_vals_S_A, 1)
        vp = maximum(@view q_vals_S_A[i,:])
        δ = abs(V[i] - vp)
        δ > δ_max && (δ_max = δ)
        V[i] = vp
    end
    return δ_max
end

function solve(solver::SparseValueIterationSolver, mdp::SparseTabularMDP)
    nS = length(states(mdp))
    nA = length(actions(mdp))
    if isempty(solver.init_util)
        v_S = zeros(nS)
    else
        @assert length(solver.init_util) == nS "Input utility dimension mismatch"
        v_S = solver.init_util
    end
    _mul_cache = similar(v_S)

    transition_A_S_S2 = transition_matrices(mdp)
    reward_S_A = reward_matrix(mdp)
    qvals_S_A = zeros(nS, nA)
    maxchanges_T = zeros(solver.max_iterations)

    total_time = 0.0
    for i in 1:solver.max_iterations
        iter_time = @elapsed begin
            qvalue!(mdp, transition_A_S_S2, reward_S_A, v_S, qvals_S_A, _mul_cache)
            maxchanges_T[i] = _value!(v_S, qvals_S_A)
        end
        total_time += iter_time
        if solver.verbose
            @info "residual: $(maxchanges_T[i]), time: $(iter_time), total time: $(total_time) " i
        end
        maxchanges_T[i] < solver.belres && break
    end
    qvalue!(mdp, transition_A_S_S2, reward_S_A, v_S, qvals_S_A)
    # Rounding to avoid floating point error noise
    policy_S = dropdims(getindex.(argmax(round.(qvals_S_A, digits=20), dims=2), 2), dims=2)

    return if solver.include_Q
        ValueIterationPolicy(mdp, qvals_S_A, v_S, policy_S)
    else
        ValueIterationPolicy(mdp, utility=v_S, policy=policy_S, include_Q=false)
    end
end

function solve(solver::SparseValueIterationSolver, mdp::MDP)
    p = solve(solver, SparseTabularMDP(mdp))
    return ValueIterationPolicy(p.qmat, p.util, p.policy, ordered_actions(mdp), p.include_Q, mdp)
end

function solve(::SparseValueIterationSolver, ::POMDP)
    throw("""
           ValueIterationError: `solve(::SparseValueIterationSolver, ::POMDP)` is not supported,
          `SparseValueIterationSolver` supports MDP models only, look at QMDP.jl for a POMDP solver that assumes full observability.
           If you still wish to use the transition and reward from your POMDP model you can use the `UnderlyingMDP` wrapper from POMDPModelTools.jl as follows:
           ```
           solver = ValueIterationSolver()
           mdp = UnderlyingMDP(pomdp)
           solve(solver, mdp)
           ```
           """)
end
