using DiscreteValueIteration
using POMDPModels
using POMDPs
using Base.Test

function serial_qtest(mdp::Union{MDP,POMDP}, file::AbstractString; niter::Int64=100, res::Float64=1e-3)
    qt = readdlm(file)
    solver = ValueIterationSolver(max_iterations=niter, belres=res)
    policy = create_policy(solver, mdp) 
    policy = solve(solver, mdp, policy, verbose=true)
    (q, u, p, am) = locals(policy)
    npolicy = ValueIterationPolicy(mdp, deepcopy(q))
    nnpolicy = ValueIterationPolicy(mdp, deepcopy(q), deepcopy(u), deepcopy(p))
    s = GridWorldState(1,1)
    a1 = action(policy, s)
    v1 = value(policy, s)
    a2 = action(npolicy, s)
    v2 = value(npolicy, s)
    @test_approx_eq_eps qt q 1e-5
    @test policy.policy == nnpolicy.policy
    return true
end


rstates = [GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)]
rvals = [-10.0, -5.0, 10.0, 3.0]
xs = 10
ys = 10
mdp = GridWorld(sx=xs, sy=ys, rs = rstates, rv = rvals)
file = "grid-world-10x10-Q-matrix.txt"
niter = 100
res = 1e-3

@test serial_qtest(mdp, file, niter=niter, res=res) == true
println("Finished serial tests")

println("Testing Requirements")
@requirements_info ValueIterationSolver() mdp
