using DiscreteValueIteration
using POMDPModels
using POMDPs
using Base.Test


function serial_test(mdp::POMDP, file::String; niter::Int64=100, res::Float64=1e-3)
    qt = readdlm(file)
    solver = ValueIterationSolver(max_iterations=niter, belres=res)
    policy = ValueIterationPolicy(mdp)
    solve(solver, mdp, policy, verbose=true)
    q = policy.qmat
    @test_approx_eq_eps qt q 1.0
    return true
end


rstates = [GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)]
rvals = [-10.0, -5.0, 10.0, 3.0]
xs = 10
ys = 10
mdp = GridWorld(xs, ys, rs = rstates, rv = rvals)
file = "grid-world-10x10-Q-matrix.txt"
niter = 100
res = 1e-3

@test serial_test(mdp, file, niter=niter, res=res) == true
println("Finished serial tests")
