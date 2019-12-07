using Revise
using Distributed
addprocs(2)
@everywhere using DiscreteValueIteration, POMDPs, POMDPModels
@everywhere begin 
    function DiscreteValueIteration.ind2state(mdp::SimpleGridWorld, si) 
        if si == length(states(mdp))
            return GWPos(-1,-1)
        end
        GWPos(CartesianIndices(mdp.size)[si][1],CartesianIndices(mdp.size)[si][2])
    end
end


mdp = SimpleGridWorld(size=(500,500))



solver = ParallelValueIterationSolver(n_procs=3, verbose=true)

parallel_policy = solve(solver, mdp)

serial_policy = solve(ValueIterationSolver(verbose=true), mdp)

@test isapprox(norm(parallel_policy.util - serial_policy.util), 0., atol=solver.belres)
