function test_sparse_vanilla_same(m)
    vanilla_solver = ValueIterationSolver(max_iterations=500, belres=1e-8)
    sparse_solver = SparseValueIterationSolver(max_iterations=500, belres=1e-8)
    vanilla_policy = solve(vanilla_solver, m)
    sparse_policy = solve(sparse_solver, m)
    @test vanilla_policy.policy == sparse_policy.policy
    @test isapprox(vanilla_policy.util, sparse_policy.util; atol=1e-3)
    for s in states(m)
        si = stateindex(m, s)
        for a in actions(m, s)
            ai = actionindex(m, a)
            @test isapprox(vanilla_policy.qmat[si, ai], sparse_policy.qmat[si, ai]; atol=1e-3)
        end
    end
end

m1 = SimpleGridWorld(size=(10,10))
test_sparse_vanilla_same(m1)

m2 = SpecialGridWorld()
test_sparse_vanilla_same(m2)

## Two state MDP to test for terminal states

struct TwoStatesMDP <: MDP{Int, Int} end

POMDPs.n_states(mdp::TwoStatesMDP) = 2
POMDPs.states(mdp::TwoStatesMDP) = 1:2
POMDPs.stateindex(mdp::TwoStatesMDP, s) = s 
POMDPs.n_actions(mdp::TwoStatesMDP) = 2
POMDPs.actions(mdp::TwoStatesMDP) = 1:2
POMDPs.actionindex(mdp::TwoStatesMDP, a) = a
POMDPs.discount(mdp::TwoStatesMDP) = 0.95
POMDPs.transition(mdp::TwoStatesMDP, s, a) = SparseCat([a], [1.0])
POMDPs.reward(mdp::TwoStatesMDP, s, a, sp) = float(sp == 2)
POMDPs.isterminal(mdp::TwoStatesMDP, s) = s == 2


mdp = TwoStatesMDP()
solver = ValueIterationSolver(verbose = true)
policy = solve(solver, mdp)

sparsesolver = SparseValueIterationSolver(verbose=true)
sparsepolicy = solve(sparsesolver, mdp)

@test sparsepolicy.qmat == policy.qmat 
@test value(sparsepolicy, 2) ≈ 0.0
@test_throws String solve(SparseValueIterationSolver(), TigerPOMDP())

@inferred solve(SparseValueIterationSolver(verbose=false), mdp)
