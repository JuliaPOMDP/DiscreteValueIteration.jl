function test_sparse_vanilla_same()
    gridworld = SimpleGridWorld(size=(10,10))
    vanilla_solver = ValueIterationSolver(max_iterations=500, belres=1e-8)
    sparse_solver = SparseValueIterationSolver(max_iterations=500, belres=1e-8)
    vanilla_policy = solve(vanilla_solver, gridworld)
    sparse_policy = solve(sparse_solver, gridworld)
    @test vanilla_policy.policy == sparse_policy.policy_S
    @test isapprox(vanilla_policy.util, sparse_policy.v_S; atol=1e-3)
    @test isapprox(vanilla_policy.qmat, sparse_policy.qvals_S_A; atol=1e-3)
end

test_sparse_vanilla_same()
