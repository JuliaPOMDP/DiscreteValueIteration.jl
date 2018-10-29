using POMDPs
using POMDPModels
using DiscreteValueIteration

using BenchmarkTools

function bench_sparsevi(size)
    world = SimpleGridWorld(size=size)
    solver = SparseValueIterationSolver(max_iterations=500, belres=1e-4)
    policy = solve(solver, world)
    return policy
end

function bench_vanillavi(size)
    world = SimpleGridWorld(size=size)
    solver = ValueIterationSolver(max_iterations=500, belres=1e-4)
    policy = solve(solver, world)
    return policy
end

for size in [(10,10), (100, 100), (1000, 1000), (10000, 10000)]
    @show size
    @info "Sparse"
    @btime bench_sparsevi($size)
    @info "Vanilla"
    @btime bench_vanillavi($size)
end
