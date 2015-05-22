addprocs(int(CPU_CORES)-1)

using DiscreteValueIteration
using GridWorlds
using Base.Test


function parallelGridWorldTest(nProcs::Int, gridSize::Int, 
                       rPos::Array, rVals::Array, file::String;
                       nIter::Int=100, nChunks::Int=1)

    qt = readdlm(file)

    mdp = GridWorldMDP(gridSize, gridSize, rPos, rVals)

    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    order = {}
    stride = int(nStates/nChunks)
    for i = 0:(nChunks-1)
        sIdx = i * stride + 1
        eIdx = sIdx + stride - 1
        if i == (nChunks-1) && eIdx != nStates
            eIdx = nStates
        end
        push!(order, sIdx:eIdx)
    end

    tolerance = 1e-10

    # test gauss-siedel
    gauss_siedel_flag = true
    pvi = ParallelSolver(nProcs, stateOrder=order, maxIterations=nIter, 
                         tolerance=tolerance, gaussSiedel=gauss_siedel_flag)
    policy = solve(pvi, mdp)
    qp = policy.Q

    @test_approx_eq_eps qt qp 1.0


    # test regualr
    gauss_siedel_flag = false
    pvi = ParallelSolver(nProcs, stateOrder=order, maxIterations=nIter, 
                         tolerance=tolerance, gaussSiedel=gauss_siedel_flag)
    policy = solve(pvi, mdp)
    pq = policy.Q

    @test_approx_eq_eps qt qp 1.0

    return true
end


function serialGridWorldTest(gridSize::Int64, rPos::Array, rVals::Array, file::String; nIter::Int64=100)

    qt = readdlm(file)

    mdp = GridWorldMDP(gridSize, gridSize, rPos, rVals)

    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    tolerance = 1e-10

    gauss_siedel_flag = true
    vi = SerialSolver(maxIterations=nIter, tolerance=tolerance, gaussSiedel=gauss_siedel_flag)
    policy = solve(vi, mdp)
    q = policy.Q

    @test_approx_eq_eps qt q 1.0

    return true
end


rPos     = Array{Int64,1}[[8,9], [3,8], [5,4], [8,4]]
rVals    = [10.0, 3.0, -5.0, -10.0]
nProcs   = int(CPU_CORES/2)
file     = "grid-world-10x10-Q-matrix.txt"
gridSize = 10

# run parallel tests only on multi-core machines
if (CPU_CORES > 1)
    @test parallelGridWorldTest(nProcs, gridSize, rPos, rVals, file) == true
    @test parallelGridWorldTest(nProcs, gridSize, rPos, rVals, file, nChunks=2) == true
    println("Finished parallel tests")
end
@test serialGridWorldTest(gridSize, rPos, rVals, file) == true
println("Finished serial tests")
