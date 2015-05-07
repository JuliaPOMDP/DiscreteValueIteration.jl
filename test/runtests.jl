using ParallelValueIteration
using Base.Test

# write your own tests here
@test 1 == 1

qTest = readdlm("grid-world-20x20-Q-matrix.txt")



rPos  = Array{Int64,1}[[8,9], [3,8], [5,4], [8,4]]
rVals = [10.0, 3.0, -5.0, -10.0]

mdp = GridWorldMDP(gridSize, gridSize, rPos, rVals)

@test_approx_eq_eps q[2] qp[2] 1e-5


function gridWorldTest(nProcs::Int, gridSize::Int=20, rPos::Array, rVals::array;
                       nIter=1, nChunks::Int=1)

    rPos  = Array{Int64,1}[[8,9], [3,8], [5,4], [8,4]]
    rVals = [10.0, 3.0, -5.0, -10.0]

    mdp = GridWorldMDP(gridSize, gridSize, rPos, rVals)

    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    order = Array(Vector{Int}, nChunks)
    stride = int(nStates/nChunks)
    for i = 0:(nChunks-1)
        sIdx = i * stride + 1
        eIdx = sIdx + stride - 1
        if i == (nChunks-1) && eIdx != nStates
            eIdx = nStates
        end
        order[i+1] = [sIdx, eIdx] 

    end

    tolerance = 1e-3
    gauss_siedel_flag = true

    pvi = ParallelSolver(nProcs, order, nIter, tolerance, gauss_siedel_flag)

    @time qp = solve(pvi, mdp)
    return q, qp

end



rPos  = Array{Int64,1}[[8,9], [3,8], [5,4], [8,4]]
rVals = [10.0, 3.0, -5.0, -10.0]

nProcs = CPU_CORES - 1
