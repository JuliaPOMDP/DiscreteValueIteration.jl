
N_PROCS = 20

addprocs(N_PROCS-1)                  
@everywhere __PARALLEL__ = true   

using GridWorld_
using ValueIteration_
using ParallelValueIteration

function parallelTest(nProcs::Int, gridSize::Int; nIter::Int=1, nChunks::Int=1)

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

    vi = ValueIteration(nIter, tolerance)
    pvi = ParallelSolver(nProcs, order, nIter, tolerance, gauss_siedel_flag)

    @time q = solve(vi, mdp)
    @time qp = solve(pvi, mdp)
    return q, qp
end


function serialTime(gridSize::Int; nIter::Int=1)

    rPos  = Array[[8,9], [3,8], [5,4], [8,4]]
    rVals = [10.0, 3.0, -5.0, -10.0]

    mdp = GridWorldMDP(gridSize, gridSize, rPos, rVals)

    nStates  = mdp.nStates
    nActions = mdp.nActions

    vi = ValueIteration("vi_policy.pol", nIter, 1e-4, zeros(nStates), zeros(nActions, nStates))

    tic()
    q = solve(mdp, vi)
    t = toc()

    return t
end

function parallelTime(nProcs::Int, gridSize::Int; nIter::Int=1, nChunks=1)

    rPos  = Array[[8,9], [3,8], [5,4], [8,4]]
    rVals = [10.0, 3.0, -5.0, -10.0]

    mdp = GridWorldMDP(gridSize, gridSize, rPos, rVals)

    nStates  = mdp.nStates
    nActions = mdp.nActions

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

    pvi = ParallelVI("pvi_policy.pol", nIter, 1e-4, nProcs, zeros(nStates), zeros(nActions, nStates), order) 

    tic()
    qp = solve(mdp, pvi)
    t = toc()

    return t 
end


function speedTest(;nIter=2, procs=[2:20], gSizes=[250,500,750,1000,1500])
    
    np = length(procs)
    ng = length(gSizes)

    # dictionary
    #ptimes = Dict{(Int64, Int64), Float64}()
    #stimes = Dict{Int64, Float64}() 

    ptimes = zeros(ng, np) 
    stimes = zeros(ng) 

    for i = 1:ng
        g = gSizes[i]
        stimes[i] = serialTime(g, nIter = nIter)
        for j = 1:np
            p = procs[j]
            println(g, " ", p)
            ptimes[i,j] = parallelTime(p, g, nIter = nIter)
        end
    end
    return stimes, ptimes
end

function multiSpeedTest(nTests::Int; nIter=2, procs=[2:20], gSizes=[250,500,750,1000,1500])

    np = length(procs)
    ng = length(gSizes)
    
    ptimes = zeros(ng, np) 
    stimes = zeros(ng) 

    for i = 1:nTests
        st, pt = speedTest(nIter=nIter, procs=procs, gSizes=gSizes)
        if i == 1
            ptimes = pt
            stimes = st
        else
            # moving average
            stimes = stimes + (st - stimes) / i
            ptimes = ptimes + (pt - ptimes) / i
        end
    end
    return stimes, ptimes
end

