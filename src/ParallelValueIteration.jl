# multi-core, single machine parallel value iteration solver

module ParallelValueIteration

export ParallelValueIteration
export solve


using DiscreteMDP

import DiscreteMDP.Solver
import DiscreteMDP.solve


type ParallelSolver <: Solver

    numProcessors::Int64 # number of processors to use

    stateOrder::Vector # order for processing chunks at each iter, e.g. [[1,1000],[1000,2000]]  

    maxIteratons::Int64 # number of iterations

    tolerance::Float64 # Bellman residual

    gaussSeidel::Bool # flag for gauss-seidel value iteration (regular uses x2 more memory)

end


# returns the utility function and the Q-matrix
function solve(solver::ParallelValueIteration, mdp::DiscreteMDP)
    # check gauss-seidel flag
    gs = solver.gaussSeidel
    gs ? return solveGS(solver, mdp) : return solveRegular(solver, mdp)
end


function solveGS(solver::ParallelValueIteration, mdp::DiscreteMDP)
    # gauss-seidel does not check tolerance
    # always runs for max iterations

    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    maxIter = solver.maxIteratons

    nProcs  = solver.numProcessors

    # number of chunks in each DP iteration
    nChunks = length(solver.stateOrder)
    order   = solver.stateOrder

    # state chunks split between each worker
    chunks = chunkOrdering(nProcs, order)

    # shared utility function and Q-matrix
    util = SharedArray(Float64, (nStates), init = S -> S[localindexes(S)] = 0.0, pids = [1:nProcs])
    valQ  = SharedArray(Float64, (nActions, nStates), init = S -> S[localindexes(S)] = 0.0, pids = [1:nProcs])

    for i = 1:maxIter
        for c = 1:nChunks
            lst = chunks[c]

            tic()
            # no residuals, so results are 0.0
            results = pmap(x -> (idxs = x; solveChunk(mdp, util, valQ, idxs)), lst)
            t = toc();
            println("Chunk: $c, time: $t")
           
        end # chunk loop 
    end # main iteration loop
    return util, valQ
end


function solveRegular(solver::ParallelValueIteration, mdp::DiscreteMDP)
    
    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    maxIter = solver.maxIteratons
    tol     = solver.tolerance

    nProcs  = solver.numProcessors

    # number of chunks in each DP iteration
    nChunks = length(solver.stateOrder)
    order   = solver.stateOrder

    # state chunks split between each worker
    chunks = chunkOrdering(nProcs, order)

    # shared utility function and Q-matrix
    util1 = SharedArray(Float64, (nStates), init = S -> S[localindexes(S)] = 0.0, pids = [1:nProcs])
    util2 = SharedArray(Float64, (nStates), init = S -> S[localindexes(S)] = 0.0, pids = [1:nProcs])
    valQ  = SharedArray(Float64, (nActions, nStates), init = S -> S[localindexes(S)] = 0.0, pids = [1:nProcs])

    uCount = 0
    lastIdx = 1
    for i = 1:maxIter
        # residual tolerance
        residual = 0.0
        for c = 1:nChunks
            # util array to update: 1 or 2
            uIdx = uCount % 2 + 1
            lst = chunks[c]

            tic()
            if uIdx1 == 1
                # returns the residual
                results = pmap(x -> (idxs = x; solveChunk(mdp, util1, util2, valQ, idxs)), lst)
                residual += results
            else
                # returns the residual
                results = pmap(x -> (idxs = x; solveChunk(mdp, util2, util1, valQ, idxs)), lst)
                residual += results
            end
            t = toc();
            println("Chunk: $c, time: $t")
           
            uCount += 1
        end # chunk loop 
        # terminate if tolerance value is reached
        if residual < tol; lastIdx = uIdx; break; end
    end # main iteration loop
    lastIdx == 1 ? return util2, valQ : return util1, valQ
end


# updates the shared array utility and returns the residual
# valOld is used to update, and valNew is the updated value function
function solveChunk(mdp::DiscreteMDP, valOld::SharedArray, valNew::SharedArray, valQ::SharedArray, stateIndices::(Int64, Int64))

    sStart = stateIndices[1]
    sEnd   = stateIndices[2]
    nActions = mdp.nActions

    residual = 0.0

    for si = sStart:sEnd
        qHi = -Inf

        for ai = 1:nActions
            probs, states = nextStates(mdp, si, ai)
            qNow = reward(mdp, si, ai)

            for sp = 1:length(states)
                spi = states[sp]
                qNow += probs[sp] * valOld[spi] 
            end # sp loop
            valQ[ai, si] = qNow
            if ai == 1 || qNow > qHi
                qHi = qNow
                valNew[si] = qHi
            end
        end # action loop
        residual += (valOld[si] - valNew[si])^2
    end # state loop
    return residual 
end


# updates shared utility and Q-Matrix for gauss-sidel value iteration
function solveChunk(mdp::DiscreteMDP, util::SharedArray, valQ::SharedArray, stateIndices::(Int64, Int64))

    sStart = stateIndices[1]
    sEnd   = stateIndices[2]
    nActions = mdp.nActions

    # no residual for gauss-siedel
    residual = 0.0

    for si = sStart:sEnd
        qHi = -Inf

        for ai = 1:nActions
            probs, states = nextStates(mdp, si, ai)
            qNow = reward(mdp, si, ai)

            for sp = 1:length(states)
                spi = states[sp]
                qNow += probs[sp] * util[spi] 
            end # sp loop
            valQ[ai, si] = qNow
            if ai == 1 || qNow > qHi
                qHi = qNow
                util[si] = qHi
            end
        end # action loop
    end # state loop
    return residual 
end


# returns an array of start and end indices for each chunk
function chunkOrdering(nProcs::Int64, order::Vector)
    nChunks = length(order)
    # start and end indices
    chunks = Array(Vector{(Int64, Int64)}, nChunks) 
    for i = 1:nChunks
        co = order[i]
        sIdx = co[1]
        eIdx = co[2]
        ns = eIdx - sIdx
        # divide the work among the processors
        stride = int(ns / (nProcs-1))
        temp = (Int64, Int64)[]
        for j = 0:(nProcs-2)
            si = j * stride + sIdx
            ei = si + stride - 1
            if j == (nProcs-2) 
                ei = eIdx
            end
            push!(temp, (si ,ei))
        end
        chunks[i] = temp
    end
    return chunks
end


end # module


