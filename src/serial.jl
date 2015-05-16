# implements vanilla value iteration solver

type SerialSolver <: Solver

    maxIterations::Int64

    tolerance::Float64

    gaussSiedel::Bool

    function SerialSolver(;maxIterations::Int64=1000, tolerance::Int64=1e-3, gaussSiedel::Bool=true)
        return SerialSolver(maxIterations, tolerance, gaussSiedel)
    end
end


function solve(solver::SerialSolver, mdp::DiscreteMDP; verbose::Bool=false)

    gs = solver.gaussSiedel
    gs ? (u, q = solveGS(solver, mdp, verbose=verbose)) : (u, q = solveRegular(solver, mdp, verbose=verbose))

end


function solveGS(solver::SerialSolver, mdp::DiscreteMDP; verbose::Bool=false)
    
    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    maxIter = alg.maxIterations
    tol     = alg.tolerance

    valU = zeros(nStates)
    valQ = zeros(nActions, nStates)

    totalTime = 0.0
    iterTime  = 0.0

    for i = 1:maxIter
        tic()
        for si = 1:nStates
            qHi = 0.0

            for ai = 1:nActions
                states, probs = nextStates(mdp, si, ai)
                qNow = reward(mdp, si, ai)

                for sp = 1:length(states)
                    spi = states[sp]
                    qNow += probs[sp] * valU[spi]
                end # sp loop
                valQ[ai, si] = qNow
                if ai == 1 || qNow > qHi
                    qHi = qNow
                    valU[si] = qHi
                end
            end # action loop
        end # state loop

        iterTime = toq();
        totalTime += iterTime
        verbose ? (println("Iteration : $i, iteration run-time: $iterTime, total run-time: $totalTime")) : nothing
    end # main loop

    return valU, valQ

end


function solveRegular(solver::SerialSolver, mdp::DiscreteMDP; verbose::Bool=false)
    
    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    maxIter = alg.maxIterations
    tol     = alg.tolerance

    valOld = zeros(nStates)
    valNew = zeros(nStates)
    valQ = zeros(nActions, nStates)

    totalTime = 0.0
    iterTime  = 0.0

    residual = 0.0

    for i = 1:maxIter
        tic()
        for si = 1:nStates
            qHi = 0.0

            for ai = 1:nActions
                states, probs = nextStates(mdp, si, ai)
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
            newResidual = (valOld[si] - valNew[si])^2 
            valOld[si] = valNew[si]
        end # state loop

        iterTime = toq();
        totalTime += iterTime
        verbose ? (println("Iteration : $i, iteration run-time: $iterTime, total run-time: $totalTime")) : nothing
        
        # terminate if tolerance value is reached
        if residual < tol; lastIdx = uIdx; break; end
    end # main loop

    return valNew, valQ
end

