# implements vanilla value iteration solver

type SerialSolver <: Solver

    maxIterations::Int64

    tolerance::Float64

    gaussSiedel::Bool

    includeV::Bool

    includeQ::Bool

    includeA::Bool

end


function SerialSolver(;maxIterations::Int64=1000, tolerance::Float64=1e-3, gaussSiedel::Bool=true,
                       includeV::Bool=true, includeQ::Bool=true, includeA::Bool=true)
    return SerialSolver(maxIterations, tolerance, gaussSiedel, includeV, includeQ, includeA)
end


function solve(solver::SerialSolver, mdp::DiscreteMDP; verbose::Bool=false)

    gs = solver.gaussSiedel
    u = []
    q = []

    if gs
        u, q = solveGS(solver, mdp, verbose=verbose)
    else
        u, q = solveRegular(solver, mdp, verbose=verbose)
    end

    p = []
    # check policy flag
    if solver.includeA
        p = computePolicy(mdp, u)
    end

    policy = DiscretePolicy(V=u, Q=q, P=p)

    return policy
end


function solveGS(solver::SerialSolver, mdp::DiscreteMDP; verbose::Bool=false)
    
    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    maxIter = solver.maxIterations
    tol     = solver.tolerance

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

