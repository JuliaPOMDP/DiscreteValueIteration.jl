

module ValueIteration_

export ValueIteration
export solve


using DiscreteMDPs
using GridWorld_


import DiscreteMDPs.solve
import DiscreteMDPs.Solver


type ValueIteration <: Solver

    maxIterations::Int

    tolerance::Float64

end


function solve(alg::ValueIteration, mdp::DiscreteMDP)

    nStates  = numStates(mdp)
    nActions = numActions(mdp)

    maxIter = alg.maxIterations
    tol     = alg.tolerance

    valU = zeros(nStates)
    valQ = zeros(nActions, nStates)

    for i = 1:maxIter
        residual = 0.0
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
    end # main loop

    return valU, valQ
end


end # module
