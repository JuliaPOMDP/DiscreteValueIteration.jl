module RandomMDPs


export RandomMDP
export states, actions
export numStates, numActions
export reward, nextStates 

using DiscreteMDPs

import DiscreteMDPs.DiscreteMDP
import DiscreteMDPs.reward
import DiscreteMDPs.nextStates
import DiscreteMDPs.states
import DiscreteMDPs.actions
import DiscreteMDPs.numStates
import DiscreteMDPs.numActions


type RandomMDP <: DiscreteMDP

    nStates::Int64
    nActions::Int64
    nNeighbors::Int64 # numbers of neighbors per state
    randomSeed::Int64

end


function states(mdp::RandomMDP)
    # returns an iterator over the states
    return 1:mdp.nStates
end

function actions(mdp::RandomMDP)
    # returns an iterator over the actions
    return 1:mdp.nActions
end

function numStates(mdp::RandomMDP)
    # returns the number of states
    return mdp.nStates
end

function numActions(mdp::RandomMDP)
    # returns the number of actions
    return mdp.nActions
end


function reward(mdp::RandomMDP, state::Int64, action::Int64)
    # reward is generated using the random seed and the action+state sum
    seed = state+action+mdp.randomSeed
    srand(seed)

    r = rand() 
    return r
end


function nextStates(mdp::RandomMDP, state::Int64, action::Int64)
    # returns the nighboring states and their probabilities
    randomSeed = mdp.randomSeed
    nNeighbors = mdp.nNeighbors
    nStates    = numStates(mdp)

    # these can be pre-allocated in the MDP type
    states = zeros(Int64, nNeighbors) 
    probs  = zeros(nNeighbors) 

    # set a unique seed for each (state, action) pair
    seed = state+action+mdp.randomSeed
    srand(seed)
    for i = 1:nNeighbors
        # random neighbors and probabilities
        states[i] = rand(1:nStates) 
        probs[i]  = rand()
    end

    # normalize
    norm = sum(probs)
    for i = 1:nNeighbors
        probs[i] = probs[i] / norm
    end
    return states, probs
end

end # module
