

type SerialSolver <: Solver

    maxIterations::Int64

    tolerance::Float64

    gausssiedel::Bool

end


function solve(solver::SerialSolver, mdp::DiscreteMDP; verbose::Bool=false)

    policy = ValueIterationPolicy(ones(Int64,1))

    return policy
end

