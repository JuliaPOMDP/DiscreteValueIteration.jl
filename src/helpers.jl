# includes helper functions for the parallel solver

segment(n::Int64, u) = error("$(typeof(u)) does not implement segment")

function segment(n::Int64, u::UnitRange)
    chunks = {}

    cov = u.stop - u.start
    stride = int(floor((u.stop - u.start) / (n-1)))
    for i = 1:(n-1)
        si = u.start + (i-1)*stride
        ei = si + stride - 1
        i == (n-1) ? (ei = u.stop) : nothing
        push!(chunks, si:ei)
    end
    return chunks
end
