"""
Evaluates a given bosonic Matsubara sum.
"""
function perform_bosonic_matsubara_sum(
    param::OneLoopParams,
    matsubara_summand,
    q,
    θ,
    φ;
    kwargs...,
)
    @unpack mgrid, Mmax = param
    # Evaluate summand S(iνₘ) over the frequency mesh specified in `param`
    s_ivm = matsubara_summand(param, q, θ, φ; kwargs...)
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end

"""
Evaluates the Matsubara sum for the specified one-loop box diagram.
The box diagram can be crossed or uncrossed, direct or exchange—all four combinations enter F2.
"""
box_matsubara_sum(param::OneLoopParams, q, θ, φ; is_direct::Bool, is_crossed::Bool) =
    perform_bosonic_matsubara_sum(
        param,
        box_matsubara_summand,
        q,
        θ,
        φ;
        is_direct=is_direct,
        is_crossed=is_crossed,
    )

"""
Evaluates the Matsubara sum for all box diagrams.
"""
total_box_matsubara_sum(param::OneLoopParams, q, θ, φ, ftype) =
    perform_bosonic_matsubara_sum(param, total_box_matsubara_summand, q, θ, φ; ftype=ftype)

"""
Evaluates the Matsubara sum for the vertex corrections Λ₁(θ₁₂).
"""
vertex_matsubara_sum(param::OneLoopParams, q, θ, φ) =
    perform_bosonic_matsubara_sum(param, vertex_matsubara_summand, q, θ, φ)
