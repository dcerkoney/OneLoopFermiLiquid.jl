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
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    summand = matsubara_summand(param, q, θ, φ; kwargs...)
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end

"""
Evaluates the Matsubara sum for all (identical left/right) vertex corrections.
"""
vertex_matsubara_sum(param::OneLoopParams, q, θ, φ) =
    perform_bosonic_matsubara_sum(param, vertex_matsubara_summand, q, θ, φ)

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
box_matsubara_sum(param::OneLoopParams, q, θ, φ; ftype="fs") =
    perform_bosonic_matsubara_sum(param, box_matsubara_summand, q, θ, φ; ftype=ftype)

"""
Evaluates the Matsubara sum for all direct box diagrams.
"""
direct_box_matsubara_sum(param::OneLoopParams, q, θ, φ; which="both") =
    perform_bosonic_matsubara_sum(
        param,
        direct_box_matsubara_summand,
        q,
        θ,
        φ;
        which=which,
    )
