function _box_summand(
    param::OneLoopParams,
    q1_4vector,
    q2_4vector,
    k1_4vector,
    k2_4vector,
    factor=1.0,
)
    summand = (
        factor *
        q^2 *
        r_interp(param, q1_4vector...) *
        r_interp(param, q2_4vector...) *
        G0(param, k1_4vector...) *
        G0(param, k2_4vector...) / β
    )
    return summand
end

"""
Builds a one-loop box Matsubara summand over the frequency mesh specified in `param`.
The box diagram can be crossed or uncrossed, direct or exchange—all four combinations enter F2.
"""
function box_matsubara_summand(
    param::OneLoopParams,
    q,
    θ,
    φ;
    is_direct::Bool,
    is_crossed::Bool,
)
    @unpack β, NF, kamp1, kamp2, θ12, mgrid, vmgrid, Mmax, iw0 = param
    # p1 = |k + q'|, p2 = |k' + q'|, p3 = |k' - q'|, qex = |k - k' + q'|
    k1vec = [0, 0, kamp1]
    k2vec = kamp2 * [sin(θ12), 0, cos(θ12)]
    qvec = q * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    vec_p1 = k1vec + qvec
    vec_p2 = k2vec + qvec
    vec_p3 = k2vec - qvec
    vec_qex = k1vec - k2vec + qvec
    p1 = norm(vec_p1)
    p2 = norm(vec_p2)
    p3 = norm(vec_p3)
    qex = norm(vec_qex)
    # There are four kinds of box Matsubara summands entering F2
    _box_summand_args = Dict(
        # crossed direct diagram
        (true, true) =>
            (m, vm) -> [(q, m), (q, m), (p1, iw0 + im * vm), (p2, iw0 + im * vm)],
        # uncrossed direct diagram
        (false, true) =>
            (m, vm) -> [(q, m), (q, m), (p1, iw0 + im * vm), (p3, iw0 - im * vm)],
        # crossed exchange diagram
        (true, false) =>
            (m, vm) ->
                [(q, m), (qex, m), (p1, iw0 + im * vm), (p1, iw0 + im * vm), -0.5],
        # uncrossed exchange diagram
        (false, false) =>
            (m, vm) ->
                [(q, m), (qex, m), (p1, iw0 + im * vm), (p3, iw0 - im * vm), -0.5],
    )
    # Build the requested box Matsubara summand S(iνₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        argfuncs = _box_summand_args[(is_direct, is_crossed)]
        s_ivm[i] = _box_summand(param::OneLoopParams, argfuncs(m, vm)...)
    end
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

"""
Builds the total one-loop box Matsubara summand for Fs/Fa over the frequency mesh specified in `param`.
"""
function box_matsubara_summand(param::OneLoopParams, q, θ, φ, ftype)
    @assert ftype in ["Fs", "Fa"]
    @unpack β, NF, kamp1, kamp2, θ12, mgrid, vmgrid, Mmax, iw0 = param
    # p1 = |k + q'|, p2 = |k' + q'|, p3 = |k' - q'|, qex = |k - k' + q'|
    k1vec = [0, 0, kamp1]
    k2vec = kamp2 * [sin(θ12), 0, cos(θ12)]
    qvec = q * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    vec_p1 = k1vec + qvec
    vec_p2 = k2vec + qvec
    vec_p3 = k2vec - qvec
    vec_qex = k1vec - k2vec + qvec
    p1 = norm(vec_p1)
    p2 = norm(vec_p2)
    p3 = norm(vec_p3)
    qex = norm(vec_qex)
    # S(iνₘ) = r(q', iν'ₘ) * r(k - k' + q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * (...)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        # Fermionized frequencies, iω₀ ± iνₘ
        ivm_Fp = iw0 + im * vm
        ivm_Fm = iw0 - im * vm
        # Ex (spin factor = 1/2), crossed + uncrossed
        s_ivm_inner =
            -r_interp(param, qex, m) * (G0(param, p1, ivm_Fp) + G0(param, p3, ivm_Fm)) / 2
        if ftype == "Fs"
            # Di (spin factor = 1), crossed + uncrossed
            s_ivm_inner +=
                r_interp(param, q, m) * (G0(param, p2, ivm_Fp) + G0(param, p3, ivm_Fm))
        end
        s_ivm[i] = q^2 * r_interp(param, q, m) * G0(param, p1, ivm_Fp) * s_ivm_inner / β
    end
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end
