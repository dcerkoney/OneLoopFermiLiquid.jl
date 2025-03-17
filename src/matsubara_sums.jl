"""
Inner Matsubara summand for the vertex corrections.
"""
function vertex_matsubara_summand(param::OneLoopParams, q, θ, φ)
    @unpack β, NF, kamp1, kamp2, θ12, mgrid, vmgrid, Mmax, iw0 = param
    # p1 = |k + q'|, p2 = |k' + q'|
    k1vec = [0, 0, kamp1]
    k2vec = kamp2 * [sin(θ12), 0, cos(θ12)]
    qvec = q * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    vec_p1 = k1vec + qvec
    vec_p2 = k2vec + qvec
    p1 = norm(vec_p1)
    p2 = norm(vec_p2)
    # S(iνₘ) = r(q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p2, iω₀ + iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        s_ivm[i] = (
            -q^2 *
            r_interp(param, q, m) *
            G0(param, p1, iw0 + im * vm) *
            G0(param, p2, iw0 + im * vm) / β
        )
    end
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

"""
Evaluates the Matsubara sum for the vertex corrections.
"""
function vertex_matsubara_sum(param::OneLoopParams, q, θ, φ)
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    summand = vertex_matsubara_summand(param, q, θ, φ)
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end

"""
Inner Matsubara summand for the box diagrams.
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
    # S(iνₘ) = R(q', iν'ₘ) * R(k - k' + q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p3, iω₀ - iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        # Fermionized frequencies, iω₀ ± iνₘ
        ivm_Fp = iw0 + im * vm
        ivm_Fm = iw0 - im * vm
        # Ex (spin factor = 1/2)
        s_ivm_inner =
            -r_interp(param, qex, m) * (G0(param, p1, ivm_Fp) + G0(param, p3, ivm_Fm)) / 2
        if ftype == "Fs"
            # Di (spin factor = 1)
            s_ivm_inner +=
                r_interp(param, q, m) * (G0(param, p2, ivm_Fp) + G0(param, p3, ivm_Fm))
        end
        s_ivm[i] = q^2 * r_interp(param, q, m) * G0(param, p1, ivm_Fp) * s_ivm_inner / β
    end
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

"""
Evaluates the Matsubara sum for the box diagrams.
"""
function box_matsubara_sum(param::OneLoopParams, q, θ, φ; ftype="fs")
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    summand = box_matsubara_summand(param, q, θ, φ, ftype)
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end

"""
Inner Matsubara summand for the direct box diagrams.
"""
function direct_box_matsubara_summand(param::OneLoopParams, q, θ, φ; which="both")
    @assert which in ["both", "ladder", "crossed"]
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
    # S(iνₘ) = r(q', iν'ₘ) * r(k - k' + q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p3, iω₀ - iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        # Fermionized frequencies, iω₀ ± iνₘ
        ivm_Fp = iw0 + im * vm
        ivm_Fm = iw0 - im * vm
        # Ex (spin factor = 1/2)
        if which == "both"
            s_ivm_inner =
                r_interp(param, q, m) * (G0(param, p2, ivm_Fp) + G0(param, p3, ivm_Fm))
        elseif which == "crossed"
            s_ivm_inner = r_interp(param, q, m) * G0(param, p2, ivm_Fp)
        elseif which == "ladder"
            s_ivm_inner = r_interp(param, q, m) * G0(param, p3, ivm_Fm)
        end
        s_ivm[i] = q^2 * r_interp(param, q, m) * G0(param, p1, ivm_Fp) * s_ivm_inner / β
    end
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

"""
Evaluates the inner Matsubara sum for the direct box diagrams.
"""
function direct_box_matsubara_sum(param::OneLoopParams, q, θ, φ; which="both")
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    @assert which in ["both", "ladder", "crossed"]
    summand = direct_box_matsubara_summand(param, q, θ, φ; which=which)
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end
