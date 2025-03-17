"""
G₀(k, iωₙ)
"""
function G0(param::OneLoopParams, k, iwn)
    @unpack me, μ, β = param
    return 1.0 / (iwn - k^2 / (2 * me) + μ)
end

"""
r(q, n) via multilinear interpolation, where n indexes bosonic Matsubara frequencies iνₙ.
"""
function r_interp(param::OneLoopParams, q, n)
    if q > param.maxQ
        return 0.0
    end
    if q ≤ param.Q_CUTOFF
        q = param.Q_CUTOFF
    end
    if param.isDynamic
        v_q_n = UEG.linear2D(param.r, param.qgrid_interp, param.mgrid, q, n)
    else
        v_q_n = Interp.interp1D(param.r_yukawa, param.qgrid_interp, q)
    end
    return v_q_n
end

"""
Dimensionless KO interaction, r(q, iνₘ) = NF * R(q, iνₘ)
"""
function r_data(param::OneLoopParams)
    @unpack paramc, qgrid_interp, mgrid, Mmax, dlr, Fs, NF = param
    Nq, Nw = length(qgrid_interp), length(mgrid)
    Pi = zeros(Float64, (Nq, Nw))
    rq = zeros(Float64, (Nq, Nw))
    for (ni, n) in enumerate(mgrid)
        for (qi, q) in enumerate(qgrid_interp)
            vq_plus_f = UEG.KOinstant(q, paramc)
            inv_vq_plus_f = 1.0 / vq_plus_f
            # rq = NF (vq + f) / (1 - (vq + f) Π0) - F
            Pi[qi, ni] = UEG.polarKW(q, n, paramc)
            rq[qi, ni] = NF / (inv_vq_plus_f - Pi[qi, ni]) - Fs
        end
    end
    # upsample to full frequency grid with indices ranging from 0 to M
    rq = matfreq2matfreq(dlr, rq, collect(0:Mmax); axis=2)
    return real.(rq)  # r(q, iνₘ) = r(q, -iνₘ) ⟹ r is real
end

"""
Dimensionless Yukawa interaction, r_yukawa(q) = NF * V_yukawa(q) = qTF^2 / (q^2 + mass2)
"""
function r_yukawa_data(param::OneLoopParams)
    @unpack qgrid_interp, qTF, mass2 = param
    @assert mass2 ≈ qTF^2
    r_yukawa(q) = qTF^2 / (q^2 + mass2)
    return r_yukawa.(qgrid_interp)
end

function initialize_one_loop_params!(param::OneLoopParams)
    if param.isDynamic
        # Dimensionless KO+ interaction, NF R = NF (v + f) / (1 - (v + f) Π₀) - f
        param.r = r_data(param)
    else
        # Dimensionless Yukawa interaction with mass √λ, NF V = qTF^2 / (q^2 + λ)
        param.r_yukawa = r_yukawa_data(param)
    end
    param.initialized = true
end

"""
δr(q, n) = r(q, n) / (v(q) + f) via multilinear interpolation of r, where n indexes bosonic Matsubara frequencies iνₙ.
"""
function dr_from_r(param::OneLoopParams, q, n)
    @assert param.isDynamic "δr(q, n) is only relevant when `isDynamic=true`!"
    if q > param.maxQ
        return 0.0
    end
    if q ≤ param.Q_CUTOFF
        q = param.Q_CUTOFF
    end
    vq_plus_f = UEG.KOinstant(q, param.paramc)
    return UEG.linear2D(param.r, param.qgrid_interp, param.mgrid, q, n) / vq_plus_f
end
