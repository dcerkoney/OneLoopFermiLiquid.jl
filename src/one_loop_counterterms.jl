"""
Calculates and returns the total counterterm contribution to F2s and F2a:

    2R(z1 - f1 Π0) - f1 Π0 f1 (+ R Π0 R)
"""
function one_loop_counterterms(paramc::UEG.ParaMC)
    @unpack rs, kF, EF, NF, Fs, basic, isDynamic = paramc
    if isDynamic == false
        @assert paramc.mass2 ≈ paramc.qTF^2 "Counterterms currently only implemented for the Thomas-Fermi interaction (Yukawa mass = qTF)!"
    end
    rstilde = rs * alpha_ueg / π

    # x = |k - k'| / 2kF
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 16, 1e-8, 16)

    # NF * ⟨R⟩ = -x N_F R(2kF x, 0)
    F1 = get_F1(paramc)

    # x R(2kF x, 0)
    x_NF_R = isDynamic ? x_NF_R0 : x_NF_VTF
    x_R0 = [x_NF_R(x, rstilde, Fs) / NF for x in xgrid]

    # Z_1(kF)
    z1 = get_Z1(paramc, 2 * kF * xgrid)

    # Π₀(q, iν=0) = -NF * 𝓁(q / 2kF)
    Π0 = -NF * lindhard.(xgrid)

    # A = z₁ + ∫₀¹ dx x R(x, 0) Π₀(x, 0)
    A = z1 + Interp.integrate1D(x_R0 .* Π0, xgrid)

    # B = ∫₀¹ dx x Π₀(x, 0) / NF
    B = Interp.integrate1D(xgrid .* Π0 / NF, xgrid)

    # (NF/2)*⟨2R(z1 - f1 Π0) - f1 Π0 f1⟩ = -2 F1 A - F1² B
    Fs2ct = Fa2ct = -(2 * F1 * A + F1^2 * B)
    return Fs2ct, Fa2ct
end
function one_loop_counterterms(param::OneLoopParams)
    return one_loop_counterterms(param.paramc)
end

function one_loop_bubble_counterterm(param::OneLoopParams)
    @unpack rs, kF, EF, NF, Fs, basic, isDynamic = param
    if isDynamic == false
        @assert param.mass2 ≈ param.qTF^2 "Counterterms currently only implemented for the Thomas-Fermi interaction (Yukawa mass = qTF)!"
    end
    rstilde = rs * alpha_ueg / π

    # x = |k - k'| / 2kF
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 16, 1e-8, 16)

    # x R(2kF x, 0)
    x_NF2_R2 = isDynamic ? x_NF2_R02 : x_NF2_VTF2
    x_R02 = [x_NF2_R2(x, rstilde, Fs) / NF^2 for x in xgrid]

    # Π₀(q, iν=0) = -NF * 𝓁(q / 2kF)
    Π0 = -NF * lindhard.(xgrid)

    # NF ⟨R Π₀ R⟩
    Fs2bct = Interp.integrate1D(NF .* x_R02 .* Π0, xgrid)
    Fa2bct = 0.0  # bubble counterterm vanishes for direct diagrams (it is improper)
    return Fs2bct, Fa2bct
end
