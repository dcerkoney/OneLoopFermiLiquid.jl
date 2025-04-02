"""
The one-loop (GW) self-energy Σ₁.
"""
function Σ1(param::UEG.ParaMC, kgrid::KGT) where {KGT<:AbstractVector}
    @unpack kF, EF, Fs, basic = param
    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14
    # Based on ElectronGas.jl defaults for G0W0 self-energy (here, minK *= 100)
    maxK = 6 * kF
    minK = 1e-6 * kF
    # Get the one-loop self-energy
    Σ_imtime, _ = SelfEnergy.G0W0(
        basic,
        kgrid;
        Euv=Euv,
        rtol=rtol,
        maxK=maxK,
        minK=minK,
        int_type=:ko_const,
        Fs=Fs,
        Fa=-0.0,
    )
    # Σ_dyn(τ, k) → Σ_dyn(iωₙ, k)
    Σ = to_imfreq(to_dlr(Σ_imtime))
    return Σ
end
function Σ1(param::OneLoopParams, kgrid::KGT) where {KGT<:AbstractVector}
    return Σ1(param.paramc, kgrid)
end

"""
Leading-order (one-loop) correction to Z_F.
"""
function get_Z1(param::UEG.ParaMC, kgrid::KGT) where {KGT<:AbstractVector}
    if param.isDynamic == false
        # the one-loop self-energy is frequency independent for the Thomas-Fermi interaction
        return 0.0
    end
    sigma1 = Σ1(param, kgrid)
    return zfactor_fermi(param.basic, sigma1)  # compute Z_F using improved finite-temperature scaling
end
function get_Z1(param::UEG.ParaMC)
    if param.isDynamic == false
        # the one-loop self-energy is frequency independent for the Thomas-Fermi interaction
        return 0.0
    end
    @unpack kF = param
    kgrid =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, 2.0 * kF], [0.0, kF], 16, 1e-8 * kF, 16)
    sigma1 = Σ1(param, kgrid)
    return zfactor_fermi(param.basic, sigma1)  # compute Z_F using improved finite-temperature scaling
end
function get_Z1(param::OneLoopParams, kgrid::KGT) where {KGT<:AbstractVector}
    return get_Z1(param.paramc, kgrid)
end
function get_Z1(param::OneLoopParams)
    return get_Z1(param.paramc)
end

"""
Integrand for the tree-level estimate of F⁺₀ ~ R(k - k', 0).
"""
function integrand_F1(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return -x / lindhard(x)
    end
    coeff = rs_tilde + Fs * x^2
    # NF (R + f)
    NF_times_Rpf_ex = coeff / (x^2 + coeff * lindhard(x))
    return -x * NF_times_Rpf_ex
end

"""
Tree-level estimate of F⁺₀ ~ ⟨R(k - k', 0)⟩.
"""
function get_F1(param::UEG.ParaMC)
    if param.isDynamic == false
        return get_F1_TF(param.rs)
    end
    @unpack rs, kF, EF, NF, Fs, basic = param
    rstilde = rs * alpha_ueg / π
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 16, 1e-8, 16)
    y = [integrand_F1(x, rstilde, Fs) for x in xgrid]
    F1 = (Fs / 2) + Interp.integrate1D(y, xgrid)
    return F1
end
function get_F1(param::OneLoopParams)
    return get_F1(param.paramc)
end

"""
Tree-level estimate of F⁺₀ ~ ⟨V_TF(k - k', 0)⟩ for the Thomas-Fermi interaction.
"""
function get_F1_TF(rs)
    if isinf(rs)
        return -0.5
    elseif rs == 0
        return -0.0
    end
    rstilde = rs * alpha_ueg / π
    F1 = (rstilde / 2) * log(rstilde / (rstilde + 1))
    return F1
end

"""
Returns the integrand x * r(x, 0) in terms of x = |k_1 - k_2| / 2k_F.
"""
function x_NF_R0(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return x / lindhard(x)
    end
    coeff = rs_tilde + Fs * x^2
    # NF (R + f)
    NF_times_Rpf_ex = coeff / (x^2 + coeff * lindhard(x))
    # NF R = NF (R + f) - Fs
    NF_times_Rp_ex = NF_times_Rpf_ex - Fs
    return x * NF_times_Rp_ex
end

"""
Returns the integrand x * r_TF(x, 0) in terms of x = |k_1 - k_2| / 2k_F.
"""
function x_NF_VTF(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return x
    elseif rs_tilde == 0
        return 0.0
    end
    NF_times_VTF_ex = rs_tilde / (x^2 + rs_tilde)
    return x * NF_times_VTF_ex
end

"""
Returns the integrand x * (r(x, 0))^2 in terms of x = |k_1 - k_2| / 2k_F.
"""
function x_NF2_R02(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return x / lindhard(x)^2
    end
    coeff = rs_tilde + Fs * x^2
    # NF (R + f)
    NF_times_Rpf_ex = coeff / (x^2 + coeff * lindhard(x))
    # x (NF R)^2 = x (NF (R + f) - Fs)^2
    NF_times_Rp_ex = NF_times_Rpf_ex - Fs
    return x * NF_times_Rp_ex^2
end

"""
Returns the integrand x * (r_TF(x, 0))^2 in terms of x = |k_1 - k_2| / 2k_F.
"""
function x_NF2_VTF2(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return x
    elseif rs_tilde == 0
        return 0.0
    end
    NF_times_VTF_ex = rs_tilde / (x^2 + rs_tilde)
    return x * NF_times_VTF_ex^2
end
