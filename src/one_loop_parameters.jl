
# Specify the type of momentum and frequency (index) grids explicitly to ensure type stability
const MomInterpGridType = CompositeGrids.CompositeG.Composite{
    Float64,
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound},
    CompositeGrids.CompositeG.Composite{
        Float64,
        CompositeGrids.SimpleG.Log{Float64},
        CompositeGrids.SimpleG.Uniform{Float64,CompositeGrids.SimpleG.ClosedBound},
    },
}
const MomGridType = CompositeGrids.CompositeG.Composite{
    Float64,
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound},
    CompositeGrids.CompositeG.Composite{
        Float64,
        CompositeGrids.SimpleG.Log{Float64},
        CompositeGrids.SimpleG.GaussLegendre{Float64},
    },
}
const MGridType = CompositeGrids.SimpleG.Arbitrary{Int64,CompositeGrids.SimpleG.ClosedBound}
const FreqGridType =
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound}
const AngularGridType = CompositeGrids.CompositeG.Composite{
    Float64,
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound},
    CompositeGrids.CompositeG.Composite{
        Float64,
        CompositeGrids.SimpleG.Log{Float64},
        CompositeGrids.SimpleG.GaussLegendre{Float64},
    },
}

"""
Parameter container for the one-loop vertex corrections in the Fermi liquid theory.
"""
@with_kw mutable struct OneLoopParams
    # UEG parameters
    beta::Float64
    rs::Float64
    dim::Int = 3
    spin::Int = 2
    Fs::Float64 = -0.0

    # mass2::Float64 = 1.0      # large Yukawa screening λ for testing
    mass2::Float64 = 1e-5     # fictitious Yukawa screening λ
    massratio::Float64 = 1.0  # mass ratio m*/m

    # true:  KO+ interaction
    # false: Yukawa interaction
    isDynamic = true

    basic::Parameter.Para = Parameter.rydbergUnit(1.0 / beta, rs, dim; Λs=mass2, spin=spin)
    paramc::ParaMC = UEG.ParaMC(;
        rs=rs,
        beta=beta,
        dim=dim,
        spin=spin,
        mass2=mass2,
        Fs=Fs,
        basic=basic,
        isDynamic=isDynamic,
    )
    kF::Float64 = basic.kF
    EF::Float64 = basic.EF
    β::Float64 = basic.β
    me::Float64 = basic.me
    ϵ0::Float64 = basic.ϵ0
    e0::Float64 = basic.e0
    μ::Float64 = basic.μ
    NF::Float64 = basic.NF
    NFstar::Float64 = basic.NF * massratio
    qTF::Float64 = basic.qTF
    fs::Float64 = Fs / NF

    # Momentum grid parameters
    maxK::Float64 = 6 * kF
    maxQ::Float64 = 6 * kF
    Q_CUTOFF::Float64 = 1e-10 * kF

    # nk ≈ 75 is sufficiently converged for all relevant euv/rtol
    Nk::Int = 7
    Ok::Int = 6

    # na ≈ 75 is sufficiently converged for all relevant euv/rtol
    Na::Int = 8
    Oa::Int = 7

    euv::Float64 = 1000.0
    rtol::Float64 = 1e-7

    # We precompute r(q, iνₘ) on a mesh of ~100 k-points
    # NOTE: EL.jl default is `Nk, Ok = 16, 16` (~700 k-points)
    qgrid_interp::MomInterpGridType =
        CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], 10, 0.01 * kF, 10)  # sufficient for 1-decimal accuracy
    # CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], Nk, 0.01 * kF, Ok)

    # Later, we integrate r(q, iνₘ) on a Gaussian mesh of ~100 k-points
    qgrid::MomGridType =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, maxQ], [0.0, 2 * kF], Nk, 0.01 * kF, Ok)

    # Sparse angular grids (~100 points each)
    # NOTE: EL.jl default is `Na, Oa = 16, 32` (~1000 θ/φ-points)
    θgrid::AngularGridType =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], Na, 0.01, Oa)
    φgrid::AngularGridType =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, 2π], [0.0, 2π], Na, 0.01, Oa)

    # Use a sparse DLR grid for the bosonic Matsubara summation (~30-50 iνₘ-points)
    dlr::DLRGrid{Float64,:ph} =
        DLRGrid(; Euv=euv * EF, β=β, rtol=rtol, isFermi=false, symmetry=:ph)
    mgrid::MGridType = SimpleG.Arbitrary{Int64}(dlr.n)
    vmgrid::FreqGridType = SimpleG.Arbitrary{Float64}(dlr.ωn)
    Mmax::Int64 = maximum(mgrid)

    # Incoming momenta k1, k2 and incident scattering angle
    kamp1::Float64 = basic.kF
    kamp2::Float64 = basic.kF
    θ12::Float64 = π / 2

    # Lowest non-zero Matsubara frequencies
    iw0 = im * π / β  # fermionic
    iv1 = im * 2π / β  # bosonic

    # r grid data is precomputed in an initialization step
    initialized::Bool = false
    r::Matrix{Float64} = Matrix{Float64}(undef, length(qgrid_interp), length(mgrid))
    r_yukawa::Array{Float64} = Array{Float64}(undef, length(qgrid_interp))
end
