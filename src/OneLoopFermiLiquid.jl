"""
Calculate the Fermi liquid parameters to one-loop order in the local Fermi-liquid effective field theory.
"""
module OneLoopFermiLiquid

using AbstractTrees
using CodecZlib
using CompositeGrids
using ElectronGas
using ElectronLiquid
using FeynmanDiagram
using GreenFunc
using JLD2
using Lehmann
using LinearAlgebra
using LQSGW
using MCIntegration
using MPI
using ProgressMeter
using Parameters
using Test

import FeynmanDiagram.FrontEnds: TwoBodyChannel, Alli, PHr, PHEr, PPr, AnyChan
import FeynmanDiagram.FrontEnds:
    Filter, NoHartree, NoFock, DirectOnly, Wirreducible, Girreducible, NoBubble, Proper
import FeynmanDiagram.FrontEnds: Response, Composite, ChargeCharge, SpinSpin, UpUp, UpDown
import FeynmanDiagram.FrontEnds: AnalyticProperty, Instant, Dynamic

const MAXIMUM_STEPS = 100
const PROJECT_ROOT = pkgdir(LQSGW)
const DATA_DIR = joinpath(PROJECT_ROOT, "data")

include("one_loop_parameters.jl")
export OneLoopParams,
    MomInterpGridType, MomGridType, MGridType, FreqGridType, AngularGridType

include("fermi_liquid_interactions.jl")
export get_Fs_DMC, get_Fa_DMC

include("propagators.jl")
export G0, r_interp

include("tree_level.jl")
export get_tree_level_self_consistent_Fs, get_F1, get_F1_TF, get_Z1
export Σ1, integrand_F1, x_NF_R0, x_NF_VTF, x_NF2_R02, x_NF2_VTF2

include("matsubara_sums.jl")
export vertex_matsubara_summand, vertex_matsubara_sum
export box_matsubara_summand, box_matsubara_sum
export direct_box_matsubara_summand, direct_box_matsubara_sum

include("one_loop.jl")
export initialize_one_loop_params!, get_one_loop_Fs, get_yukawa_one_loop_neft, get_one_loop_diagrams
export one_loop_vertex_corrections,
    one_loop_box_diagrams, one_loop_direct_box_diagrams, one_loop_counterterms

const alpha_ueg = (4 / 9π)^(1 / 3)
export alpha_ueg

function count_sum(g::Graph{F,W}) where {F,W}
    res = 0
    for sg in PostOrderDFS(g)
        if haschildren(sg) && sg.operator == ComputationalGraphs.Sum
            res += 1
        end
    end
    return res
end

"""
    lerp(M_start, M_end, alpha)

Helper function for linear interpolation with mixing parameter α: x * (1 - α) + y * α
"""
function lerp(x, y, alpha)
    return (1 - alpha) * x + alpha * y
end

"""
The Lindhard function l(x) in terms of x = q / 2kF.
"""
function lindhard(x)
    if abs(x) < 1e-4
        return 1.0 - x^2 / 3 - x^4 / 15
    elseif abs(x - 1) < 1e-7
        return 0.5
    elseif x > 20
        return 1 / (3 * x^2) + 1 / (15 * x^4)
    end
    return 0.5 + ((1 - x^2) / (4 * x)) * log(abs((1 + x) / (1 - x)))
end

"""
    function uniqueperm(a)

Return the indices of unique elements of `a` in the order 
they first appear, such that `a[uniqueperm(a)] == unique(a)`.
"""
function uniqueperm(a)
    return unique(i -> a[i], eachindex(a))
end

"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`.
Used to chunk polarization for MPI parallelization.
"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i in 1:n]
end

function println_root(io::IO, msg)
    MPI.Initialized() == false && MPI.Init()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(io, msg)
    end
end

function println_root(msg)
    MPI.Initialized() == false && MPI.Init()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(msg)
    end
end

function print_root(io::IO, msg)
    MPI.Initialized() == false && MPI.Init()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        print(io, msg)
    end
end

function print_root(msg)
    MPI.Initialized() == false && MPI.Init()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        print(msg)
    end
end

const TimedResultType{T} = @NamedTuple{
    value::T,
    time::Float64,
    bytes::Int64,
    gctime::Float64,
    gcstats::Base.GC_Diff,
} where {T}

function timed_result_to_string(timed_res::TimedResultType)
    time = round(timed_res.time; sigdigits=3)
    num_bytes = Base.format_bytes(timed_res.bytes)
    num_allocs = timed_res.gcstats.malloc + timed_res.gcstats.poolalloc
    return "  $time seconds ($num_allocs allocations: $num_bytes)"
end

function check_sign_Fs(param::OneLoopParams)
    # ElectronLiquid.jl sign convention: Fs < 0
    @unpack Fs, paramc = param
    if param.rs > 0.25
        @assert Fs ≤ 0 "Fs = $Fs must be negative in the ElectronLiquid convention!"
        @assert paramc.Fs ≤ 0 "Fs = $Fs must be negative in the ElectronLiquid convention!"
    else
        println("WARNING: when rs is nearly zero, we cannot check the sign of Fs!")
    end
end

function check_signs_Fs_Fa(rs, Fs, Fa)
    # ElectronLiquid.jl sign convention: Fs < 0
    if rs > 0.25
        @assert Fs ≤ 0 "Fs = $Fs must be negative in the ElectronLiquid convention!"
        @assert Fa ≤ 0 "Fa = $Fa must be negative in the ElectronLiquid convention!"
    else
        println("WARNING: when rs is nearly zero, we cannot check the signs of Fs/Fa!")
    end
end

function testdlr(rs, euv, rtol; rpa=false, verbose=false)
    verbose && println("(rs = $rs) Testing DLR grid with Euv / EF = $euv, rtol = $rtol")
    param = Parameter.rydbergUnit(1.0 / 40.0, rs, 3)
    @unpack β, kF, EF, NF = param
    if rpa
        Fs = 0.0
        fs = 0.0
    else
        Fs = -get_Fs_DMC(param)
        fs = Fs / NF
    end
    paramc = ParaMC(; rs=rs, beta=40.0, dim=3, Fs=Fs)

    qgrid_interp = CompositeGrid.LogDensedGrid(
        :uniform,
        [0.0, 6 * kF],
        [0.0, 2 * kF],
        16,
        0.01 * kF,
        16,
    )

    dlr = DLRGrid(; Euv=euv * EF, β=β, rtol=rtol, isFermi=false, symmetry=:ph)
    mgrid = SimpleG.Arbitrary{Int64}(dlr.n)
    Mmax = maximum(mgrid)

    verbose && println("Nw = $(length(dlr.n)), Mmax = $Mmax")

    Nq, Nw = length(qgrid_interp), length(mgrid)
    Pi = zeros(Float64, (Nq, Nw))
    Rs = zeros(Float64, (Nq, Nw))
    for (ni, n) in enumerate(mgrid)
        for (qi, q) in enumerate(qgrid_interp)
            rq = UEG.KOinstant(q, paramc)
            invrq = 1.0 / rq
            # Rq = (vq + f) / (1 - (vq + f) Π0) - f
            Pi[qi, ni] = UEG.polarKW(q, n, paramc)
            Rs[qi, ni] = 1 / (invrq - Pi[qi, ni]) - fs
        end
    end
    # upsample to full frequency grid with indices ranging from 0 to M
    Rs = matfreq2matfreq(dlr, Rs, collect(0:Mmax); axis=2)
    return Rs, qgrid_interp, paramc
end

function compare(data, expect, ratio=5)
    println(data, ", ", expect)
    @test isapprox(data.val, expect, atol=ratio * data.err)
end

end  # module LQSGW
