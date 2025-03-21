using AbstractTrees
using Colors
using CompositeGrids
using ElectronGas
using ElectronLiquid
using FeynmanDiagram
using GreenFunc
using JLD2
using LinearAlgebra
using Lehmann
using LQSGW
using Measurements
using MPI
using OneLoopFermiLiquid
using Parameters
using ProgressMeter
using PyCall
using Roots
using Test

import OneLoopFermiLiquid: lindhard, check_sign_Fs, check_signs_Fs_Fa, testdlr
import LQSGW: split_count, println_root, timed_result_to_string

import FeynmanDiagram.FrontEnds: TwoBodyChannel, Alli, PHr, PHEr, PPr, AnyChan
import FeynmanDiagram.FrontEnds:
    Filter, NoHartree, NoFock, DirectOnly, Wirreducible, Girreducible, NoBubble, Proper
import FeynmanDiagram.FrontEnds: Response, Composite, ChargeCharge, SpinSpin, UpUp, UpDown
import FeynmanDiagram.FrontEnds: AnalyticProperty, Instant, Dynamic

@pyimport numpy as np   # for saving/loading numpy data

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    root = 0
    rank = MPI.Comm_rank(comm)

    rslist = [1, 5, 10]
    beta = 40.0

    # For fast Fa
    # rslist = [[0.01, 0.1, 0.5]; 1:2:10]

    # # Standard rslist
    # rslist = [[0.01, 0.1, 0.25, 0.5]; 1:1:10]

    save = true
    debug = true
    verbose = true
    z_renorm = false
    show_progress = true

    run_neft = true
    run_ours = false

    # Yukawa interaction
    isDynamic = false

    ftype = "Fs"  # f^{Di} + f^{Ex} / 2
    # ftype = "Fa"  # f^{Ex} / 2

    # nk ≈ na ≈ 100 is sufficiently converged for all relevant euv/rtol
    Nk, Ok = 7, 6
    Na, Oa = 8, 7

    # DLR parameters for which r(q, 0) is smooth in the q → 0 limit (tested for rs = 1, 10)
    euv = 10.0
    rtol = 1e-7

    # Calculate the one-loop results for F↑↑ and F↑↓ using NEFT and/or our code
    if run_neft
        FsDMCs,
        FaDMCs,
        F1s,
        Fuu1s,
        Fud1s,
        Fs2s,
        Fa2s,
        Fuu2vpbs,
        Fud2vpbs,
        Fuu2cts,
        Fud2cts,
        Fuu2s,
        Fud2s = get_yukawa_one_loop_neft(rslist, beta; neval=1e7)

        FstotalNEFTs = F1s .+ Fs2s
        FatotalNEFTs = F1s .+ Fa2s
        FuutotalNEFTs = Fuu1s .+ Fuu2s
        FudtotalNEFTs = Fud1s .+ Fud2s

        # Get Thomas-Fermi result for F1 using exact expression
        rs_exact = LinRange(0, 10, 1000)
        F1s_exact = get_F1_TF.(rs_exact)

        # Get Thomas-Fermi result for F1 using exact expression
        function F2ct_exact(rs)
            xgrid =
                CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
            rstilde = rs * alpha_ueg / π
            F1 = get_F1_TF(rs)
            A = Interp.integrate1D(
                lindhard.(xgrid) .* (xgrid * rstilde) ./ (xgrid .^ 2 .+ rstilde),
                xgrid,
            )
            B = Interp.integrate1D(lindhard.(xgrid) .* xgrid, xgrid)
            return 2 * F1 * A + F1^2 * B
        end
        F2cts_exact = F2ct_exact.(rs_exact)

        # Save NEFT and exact results to np files
        if save && rank == root
            @save "one_loop_F_neft.jld2" rslist FsDMCs FaDMCs FstotalNEFTs FatotalNEFTs Fuu2NEFTs Fud2NEFTs FuutotalNEFTs FudtotalNEFTs F1s_exact F2cts_exact Fuu1s Fud1s Fs2s Fa2s Fuu2vpbs Fud2vpbs Fuu2cts Fud2cts Fuu2s Fud2s
        end
    end
    if run_ours
        Fs_DMCs = []
        Fa_DMCs = []
        F1s = []
        F2vs = []
        F2bs = []
        F2cts = []
        F2zs = []
        F2s = []
        for (i, rs) in enumerate(rslist)
            if debug && rank == root
                testdlr(rs, euv, rtol; verbose=verbose)
            end
            basic_param = Parameter.rydbergUnit(1.0 / beta, rs, 3)
            mass2 = isDynamic ? 1e-5 : basic_param.qTF^2
            Fs_DMC = -get_Fs_DMC(basic_param)
            Fa_DMC = -get_Fa_DMC(basic_param)
            println_root("\nrs = $rs:")
            println_root("F+ from DMC: $(Fs_DMC)")
            param = OneLoopParams(;
                rs=rs,
                beta=beta,
                Fs=Fs_DMC,
                euv=euv,
                rtol=rtol,
                Nk=Nk,
                Ok=Ok,
                Na=Na,
                Oa=Oa,
                isDynamic=isDynamic,
                mass2=mass2,
            )
            if debug && rank == root && rs > 0.25
                check_sign_Fs(param)
                check_signs_Fs_Fa(rs, Fs_DMC, Fa_DMC)
            end
            if verbose && rank == root && i == 1
                println_root("nk=$(length(param.qgrid)), na=$(length(param.θgrid))")
                println_root("nk=$(length(param.qgrid)), na=$(length(param.θgrid))")
                println_root("euv=$(param.euv), rtol=$(param.rtol)")
                println_root(
                    "\nrs=$(param.rs), beta=$(param.beta), Fs=$(Fs_DMC), Fa=$(Fa_DMC)",
                )
            end
            initialize_one_loop_params!(param)  # precomputes the interaction interpoland r(q, iνₘ)
            F1, F2v, F2b, F2ct, F2z, F2 = get_one_loop_Fs(
                param;
                verbose=verbose,
                show_progress=show_progress,
                ftype=ftype,
                z_renorm=z_renorm,
            )
            push!(Fs_DMCs, Fs_DMC)
            push!(Fa_DMCs, Fa_DMC)
            push!(F1s, F1)
            push!(F2vs, F2v)
            push!(F2bs, F2b)
            push!(F2cts, F2ct)
            push!(F2zs, F2z)
            push!(F2s, F2)
            GC.gc()
        end
        # Save our results to np files
        if save && rank == root
            @save "one_loop_$(ftype)_ours.jld2" rslist Fs_DMCs Fa_DMCs F1s F2vs F2bs F2cts F2zs F2s
        end
    end
    MPI.Finalize()
    return
end

main()
