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

    beta = 40.0

    # rslist = [2.0, 4.0, 6.0, 8.0] 

    # # For fast runs
    # rslist = [[0.01, 0.1, 0.5]; 1:2:10]

    # Standard rslist
    rslist = [[0.01, 0.1, 0.25, 0.5]; 1:1:10]

    save = true
    debug = true
    verbose = true
    z_renorm = false
    show_progress = true
    leg_convention = :PH  # or :PP — just an external leg ordering convention, doesn't affect the final result

    neval = 1e6
    # neval = 4e8  # final results

    run_channels = [PHr, PHEr, PPr, AnyChan]
    run_neft = true
    run_ours = false

    # # Yukawa interaction
    # isDynamic = false

    # RPA
    isDynamic = true

    if isDynamic
        rpastr = "_rpa"
    else
        rpastr = ""
    end

    # Fs = f^{Di} + f^{Ex} / 2
    # Fa = f^{Ex} / 2
    # f^{Di} = Fs - Fa
    # f^{Ex} = 2 Fa

    # nk ≈ na ≈ 100 is sufficiently converged for all relevant euv/rtol
    Nk, Ok = 7, 6
    Na, Oa = 8, 7

    # DLR parameters for which r(q, 0) is smooth in the q → 0 limit (tested for rs = 1, 10)
    euv = 10.0
    rtol = 1e-7

    # Calculate the one-loop results for F↑↑ and F↑↓ using NEFT and/or our code
    if run_neft
        run_neft = isDynamic ? get_rpa_one_loop_neft : get_yukawa_one_loop_neft
        if PHr in run_channels
            oneloop_sa_phr_neft, oneloop_ud_phr_neft = run_neft(
                rslist,
                beta;
                neval=neval,
                leg_convention=leg_convention,
                z_renorm=z_renorm,
                channels=[PHr],
            )
            if save && rank == root
                @assert isempty(oneloop_sa_phr_neft) ==
                        isempty(oneloop_ud_phr_neft) ==
                        false
                @save "one_loop_F$(rpastr)_phr_neft_$(leg_convention).jld2" rslist oneloop_sa_phr_neft oneloop_ud_phr_neft
            end
        end
        if PHEr in run_channels
            oneloop_sa_pher_neft, oneloop_ud_pher_neft = run_neft(
                rslist,
                beta;
                neval=neval,
                leg_convention=leg_convention,
                z_renorm=z_renorm,
                channels=[PHEr],
            )
            if save && rank == root
                @assert isempty(oneloop_sa_pher_neft) ==
                        isempty(oneloop_ud_pher_neft) ==
                        false
                @save "one_loop_F$(rpastr)_pher_neft_$(leg_convention).jld2" rslist oneloop_sa_pher_neft oneloop_ud_pher_neft
            end
        end
        if PPr in run_channels
            oneloop_sa_ppr_neft, oneloop_ud_ppr_neft = run_neft(
                rslist,
                beta;
                neval=neval,
                leg_convention=leg_convention,
                z_renorm=z_renorm,
                channels=[PPr],
            )
            if save && rank == root
                @assert isempty(oneloop_sa_ppr_neft) ==
                        isempty(oneloop_ud_ppr_neft) ==
                        false
                @save "one_loop_F$(rpastr)_ppr_neft_$(leg_convention).jld2" rslist oneloop_sa_ppr_neft oneloop_ud_ppr_neft
            end
        end
        if AnyChan in run_channels
            oneloop_sa_neft, oneloop_ud_neft = run_neft(
                rslist,
                beta;
                neval=neval,
                leg_convention=leg_convention,
                z_renorm=z_renorm,
            )
            # Save full NEFT result to np file
            if save && rank == root
                @assert isempty(oneloop_sa_neft) == isempty(oneloop_ud_neft) == false
                @save "one_loop_F$(rpastr)_neft_$(leg_convention).jld2" rslist oneloop_sa_neft oneloop_ud_neft
            end
        end
    end
    if run_ours
        oneloop_sa_ours = []
        oneloop_ud_ours = []
        for (i, rs) in enumerate(rslist)
            if debug && rank == root
                testdlr(rs, euv, rtol; verbose=verbose)
            end
            basic_param = Parameter.rydbergUnit(1.0 / beta, rs, 3)
            mass2 = isDynamic ? 1e-5 : basic_param.qTF^2
            Fs = isDynamic ? -get_Fs_DMC(basic_param) : 0.0
            println_root("\nrs = $rs:")
            param = OneLoopParams(;
                rs=rs,
                beta=beta,
                Fs=Fs,
                # Fs=Fs_DMC,  # TODO: verify this was a bug!
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
            end
            if verbose && rank == root && i == 1
                println_root("nk=$(length(param.qgrid)), na=$(length(param.θgrid))")
                println_root("nk=$(length(param.qgrid)), na=$(length(param.θgrid))")
                println_root("euv=$(param.euv), rtol=$(param.rtol)")
                println_root("\nrs=$(param.rs), beta=$(param.beta), Fs=$(Fs)")
            end
            initialize_one_loop_params!(param)  # precomputes the interaction interpoland r(q, iνₘ)
            oneloop_sa, oneloop_ud = get_one_loop_Fs(
                param;
                verbose=verbose,
                z_renorm=z_renorm,
                show_progress=show_progress,
            )
            push!(oneloop_sa_ours, oneloop_sa)
            push!(oneloop_ud_ours, oneloop_ud)
            GC.gc()
        end
        # Save our results to np files
        if save && rank == root
            @assert isempty(oneloop_sa_ours) == isempty(oneloop_ud_ours) == false
            @save "one_loop_F_ours_new.jld2" rslist oneloop_sa_ours oneloop_ud_ours
        end
    end
    MPI.Finalize()
    return
end

main()
