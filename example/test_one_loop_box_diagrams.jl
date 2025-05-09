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

    # rslist = [1.0]

    # # For fast runs
    # rslist = [[0.01, 0.1, 0.5]; 1:2:10]

    # Standard rslist
    rslist = [[0.01, 0.1, 0.25, 0.5]; 1:1:10]

    save = true
    debug = true
    verbose = true
    show_progress = true

    # Yukawa interaction
    isDynamic = false

    # nk ≈ na ≈ 100 is sufficiently converged for all relevant euv/rtol
    Nk, Ok = 7, 6
    Na, Oa = 8, 7

    # DLR parameters for which r(q, 0) is smooth in the q → 0 limit (tested for rs = 1, 10)
    euv = 10.0
    rtol = 1e-7

    # Calculate the one-loop results for Fs and Fa, F↑↑ and F↑↓ of each box diagram type
    boxdiags_sa_ours = []
    boxdiags_ud_ours = []
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
        boxdiags_sa_this_rs, boxdiags_ud_this_rs = get_one_loop_box_contributions(
            param;
            verbose=verbose,
            show_progress=show_progress,
        )
        push!(boxdiags_sa_ours, boxdiags_sa_this_rs)
        push!(boxdiags_ud_ours, boxdiags_ud_this_rs)
        GC.gc()
    end
    # Save our results to np file
    if save && rank == root
        @assert isempty(boxdiags_sa_ours) == isempty(boxdiags_ud_ours) == false
        @save "one_loop_box_diagrams_ours.jld2" rslist boxdiags_sa_ours boxdiags_ud_ours
    end
    MPI.Finalize()
    return
end

main()
