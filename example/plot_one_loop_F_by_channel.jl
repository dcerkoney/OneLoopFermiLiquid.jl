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
using PyPlot
using Roots
using Test

import LQSGW: split_count, println_root, timed_result_to_string

import FeynmanDiagram.FrontEnds: TwoBodyChannel, Alli, PHr, PHEr, PPr, AnyChan
import FeynmanDiagram.FrontEnds:
    Filter, NoHartree, NoFock, DirectOnly, Wirreducible, Girreducible, NoBubble, Proper
import FeynmanDiagram.FrontEnds: Response, Composite, ChargeCharge, SpinSpin, UpUp, UpDown
import FeynmanDiagram.FrontEnds: AnalyticProperty, Instant, Dynamic

@pyimport numpy as np   # for saving/loading numpy data
@pyimport scienceplots  # for style "science"
@pyimport scipy.interpolate as interp

"""
Fits a spline to the data (x, y) with weights e in the range [xmin, xmax].
"""
function spline(x, y, e=1e-6 * one.(y); xmin=0.0, xmax=x[end])
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(xmin, xmax, 1000))
    yfit = spl(__x)
    return __x, yfit
end

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "black" => "black",
    "orange" => "#EE7733",
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "magenta" => "#EE3377",
    "red" => "#CC3311",
    "teal" => "#009988",
    "grey" => "#BBBBBB",
]);
style = PyPlot.matplotlib."style"
style.use(["science", "std-colors"])
const color = [
    "black",
    cdict["orange"],
    cdict["blue"],
    cdict["cyan"],
    cdict["magenta"],
    cdict["red"],
    cdict["teal"],
    cdict["grey"],
]
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 16
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "Times New Roman"

function ex_to_di_sa(Wse, Wae)
    Ws = (Wse + 3 * Wae) / 2
    Wa = (Wse - Wae) / 2
    return Ws, Wa
end

function ex_to_di_ud(Wuue, Wude)
    Wuu = -Wuue
    Wud = Wude - Wuue
    return Wuu, Wud
end

function plot_F_by_channel(isDynamic=false, z_renorm=false)
    # NOTE: NEFT tree-level data is missing an overall minus sign (change sign convention for F).
    #       However, we now use the exact expressions for F1 and F2ct to isolate the problem with F2d.
    neft_factor_tree_level = 1.0
    neft_factor_one_loop = 1.0

    leg_convention = :PH
    # leg_convention = :PP
    neft_splines = false

    function plotdata(x, y, e; error_multiplier=1.0)
        if neft_splines == false
            return x, y
        end
        xspline, yspline = spline(x, y, error_multiplier * e)
        return xspline, yspline
    end

    interactionstr = isDynamic ? "" : "_yukawa"
    zstr = z_renorm ? "_z_renorm" : ""

    # Load PHr NEFT benchmark data using jld2
    @load "one_loop_F_phr_neft_$(leg_convention).jld2" rslist oneloop_sa_phr_neft oneloop_ud_phr_neft
    rslist_big = rslist
    function loaddata_oneloop_phr_neft(
        property::Symbol,
        representation="sa",
        factor=1.0,
        ex_to_di=false,
    )
        @assert representation in ["sa", "ud"]
        # data = representation == "sa" ? oneloop_sa_phr_neft : oneloop_ud_phr_neft
        data = oneloop_ud_phr_neft
        res = [factor .* x for x in getproperty.(data, property)]
        # res_s, res_a = first.(res), last.(res)
        res_uu, res_ud = first.(res), last.(res)
        if ex_to_di
            # exchange_to_direct = representation == "sa" ? ex_to_di_sa : ex_to_di_ud
            # return exchange_to_direct(res_s, res_a)
            res_uu, res_ud = ex_to_di_ud(res_uu, res_ud)
        end
        if representation == "sa"
            res_s, res_a = Ver4.ud2sa(res_uu, res_ud)
            return res_s, res_a
        end
        return res_uu, res_ud
    end

    # Fs = (F↑↑ + F↑↓) / 2, Fa = (F↑↑ - F↑↓) / 2
    Fs2ds_phr_neft, Fa2ds_phr_neft =
        loaddata_oneloop_phr_neft(:F2d, "sa", neft_factor_one_loop)

    # F↑↑ = Fs + Fa, F↑↓ = Fs - Fa
    Fuu2ds_phr_neft, Fud2ds_phr_neft =
        loaddata_oneloop_phr_neft(:F2d, "ud", neft_factor_one_loop)

    # Load PHEr NEFT benchmark data using jld2
    @load "one_loop_F_pher_neft_$(leg_convention).jld2" rslist oneloop_sa_pher_neft oneloop_ud_pher_neft
    rslist_big = rslist
    function loaddata_oneloop_pher_neft(
        property::Symbol,
        representation="sa",
        factor=1.0,
        ex_to_di=false,
    )
        @assert representation in ["sa", "ud"]
        # data = representation == "sa" ? oneloop_sa_pher_neft : oneloop_ud_pher_neft
        data = oneloop_ud_pher_neft
        res = [factor .* x for x in getproperty.(data, property)]
        # res_s, res_a = first.(res), last.(res)
        res_uu, res_ud = first.(res), last.(res)
        if ex_to_di
            # exchange_to_direct = representation == "sa" ? ex_to_di_sa : ex_to_di_ud
            # return exchange_to_direct(res_s, res_a)
            res_uu, res_ud = ex_to_di_ud(res_uu, res_ud)
        end
        if representation == "sa"
            res_s, res_a = Ver4.ud2sa(res_uu, res_ud)
            return res_s, res_a
        end
        return res_uu, res_ud
    end

    # Fs = (F↑↑ + F↑↓) / 2, Fa = (F↑↑ - F↑↓) / 2
    Fs2ds_pher_neft, Fa2ds_pher_neft =
        loaddata_oneloop_pher_neft(:F2d, "sa", neft_factor_one_loop)

    # F↑↑ = Fs + Fa, F↑↓ = Fs - Fa
    Fuu2ds_pher_neft, Fud2ds_pher_neft =
        loaddata_oneloop_pher_neft(:F2d, "ud", neft_factor_one_loop)

    # Load PPr NEFT benchmark data using jld2
    @load "one_loop_F_ppr_neft_$(leg_convention).jld2" rslist oneloop_sa_ppr_neft oneloop_ud_ppr_neft
    rslist_big = rslist
    function loaddata_oneloop_ppr_neft(
        property::Symbol,
        representation="sa",
        factor=1.0,
        ex_to_di=false,
    )
        @assert representation in ["sa", "ud"]
        # data = representation == "sa" ? oneloop_sa_ppr_neft : oneloop_ud_ppr_neft
        data = oneloop_ud_ppr_neft
        res = [factor .* x for x in getproperty.(data, property)]
        # res_s, res_a = first.(res), last.(res)
        res_uu, res_ud = first.(res), last.(res)
        if ex_to_di
            # exchange_to_direct = representation == "sa" ? ex_to_di_sa : ex_to_di_ud
            # return exchange_to_direct(res_s, res_a)
            res_uu, res_ud = ex_to_di_ud(res_uu, res_ud)
        end
        if representation == "sa"
            res_s, res_a = Ver4.ud2sa(res_uu, res_ud)
            return res_s, res_a
        end
        return res_uu, res_ud
    end

    # Fs = (F↑↑ + F↑↓) / 2, Fa = (F↑↑ - F↑↓) / 2
    Fs2ds_ppr_neft, Fa2ds_ppr_neft =
        loaddata_oneloop_ppr_neft(:F2d, "sa", neft_factor_one_loop)

    # F↑↑ = Fs + Fa, F↑↓ = Fs - Fa
    Fuu2ds_ppr_neft, Fud2ds_ppr_neft =
        loaddata_oneloop_ppr_neft(:F2d, "ud", neft_factor_one_loop)

    # Load AnyChan NEFT benchmark data using jld2
    @load "one_loop_F_neft_$(leg_convention).jld2" rslist oneloop_sa_neft oneloop_ud_neft
    rslist_big = rslist
    function loaddata_oneloop_anychan_neft(
        property::Symbol,
        representation="sa",
        factor=1.0,
        ex_to_di=false,
    )
        @assert representation in ["sa", "ud"]
        # data = representation == "sa" ? oneloop_sa_neft : oneloop_ud_neft
        data = oneloop_ud_neft
        res = [factor .* x for x in getproperty.(data, property)]
        # res_s, res_a = first.(res), last.(res)
        res_uu, res_ud = first.(res), last.(res)
        if ex_to_di
            # exchange_to_direct = representation == "sa" ? ex_to_di_sa : ex_to_di_ud
            # return exchange_to_direct(res_s, res_a)
            res_uu, res_ud = ex_to_di_ud(res_uu, res_ud)
        end
        if representation == "sa"
            res_s, res_a = Ver4.ud2sa(res_uu, res_ud)
            return res_s, res_a
        end
        return res_uu, res_ud
    end

    # Fs = (F↑↑ + F↑↓) / 2, Fa = (F↑↑ - F↑↓) / 2
    Fs2ds_neft, Fa2ds_neft =
        loaddata_oneloop_anychan_neft(:F2d, "sa", neft_factor_one_loop)

    # F↑↑ = Fs + Fa, F↑↓ = Fs - Fa
    Fuu2ds_neft, Fud2ds_neft =
        loaddata_oneloop_anychan_neft(:F2d, "ud", neft_factor_one_loop)

    # Load our data using jld2
    @load "one_loop_F_ours.jld2" rslist oneloop_sa_ours oneloop_ud_ours
    rslist_small = rslist
    function loaddata_oneloop_ours(property::Symbol, representation="sa", factor=1.0)
        @assert representation in ["sa", "ud"]
        data = representation == "sa" ? oneloop_sa_ours : oneloop_ud_ours
        res = [factor .* x for x in getproperty.(data, property)]
        res_s, res_a = first.(res), last.(res)
        return res_s, res_a
    end

    # Fs = (F↑↑ + F↑↓) / 2, Fa = (F↑↑ - F↑↓) / 2
    Fs1s_ours, Fa1s_ours = loaddata_oneloop_ours(:F1, "sa")
    Fs2vs_ours, Fa2vs_ours = loaddata_oneloop_ours(:F2v, "sa")
    Fs2ds_ours, Fa2ds_ours = loaddata_oneloop_ours(:F2d, "sa")
    Fs2cts_ours, Fa2cts_ours = loaddata_oneloop_ours(:F2ct, "sa")

    # F↑↑ = Fs + Fa, F↑↓ = Fs - Fa
    Fuu1s_ours, Fud1s_ours = loaddata_oneloop_ours(:F1, "ud")
    Fuu2vs_ours, Fud2vs_ours = loaddata_oneloop_ours(:F2v, "ud")
    Fuu2ds_ours, Fud2ds_ours = loaddata_oneloop_ours(:F2d, "ud")
    Fuu2cts_ours, Fud2cts_ours = loaddata_oneloop_ours(:F2ct, "ud")

    # Load our data for the four individual box diagram contributions
    @load "one_loop_box_diagrams_ours.jld2" rslist boxdiags_sa boxdiags_ud
    function loaddata_boxdiags_ours(property::Symbol, representation="sa", factor=1.0)
        @assert representation in ["sa", "ud"]
        data = representation == "sa" ? boxdiags_sa : boxdiags_ud
        res = [factor .* x for x in getproperty.(data, property)]
        res_s, res_a = first.(res), last.(res)
        return res_s, res_a
    end

    # Fs = (F↑↑ + F↑↓) / 2, Fa = (F↑↑ - F↑↓) / 2
    Fs2bs_DC_ours, Fa2bs_DC_ours = loaddata_boxdiags_ours(:F2b_direct_crossed, "sa")
    Fs2bs_DU_ours, Fa2bs_DU_ours = loaddata_boxdiags_ours(:F2b_direct_uncrossed, "sa")
    Fs2bs_EC_ours, Fa2bs_EC_ours = loaddata_boxdiags_ours(:F2b_exchange_crossed, "sa")
    Fs2bs_EU_ours, Fa2bs_EU_ours = loaddata_boxdiags_ours(:F2b_exchange_uncrossed, "sa")

    # F↑↑ = Fs + Fa, F↑↓ = Fs - Fa
    Fuu2bs_DC_ours, Fud2bs_DC_ours = loaddata_boxdiags_ours(:F2b_direct_crossed, "ud")
    Fuu2bs_DU_ours, Fud2bs_DU_ours = loaddata_boxdiags_ours(:F2b_direct_uncrossed, "ud")
    Fuu2bs_EC_ours, Fud2bs_EC_ours = loaddata_boxdiags_ours(:F2b_exchange_crossed, "ud")
    Fuu2bs_EU_ours, Fud2bs_EU_ours = loaddata_boxdiags_ours(:F2b_exchange_uncrossed, "ud")

    # F2d_PHr = F2_b_EC
    Fs2ds_phr_ours, Fa2ds_phr_ours = Fs2bs_EC_ours, Fa2bs_EC_ours
    Fuu2ds_phr_ours, Fud2ds_phr_ours = Fuu2bs_EC_ours, Fud2bs_EC_ours

    # F2d_PHEr = F2_v + F2_b_DC (exchange bubble cancelled exactly in CTs)
    Fs2ds_pher_ours = Fs2vs_ours + Fs2bs_DC_ours
    Fa2ds_pher_ours = Fa2vs_ours + Fa2bs_DC_ours
    Fuu2ds_pher_ours = Fuu2vs_ours + Fuu2bs_DC_ours
    Fud2ds_pher_ours = Fud2vs_ours + Fud2bs_DC_ours

    # F2d_PPr = F2_b_DU + F2_b_EU
    Fs2ds_ppr_ours = Fs2bs_DU_ours + Fs2bs_EU_ours
    Fa2ds_ppr_ours = Fa2bs_DU_ours + Fa2bs_EU_ours
    Fuu2ds_ppr_ours = Fuu2bs_DU_ours + Fuu2bs_EU_ours
    Fud2ds_ppr_ours = Fud2bs_DU_ours + Fud2bs_EU_ours
    
    # # F2d_PHr = F2_b_DC
    # Fs2ds_phr_ours, Fa2ds_phr_ours = Fs2bs_DC_ours, Fa2bs_DC_ours
    # Fuu2ds_phr_ours, Fud2ds_phr_ours = Fuu2bs_DC_ours, Fud2bs_DC_ours

    # # F2d_PHEr = F2_v + F2_b_EC
    # Fs2ds_pher_ours = Fs2vs_ours + Fs2bs_EC_ours
    # Fa2ds_pher_ours = Fa2vs_ours + Fa2bs_EC_ours
    # Fuu2ds_pher_ours = Fuu2vs_ours + Fuu2bs_EC_ours
    # Fud2ds_pher_ours = Fud2vs_ours + Fud2bs_EC_ours

    # # F2d_PPr = F2_b_DU + F2_b_EU
    # Fs2ds_ppr_ours = Fs2bs_DU_ours + Fs2bs_EU_ours
    # Fa2ds_ppr_ours = Fa2bs_DU_ours + Fa2bs_EU_ours
    # Fuu2ds_ppr_ours = Fuu2bs_DU_ours + Fuu2bs_EU_ours
    # Fud2ds_ppr_ours = Fud2bs_DU_ours + Fud2bs_EU_ours

    # ####################################################
    # ### Compare NEFT/our results for F2↑↑ by channel ###
    # ####################################################

    # # @load "F2_DC_theta12=piover2.jld2" F2_DC_θ12

    # fig, ax = plt.subplots(; figsize=(5, 5))
    # # Our data
    # ax.plot(
    #     spline(rslist_small, Fuu2ds_phr_ours)...;
    #     color=cdict["orange"],
    #     label="\$\\uparrow\\uparrow\$",
    # )
    # ax.plot(
    #     spline(rslist_small, Fud2ds_phr_ours)...;
    #     color=cdict["magenta"],
    #     label="\$\\uparrow\\downarrow\$",
    # )
    # # NEFT data
    # # Fuu2ds_phr_neft *= 4.0
    # mean_uu = Measurements.value.(Fuu2ds_phr_neft)
    # stddev = Measurements.uncertainty.(Fuu2ds_phr_neft)
    # ax.plot(
    #     plotdata(rslist_big, mean_uu, stddev)...;
    #     color=cdict["blue"],
    #     linestyle="--",
    #     label="\$\\uparrow\\uparrow\$ (NEFT)",
    # )
    # ax.scatter(rslist_big, mean_uu; color=cdict["blue"], s=4)
    # mean_ud = Measurements.value.(Fud2ds_phr_neft)
    # stddev = Measurements.uncertainty.(Fud2ds_phr_neft)
    # ax.plot(
    #     plotdata(rslist_big, mean_ud, stddev)...;
    #     color=cdict["cyan"],
    #     linestyle="--",
    #     label="\$\\uparrow\\downarrow\$ (NEFT)",
    # )
    # ax.scatter(rslist_big, mean_ud; color=cdict["cyan"], s=4)
    # # # 2D VEGAS (numerically exact result)
    # # mean_vegas = Measurements.value.(F2_DC_θ12)
    # # stddev_vegas = Measurements.uncertainty.(F2_DC_θ12)
    # # ax.plot(
    # #     plotdata(rslist_big, mean_vegas, stddev_vegas)...;
    # #     color=cdict["black"],
    # #     linestyle="-",
    # #     label="(2D VEGAS)",
    # # )
    # # ax.scatter(rslist_big, mean_vegas; color=cdict["black"], s=4)
    # ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\sigma_1 \\sigma_2}_{2} \\xi^2\$")
    # if isDynamic == false
    #     # ax.set_ylim(-0.9, 0.9)
    # end
    # ax.set_xlim(0, 10)
    # ax.legend(; ncol=2, fontsize=10, loc="best", columnspacing=0.5)
    # plt.tight_layout()
    # fig.savefig("test_$(leg_convention).pdf")

    # return

    ####################################################
    ### Compare NEFT/our results for F2↑↑ by channel ###
    ####################################################

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT data
    mean = Measurements.value.(Fuu2ds_phr_neft)
    stddev = Measurements.uncertainty.(Fuu2ds_phr_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["blue"],
        linestyle="--",
        label="\$F^{\\uparrow\\uparrow}_{2,\\text{PHr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["blue"], s=4)
    mean = Measurements.value.(Fuu2ds_pher_neft)
    stddev = Measurements.uncertainty.(Fuu2ds_pher_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["cyan"],
        linestyle="--",
        label="\$F^{\\uparrow\\uparrow}_{2,\\text{PHEr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["cyan"], s=4)
    mean = Measurements.value.(Fuu2ds_ppr_neft)
    stddev = Measurements.uncertainty.(Fuu2ds_ppr_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["teal"],
        linestyle="--",
        label="\$F^{\\uparrow\\uparrow}_{2,\\text{PPr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["teal"], s=4)
    mean = Measurements.value.(Fuu2ds_neft)
    stddev = Measurements.uncertainty.(Fuu2ds_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["grey"],
        linestyle="--",
        label="\$F^{\\uparrow\\uparrow}_{2}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["grey"], s=4)
    # Our data
    ax.plot(
        spline(rslist_small, Fuu2ds_phr_ours)...;
        color=cdict["orange"],
        label="\$F^{\\uparrow\\uparrow}_{2,\\text{PHr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fuu2ds_pher_ours)...;
        color=cdict["magenta"],
        label="\$F^{\\uparrow\\uparrow}_{2,\\text{PHEr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fuu2ds_ppr_ours)...;
        color=cdict["red"],
        label="\$F^{\\uparrow\\uparrow}_{2,\\text{PPr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fuu2ds_ours)...;
        color=cdict["black"],
        label="\$F^{\\uparrow\\uparrow}_{2}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\uparrow\\uparrow}_{2} \\xi^2\$")
    if isDynamic == false
        # ax.set_ylim(-0.9, 0.9)
    end
    ax.set_xlim(0, 10)
    ax.set_ylim(nothing, 0.76)
    ax.legend(; ncol=2, fontsize=10, loc="upper left", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneloop_F2_uu_channels_vs_rs$(interactionstr)_$(leg_convention).pdf")

    ####################################################
    ### Compare NEFT/our results for F2↑↓ by channel ###
    ####################################################

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT data
    mean = Measurements.value.(Fud2ds_phr_neft)
    stddev = Measurements.uncertainty.(Fud2ds_phr_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["blue"],
        linestyle="--",
        label="\$F^{\\uparrow\\downarrow}_{2,\\text{PHr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["blue"], s=4)
    mean = Measurements.value.(Fud2ds_pher_neft)
    stddev = Measurements.uncertainty.(Fud2ds_pher_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["cyan"],
        linestyle="--",
        label="\$F^{\\uparrow\\downarrow}_{2,\\text{PHEr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["cyan"], s=4)
    mean = Measurements.value.(Fud2ds_ppr_neft)
    stddev = Measurements.uncertainty.(Fud2ds_ppr_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["teal"],
        linestyle="--",
        label="\$F^{\\uparrow\\downarrow}_{2,\\text{PPr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["teal"], s=4)
    mean = Measurements.value.(Fud2ds_neft)
    stddev = Measurements.uncertainty.(Fud2ds_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["grey"],
        linestyle="--",
        label="\$F^{\\uparrow\\downarrow}_{2}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["grey"], s=4)
    # Our data
    ax.plot(
        spline(rslist_small, Fud2ds_phr_ours)...;
        color=cdict["orange"],
        label="\$F^{\\uparrow\\downarrow}_{2,\\text{PHr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fud2ds_pher_ours)...;
        color=cdict["magenta"],
        label="\$F^{\\uparrow\\downarrow}_{2,\\text{PHEr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fud2ds_ppr_ours)...;
        color=cdict["red"],
        label="\$F^{\\uparrow\\downarrow}_{2,\\text{PPr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fud2ds_ours)...;
        color=cdict["black"],
        label="\$F^{\\uparrow\\downarrow}_{2}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\uparrow\\downarrow}_{2} \\xi^2\$")
    if isDynamic == false
        # ax.set_ylim(-0.9, 0.9)
    end
    ax.set_xlim(0, 10)
    ax.set_ylim(nothing, 0.95)
    ax.legend(; ncol=2, fontsize=10, loc="upper left", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneloop_F2_ud_channels_vs_rs$(interactionstr)_$(leg_convention).pdf")

    ###################################################
    ### Compare NEFT/our results for F2s by channel ###
    ###################################################

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT data
    mean = Measurements.value.(Fs2ds_phr_neft)
    stddev = Measurements.uncertainty.(Fs2ds_phr_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["blue"],
        linestyle="--",
        label="\$F^{s}_{2,\\text{PHr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["blue"], s=4)
    mean = Measurements.value.(Fs2ds_pher_neft)
    stddev = Measurements.uncertainty.(Fs2ds_pher_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["cyan"],
        linestyle="--",
        label="\$F^{s}_{2,\\text{PHEr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["cyan"], s=4)
    mean = Measurements.value.(Fs2ds_ppr_neft)
    stddev = Measurements.uncertainty.(Fs2ds_ppr_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["teal"],
        linestyle="--",
        label="\$F^{s}_{2,\\text{PPr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["teal"], s=4)
    mean = Measurements.value.(Fs2ds_neft)
    stddev = Measurements.uncertainty.(Fs2ds_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["grey"],
        linestyle="--",
        label="\$F^{s}_{2}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["grey"], s=4)
    # Our data
    ax.plot(
        spline(rslist_small, Fs2ds_phr_ours)...;
        color=cdict["orange"],
        label="\$F^{s}_{2,\\text{PHr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fs2ds_pher_ours)...;
        color=cdict["magenta"],
        label="\$F^{s}_{2,\\text{PHEr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fs2ds_ppr_ours)...;
        color=cdict["red"],
        label="\$F^{s}_{2,\\text{PPr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fs2ds_ours)...;
        color=cdict["black"],
        label="\$F^{s}_{2}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{s}_{2} \\xi^2\$")
    if isDynamic == false
        # ax.set_ylim(-0.9, 0.9)
    end
    ax.set_xlim(0, 10)
    ax.set_ylim(nothing, 0.58)
    ax.legend(; ncol=2, fontsize=10, loc="upper left", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneloop_F2_s_channels_vs_rs$(interactionstr)_$(leg_convention).pdf")

    ###################################################
    ### Compare NEFT/our results for F2a by channel ###
    ###################################################

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT data
    mean = Measurements.value.(Fa2ds_phr_neft)
    stddev = Measurements.uncertainty.(Fa2ds_phr_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["blue"],
        linestyle="--",
        label="\$F^{a}_{2,\\text{PHr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["blue"], s=4)
    mean = Measurements.value.(Fa2ds_pher_neft)
    stddev = Measurements.uncertainty.(Fa2ds_pher_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["cyan"],
        linestyle="--",
        label="\$F^{a}_{2,\\text{PHEr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["cyan"], s=4)
    mean = Measurements.value.(Fa2ds_ppr_neft)
    stddev = Measurements.uncertainty.(Fa2ds_ppr_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["teal"],
        linestyle="--",
        label="\$F^{a}_{2,\\text{PPr}}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["teal"], s=4)
    mean = Measurements.value.(Fa2ds_neft)
    stddev = Measurements.uncertainty.(Fa2ds_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["grey"],
        linestyle="--",
        label="\$F^{a}_{2}\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["grey"], s=4)
    # Our data
    ax.plot(
        spline(rslist_small, Fa2ds_phr_ours)...;
        color=cdict["orange"],
        label="\$F^{a}_{2,\\text{PHr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fa2ds_pher_ours)...;
        color=cdict["magenta"],
        label="\$F^{a}_{2,\\text{PHEr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fa2ds_ppr_ours)...;
        color=cdict["red"],
        label="\$F^{a}_{2,\\text{PPr}}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.plot(
        spline(rslist_small, Fa2ds_ours)...;
        color=cdict["black"],
        label="\$F^{a}_{2}(\\theta_{12} = \\frac{\\pi}{2})\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{a}_{2} \\xi^2\$")
    if isDynamic == false
        # ax.set_ylim(-0.9, 0.9)
    end
    ax.set_xlim(0, 10)
    ax.set_ylim(nothing, 0.48)
    ax.legend(; ncol=2, fontsize=10, loc="upper left", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneloop_F2_a_channels_vs_rs$(interactionstr)_$(leg_convention).pdf")

    plt.close("all")
    return
end

function main()
    plot_F_by_channel()
    return
end

main()
