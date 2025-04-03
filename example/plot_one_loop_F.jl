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

function plot_vertex_matsubara_summand(param::OneLoopParams)
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF, isDynamic = param
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
        coordinates = [
            [q, 0, rand(0:(2π))],  # q || k1 (equivalent to q || k2)
            [q, π, rand(0:(2π))],  # q || -k1 (equivalent to q || -k2)
            [q, 3π / 4, π],      # q maximally spaced from (anti-bisects) k1 & k2
            [q, π / 4, 0],       # q bisects k1 & k2
            [q, π / 2, π / 2],   # q || y-axis
            [q, 2π / 3, π / 3],  # general asymmetrically placed q #1
        ]
        labels = [
            "\$\\theta=0, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\pi, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\frac{3\\pi}{4}, \\varphi=\\pi\$",
            "\$\\theta=\\frac{\\pi}{4}, \\varphi=0\$",
            "\$\\theta=\\frac{\\pi}{2}, \\varphi=\\frac{\\pi}{2}\$",
            "\$\\theta=\\frac{2\\pi}{3}, \\varphi=\\frac{\\pi}{3}\$",
        ]
        # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
        fig, ax = plt.subplots(; figsize=(5, 5))
        vms = (0:Mmax) * (2π / β)
        for (i, (label, coord)) in enumerate(zip(labels, coordinates))
            summand = vertex_matsubara_summand(param, coord...)
            ax.plot(
                vms / EF,
                real(summand);
                color=color[i],
                label=label,
                marker="o",
                markersize=4,
                markerfacecolor="none",
            )
        end
        ax.set_xlabel("\$i\\nu_m / \\epsilon_F\$")
        ax.set_ylabel("\$S^\\text{v}_\\mathbf{q}(i\\nu_m)\$")
        ax.set_xlim(0, 4)
        ax.legend(;
            loc="best",
            fontsize=14,
            title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
        )
        # fig.tight_layout()
        kindstr = param.Fs == 0.0 ? "rpa" : "kop"
        interactionstr = isDynamic ? "" : "yukawa"
        fig.savefig("vertex_matsubara_summand_q=$(qstr)_$(kindstr)_$(interactionstr).pdf")
    end
    return
end

function plot_vertex_matsubara_sum(param::OneLoopParams)
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF, θgrid, isDynamic = param
    clabels = ["Re", "Im"]
    cparts = [real, imag]
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (clabel, cpart) in zip(clabels, cparts)
        for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
            phis = [0, π / 4, π / 2, 3π / 4, π]
            labels = [
                "\$\\varphi=0\$",
                "\$\\varphi=\\frac{\\pi}{4}\$",
                "\$\\varphi=\\frac{\\pi}{2}\$",
                "\$\\varphi=\\frac{3\\pi}{4}\$",
                "\$\\varphi=\\pi\$",
            ]
            # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
            fig, ax = plt.subplots(; figsize=(5, 5))
            for (i, (label, φ)) in enumerate(zip(labels, phis))
                matsubara_sum_vs_θ =
                    [vertex_matsubara_sum(param, q, θ, φ) for θ in θgrid.grid]
                ax.plot(
                    θgrid.grid,
                    cpart(matsubara_sum_vs_θ);
                    color=color[i],
                    label=label,
                    # marker="o",
                    # markersize=4,
                    # markerfacecolor="none",
                )
            end
            ax.set_xlabel("\$\\theta\$")
            ax.set_ylabel("\$S_\\text{v}(q, \\theta, \\phi; \\theta_{12} = \\pi / 2)\$")
            ax.set_xlim(0, π)
            ax.set_xticks([0, π / 4, π / 2, 3π / 4, π])
            ax.set_xticklabels([
                "0",
                "\$\\frac{\\pi}{4}\$",
                "\$\\frac{\\pi}{2}\$",
                "\$\\frac{3\\pi}{4}\$",
                "\$\\pi\$",
            ])
            ax.legend(;
                loc="best",
                fontsize=14,
                title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
                ncol=2,
            )
            # fig.tight_layout()
            kindstr = param.Fs == 0.0 ? "rpa" : "kop"
            interactionstr = isDynamic ? "" : "yukawa"
            fig.savefig(
                "$(clabel)_vertex_matsubara_sum_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
            )
        end
    end
end

function plot_box_matsubara_summand(param::OneLoopParams; ftype="Fs")
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF, isDynamic = param
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
        coordinates = [
            [q, 0, rand(0:(2π))],  # q || k1 (equivalent to q || k2)
            [q, π, rand(0:(2π))],  # q || -k1 (equivalent to q || -k2)
            [q, 3π / 4, π],      # q maximally spaced from (anti-bisects) k1 & k2
            [q, π / 4, 0],       # q bisects k1 & k2
            [q, π / 2, π / 2],   # q || y-axis
            [q, 2π / 3, π / 3],  # general asymmetrically placed q #1
        ]
        labels = [
            "\$\\theta=0, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\pi, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\frac{3\\pi}{4}, \\varphi=\\pi\$",
            "\$\\theta=\\frac{\\pi}{4}, \\varphi=0\$",
            "\$\\theta=\\frac{\\pi}{2}, \\varphi=\\frac{\\pi}{2}\$",
            "\$\\theta=\\frac{2\\pi}{3}, \\varphi=\\frac{\\pi}{3}\$",
        ]
        # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
        fig, ax = plt.subplots(; figsize=(5, 5))
        vms = (0:Mmax) * (2π / β)
        for (i, (label, coord)) in enumerate(zip(labels, coordinates))
            summand = box_matsubara_summand(param, coord..., ftype)
            ax.plot(
                vms / EF,
                real(summand);
                color=color[i],
                label=label,
                marker="o",
                markersize=4,
                markerfacecolor="none",
            )
        end
        ax.set_xlabel("\$i\\nu_m / \\epsilon_F\$")
        ax.set_ylabel("\$S^\\text{b}_\\mathbf{q}(i\\nu_m)\$")
        ax.set_xlim(0, 4)
        ax.legend(;
            loc="best",
            fontsize=14,
            title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
        )
        # fig.tight_layout()
        kindstr = param.Fs == 0.0 ? "rpa" : "kop"
        interactionstr = isDynamic ? "" : "yukawa"
        fig.savefig(
            "box_matsubara_summand_$(ftype)_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
        )
    end
    return
end

function plot_box_matsubara_sum(param::OneLoopParams; ftype="Fs")
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF, θgrid, isDynamic = param
    clabels = ["Re", "Im"]
    cparts = [real, imag]
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (clabel, cpart) in zip(clabels, cparts)
        for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
            phis = [0, π / 4, π / 2, 3π / 4, π]
            labels = [
                "\$\\varphi=0\$",
                "\$\\varphi=\\frac{\\pi}{4}\$",
                "\$\\varphi=\\frac{\\pi}{2}\$",
                "\$\\varphi=\\frac{3\\pi}{4}\$",
                "\$\\varphi=\\pi\$",
            ]
            # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
            fig, ax = plt.subplots(; figsize=(5, 5))
            for (i, (label, φ)) in enumerate(zip(labels, phis))
                matsubara_sum_vs_θ =
                    [box_matsubara_sum(param, q, θ, φ; ftype=ftype) for θ in θgrid.grid]
                ax.plot(
                    θgrid.grid,
                    cpart(matsubara_sum_vs_θ);
                    color=color[i],
                    label=label,
                    # marker="o",
                    # markersize=4,
                    # markerfacecolor="none",
                )
            end
            ax.set_xlabel("\$\\theta\$")
            ax.set_ylabel("\$S_\\text{b}(q, \\theta, \\phi; \\theta_{12} = \\pi / 2)\$")
            ax.set_xlim(0, π)
            ax.set_xticks([0, π / 4, π / 2, 3π / 4, π])
            ax.set_xticklabels([
                "0",
                "\$\\frac{\\pi}{4}\$",
                "\$\\frac{\\pi}{2}\$",
                "\$\\frac{3\\pi}{4}\$",
                "\$\\pi\$",
            ])
            ax.legend(;
                loc="best",
                fontsize=14,
                title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
                ncol=2,
            )
            # fig.tight_layout()
            kindstr = param.Fs == 0.0 ? "rpa" : "kop"
            interactionstr = isDynamic ? "" : "yukawa"
            fig.savefig(
                "$(clabel)_box_matsubara_sum_$(ftype)_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
            )
        end
    end
end

function plot_direct_box_matsubara_summand(param::OneLoopParams; which="both")
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @assert which in ["both", "ladder", "crossed"]
    @unpack β, kF, EF, Mmax, Q_CUTOFF, isDynamic = param
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
        coordinates = [
            [q, 0, rand(0:(2π))],  # q || k1 (equivalent to q || k2)
            [q, π, rand(0:(2π))],  # q || -k1 (equivalent to q || -k2)
            [q, 3π / 4, π],      # q maximally spaced from (anti-bisects) k1 & k2
            [q, π / 4, 0],       # q bisects k1 & k2
            [q, π / 2, π / 2],   # q || y-axis
            [q, 2π / 3, π / 3],  # general asymmetrically placed q #1
        ]
        labels = [
            "\$\\theta=0, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\pi, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\frac{3\\pi}{4}, \\varphi=\\pi\$",
            "\$\\theta=\\frac{\\pi}{4}, \\varphi=0\$",
            "\$\\theta=\\frac{\\pi}{2}, \\varphi=\\frac{\\pi}{2}\$",
            "\$\\theta=\\frac{2\\pi}{3}, \\varphi=\\frac{\\pi}{3}\$",
        ]
        # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
        fig, ax = plt.subplots(; figsize=(5, 5))
        vms = (0:Mmax) * (2π / β)
        for (i, (label, coord)) in enumerate(zip(labels, coordinates))
            summand = direct_box_matsubara_summand(param, coord...; which=which)
            ax.plot(
                vms / EF,
                real(summand);
                color=color[i],
                label=label,
                marker="o",
                markersize=4,
                markerfacecolor="none",
            )
        end
        ax.set_xlabel("\$i\\nu_m / \\epsilon_F\$")
        ax.set_ylabel("\$S^\\text{b,Di}_\\mathbf{q}(i\\nu_m)\$")
        ax.set_xlim(0, 4)
        ax.legend(;
            loc="best",
            fontsize=14,
            title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
        )
        # fig.tight_layout()
        kindstr = param.Fs == 0.0 ? "rpa" : "kop"
        interactionstr = isDynamic ? "" : "yukawa"
        fig.savefig(
            "$(which)_direct_box_matsubara_summand_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
        )
    end
    return
end

function plot_direct_box_matsubara_sum(param::OneLoopParams; which="both")
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @assert which in ["both", "ladder", "crossed"]
    @unpack β, kF, EF, Mmax, Q_CUTOFF, θgrid, isDynamic = param
    clabels = ["Re", "Im"]
    cparts = [real, imag]
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (clabel, cpart) in zip(clabels, cparts)
        for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
            phis = [0, π / 4, π / 2, 3π / 4, π]
            labels = [
                "\$\\varphi=0\$",
                "\$\\varphi=\\frac{\\pi}{4}\$",
                "\$\\varphi=\\frac{\\pi}{2}\$",
                "\$\\varphi=\\frac{3\\pi}{4}\$",
                "\$\\varphi=\\pi\$",
            ]
            # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
            fig, ax = plt.subplots(; figsize=(5, 5))
            for (i, (label, φ)) in enumerate(zip(labels, phis))
                matsubara_sum_vs_θ = [
                    direct_box_matsubara_sum(param, q, θ, φ; which=which) for
                    θ in θgrid.grid
                ]
                ax.plot(
                    θgrid.grid,
                    cpart(matsubara_sum_vs_θ);
                    color=color[i],
                    label=label,
                    # marker="o",
                    # markersize=4,
                    # markerfacecolor="none",
                )
            end
            ax.set_xlabel("\$\\theta\$")
            ax.set_ylabel("\$S_\\text{b,Di}(q, \\theta, \\phi; \\theta_{12} = \\pi / 2)\$")
            ax.set_xlim(0, π)
            ax.set_xticks([0, π / 4, π / 2, 3π / 4, π])
            ax.set_xticklabels([
                "0",
                "\$\\frac{\\pi}{4}\$",
                "\$\\frac{\\pi}{2}\$",
                "\$\\frac{3\\pi}{4}\$",
                "\$\\pi\$",
            ])
            ax.legend(;
                loc="best",
                fontsize=14,
                title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
                ncol=2,
            )
            # fig.tight_layout()
            kindstr = param.Fs == 0.0 ? "rpa" : "kop"
            interactionstr = isDynamic ? "" : "yukawa"
            fig.savefig(
                "$(clabel)_$(which)_direct_box_matsubara_sum_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
            )
        end
    end
end

function plot_extras(rslist, ftype)
    ftypestr = ftype == "Fs" ? "F^{s}" : "F^{a}"

    interactionstr = isDynamic ? "" : "_yukawa"
    zstr = z_renorm ? "_z_renorm" : ""

    # nk ≈ na ≈ 100 is sufficiently converged for all relevant euv/rtol
    Nk, Ok = 7, 6
    Na, Oa = 8, 7

    # DLR parameters for which r(q, 0) is smooth in the q → 0 limit (tested for rs = 1, 10)
    euv = 10.0
    rtol = 1e-7

    # Load exact data using np
    FsDMCs = np.load("FsDMCs.npy")
    FaDMCs = np.load("FaDMCs.npy")
    F1s_exact = np.load("F1s_exact.npy")
    F2cts_exact = np.load("F2cts_exact.npy")

    # Load our data using np
    F1s = np.load("F1s.npy")
    F2vs = np.load("F2vs.npy")
    F2bs = np.load("F2bs.npy")
    F2cts = np.load("F2cts.npy")
    F2zs = np.load("F2zs.npy")
    F2s = np.load("F2s.npy")

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    ax.errorbar(
        rslist,
        # 1 .+ Measurements.value.(F1NEFTs),
        Measurements.value.(F1NEFTs),
        Measurements.uncertainty.(F1NEFTs);
        label="\${$ftypestr}_1 \\xi\$ (NEFT)",
        capthick=1,
        capsize=4,
        fmt="o",
        ms=5,
        color=cdict["cyan"],
    )
    ax.errorbar(
        rslist,
        # 1 .+ Measurements.value.(F2NEFTs),
        Measurements.value.(F2NEFTs),
        Measurements.uncertainty.(F2NEFTs);
        label="\${$ftypestr}_2 \\xi^2\$ (NEFT)",
        capthick=1,
        capsize=4,
        fmt="o",
        ms=5,
        color=cdict["magenta"],
    )
    ax.errorbar(
        rslist,
        # 1 .+ Measurements.value.(FtotalNEFTs),
        Measurements.value.(FtotalNEFTs),
        Measurements.uncertainty.(FtotalNEFTs);
        label="\${$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$ (NEFT)",
        # label="\$\\kappa_0 / \\kappa \\approx 1 + \${$ftypestr}_1 \\xi + \${$ftypestr}_2 \\xi^2\$ (NEFT)",
        capthick=1,
        capsize=4,
        fmt="o",
        ms=5,
        color=cdict["teal"],
    )
    ax.set_xlabel("\$r_s\$")
    ax.set_ylabel("\$$ftypestr\$")
    # ax.set_ylabel("\${\\kappa_0}/{\\kappa} \\approx 1 + F^s\$")
    if isDynamic == false && ftype == "Fs"
        # ax.set_ylim(0.58, 1.42)
        ax.set_ylim(-0.42, 0.42)
    end
    # ax.set_ylabel("\$F^s\$")
    ax.legend(; ncol=2, fontsize=12, loc="best", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_$(ftype)_yukawa_vs_rs$(interactionstr)_benchmark.pdf")
    plt.close(fig)

    FsDMCs, FaDMCs, F1NEFTs, Fs2NEFTs, Fa2NEFTs =
        get_yukawa_one_loop_neft(rslist, beta; neval=1e5)
    FstotalNEFTs = F1NEFTs .+ Fs2NEFTs
    FatotalNEFTs = F1NEFTs .+ Fa2NEFTs
    F2NEFTs = ftype == "Fs" ? Fs2NEFTs : Fa2NEFTs
    FtotalNEFTs = ftype == "Fs" ? FstotalNEFTs : FatotalNEFTs

    # Get Thomas-Fermi result for F1 using exact expression
    rs_exact = LinRange(0, 10, 1000)
    F1s_exact = get_F1_TF.(rs_exact)

    # Get Thomas-Fermi result for F1 using exact expression
    function F2ct_exact(rs)
        xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
        rstilde = rs * alpha_ueg / π
        F1 = get_F1_TF(rs)
        A = Interp.integrate1D(
            OneLoopFermiLiquid.lindhard.(xgrid) .* (xgrid * rstilde) ./
            (xgrid .^ 2 .+ rstilde),
            xgrid,
        )
        B = Interp.integrate1D(OneLoopFermiLiquid.lindhard.(xgrid) .* xgrid, xgrid)
        return 2 * F1 * A + F1^2 * B
    end
    F2cts_exact = F2ct_exact.(rs_exact)

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    ax.errorbar(
        rslist,
        # 1 .+ Measurements.value.(F1NEFTs),
        Measurements.value.(F1NEFTs),
        Measurements.uncertainty.(F1NEFTs);
        label="\${$ftypestr}_1 \\xi\$ (NEFT)",
        capthick=1,
        capsize=4,
        fmt="o",
        ms=5,
        color=cdict["cyan"],
    )
    ax.errorbar(
        rslist,
        # 1 .+ Measurements.value.(F2NEFTs),
        Measurements.value.(F2NEFTs),
        Measurements.uncertainty.(F2NEFTs);
        label="\${$ftypestr}_2 \\xi^2\$ (NEFT)",
        capthick=1,
        capsize=4,
        fmt="o",
        ms=5,
        color=cdict["magenta"],
    )
    ax.errorbar(
        rslist,
        # 1 .+ Measurements.value.(FtotalNEFTs),
        Measurements.value.(FtotalNEFTs),
        Measurements.uncertainty.(FtotalNEFTs);
        label="\${$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$ (NEFT)",
        # label="\$\\kappa_0 / \\kappa \\approx 1 + \${$ftypestr}_1 \\xi + \${$ftypestr}_2 \\xi^2\$ (NEFT)",
        capthick=1,
        capsize=4,
        fmt="o",
        ms=5,
        color=cdict["teal"],
    )
    # Our results
    error = 1e-6 * ones(length(rslist))
    fp1label = ftype == "Fs" ? "\$\\kappa_0 / \\kappa\$" : "\$\\chi_0 / \\chi\$"
    fdata = ftype == "Fs" ? Fs_DMCs : Fa_DMCs
    fp1type = ftype == "Fs" ? "kappa0_over_kappa" : "chi0_over_chi"
    if isDynamic
        ax.plot(
            # spline(rslist, 1.0 .+ fdata, error)...;
            spline(rslist, fdata, error)...;
            label=fp1label,
            color=cdict["grey"],
        )
    end
    ax.plot(
        # spline(rslist, 1.0 .+ F1s, error)...;
        # label="\$1 + {$ftypestr}_1 \\xi\$",
        spline(rslist, F1s, error)...;
        label="\${$ftypestr}_1 \\xi\$",
        color=cdict["orange"],
    )
    ax.plot(
        # spline(rslist, 1.0 .+ F2s, error)...;
        # label="\$1 + {$ftypestr}_2 \\xi^2\$",
        spline(rslist, F2s, error)...;
        label="\${$ftypestr}_2 \\xi^2\$",
        color=cdict["blue"],
    )
    ax.plot(
        # spline(rslist, 1.0 .+ F1s .+ F2s, error)...;
        # label="\$1 + {$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$",
        spline(rslist, F1s .+ F2s, error)...;
        label="\${$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$",
        color=cdict["black"],
    )
    ax.set_xlabel("\$r_s\$")
    ax.set_ylabel("\$$ftypestr\$")
    # ax.set_ylabel("\${\\kappa_0}/{\\kappa} \\approx 1 + F^s\$")
    if isDynamic == false && ftype == "Fs"
        # ax.set_ylim(0.58, 1.42)
        ax.set_ylim(-0.42, 0.42)
    end
    # ax.set_ylabel("\$F^s\$")
    ax.legend(; ncol=2, fontsize=12, loc="best", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_$(ftype)_yukawa_vs_rs$(interactionstr).pdf")
    # fig.savefig("kappa0_over_kappa_yukawa_vs_rs.pdf")
    plt.close(fig)

    # Plot spline fits to data vs rs
    fig, ax = plt.subplots(; figsize=(5, 5))
    error = 1e-6 * ones(length(rslist))
    flabel = ftype == "Fs" ? "\$F^{s}_{\\text{DMC}}\$" : "\$F^{a}_{\\text{DMC}}\$"
    fdata = ftype == "Fs" ? Fs_DMCs : Fa_DMCs
    if isDynamic
        ax.plot(spline(rslist, fdata, error)...; label=flabel, color=cdict["grey"])
    end
    ax.plot(
        spline(rslist, F1s, error)...;
        label="\${$ftypestr}_1 \\xi\$",
        color=cdict["orange"],
    )
    ax.plot(
        spline(rslist, F2s, error)...;
        label="\${$ftypestr}_2 \\xi^2\$",
        color=cdict["blue"],
    )
    ax.plot(
        spline(rslist, F1s .+ F2s, error)...;
        label="\${$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$",
        color=cdict["black"],
    )
    ax.plot(
        spline(rslist, F2vs, error)...;
        label="\${$ftypestr}_{\\text{v},2}\$",
        color=cdict["cyan"],
    )
    ax.plot(
        spline(rslist, F2bs, error)...;
        label="\${$ftypestr}_{\\text{b},2}\$",
        color=cdict["magenta"],
    )
    ax.plot(
        spline(rslist, F2cts, error)...;
        label="\${$ftypestr}_{\\text{ct},2}\$",
        color=cdict["teal"],
    )
    # Exact CT expression
    ax.plot(
        rs_exact,
        F2cts_exact,
        "--";
        label="\${$ftypestr}_{\\text{ct},2}\$ (exact)",
        color=cdict["red"],
    )
    if z_renorm
        ax.plot(
            spline(rslist, F2zs, error)...;
            label="\${$ftypestr}_{\\text{z},2}\$",
            color=cdict["red"],
        )
    end
    ax.set_xlabel("\$r_s\$")
    ax.set_xlim(0, maximum(rslist))
    if isDynamic && ftype == "Fs"
        ax.set_ylim(-5.5, 5.5)
    elseif ftype == "Fs"
        ax.set_ylim(-0.42, 0.42)
    end
    ax.legend(;
        ncol=2,
        loc="upper left",
        fontsize=12,
        title_fontsize=16,
        title=isDynamic ?
              "\$\\Lambda_\\text{UV} = $(Int(round(euv)))\\epsilon_F, \\varepsilon = 10^{$(Int(round(log10(rtol))))}\$" :
              nothing,
    )
    fig.tight_layout()
    fig.savefig(
        "oneshot_one_loop_$(ftype)_vs_rs_euv=$(euv)_rtol=$(rtol)$(zstr)$(interactionstr).pdf",
    )
    plt.close(fig)

    # Plot spline fits to just 1+FDMC, 1+F1, 1+F2 and 1+F1+F2 vs rs
    fig, ax = plt.subplots(; figsize=(5, 5))
    error = 1e-6 * ones(length(rslist))
    fp1label = ftype == "Fs" ? "\$\\kappa_0 / \\kappa\$" : "\$\\chi_0 / \\chi\$"
    fdata = ftype == "Fs" ? Fs_DMCs : Fa_DMCs
    fp1type = ftype == "Fs" ? "kappa0_over_kappa" : "chi0_over_chi"
    if isDynamic
        # ax.plot(spline(rslist, 1.0 .+ fdata, error)...; label=flabel, color=cdict["grey"])
        ax.plot(spline(rslist, 1.0 .+ fdata, error)...; label=fp1label, color=cdict["grey"])
    end
    ax.plot(
        spline(rslist, 1.0 .+ F1s, error)...;
        label="\$1 + {$ftypestr}_1 \\xi\$",
        color=cdict["orange"],
    )
    ax.plot(
        spline(rslist, 1.0 .+ F2s, error)...;
        label="\$1 + {$ftypestr}_2 \\xi^2\$",
        color=cdict["blue"],
    )
    ax.plot(
        spline(rslist, 1.0 .+ F1s .+ F2s, error)...;
        label="\$1 + {$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$",
        color=cdict["black"],
    )
    ax.set_xlabel("\$r_s\$")
    ax.set_xlim(0, maximum(rslist))
    if isDynamic && ftype == "Fs"
        ax.set_ylim(-0.5, 2.0)
    end
    ax.legend(;
        ncol=2,
        loc="best",
        fontsize=12,
        title_fontsize=16,
        title=isDynamic ?
              "\$\\Lambda_\\text{UV} = $(Int(round(euv)))\\epsilon_F, \\varepsilon = 10^{$(Int(round(log10(rtol))))}\$" :
              nothing,
    )
    fig.tight_layout()
    fig.savefig(
        # "oneshot_one_loop_$(ftype)_vs_rs_euv=$(euv)_rtol=$(rtol)$(zstr)$(interactionstr)_zoom.pdf",
        "oneshot_one_loop_$(fp1type)_vs_rs_euv=$(euv)_rtol=$(rtol)$(zstr)$(interactionstr)_zoom.pdf",
    )
    plt.close(fig)
end

function plot_F_uu_ud_NEFT(isDynamic, z_renorm)
    # ftypestr = ftype == "Fs" ? "F^{s}" : "F^{a}"

    # NOTE: NEFT tree-level data is missing an overall minus sign (change sign convention for F).
    #       However, we now use the exact expressions for F1 and F2ct to isolate the problem with F2d.
    neft_factor_tree_level = 1.0
    neft_factor_one_loop = 1.0

    chan = :PH
    neft_splines = true

    function plotdata(x, y, e; error_multiplier=1.0)
        if neft_splines == false
            return x, y
        end
        xspline, yspline = spline(x, y, error_multiplier * e)
        return xspline, yspline
    end

    interactionstr = isDynamic ? "" : "_yukawa"
    zstr = z_renorm ? "_z_renorm" : ""

    # Load NEFT benchmark data using jld2
    @load "one_loop_F_neft_$(chan).jld2" rslist oneloop_sa_neft oneloop_ud_neft
    function getprop_sa_neft(p::Symbol, factor=1.0)
        res = [factor .* x for x in getproperty.(oneloop_sa_neft, p)]
        res_s, res_a = first.(res), last.(res)
        return res_s, res_a
    end
    function getprop_ud_neft(p::Symbol, factor=1.0)
        res = [factor .* x for x in getproperty.(oneloop_ud_neft, p)]
        res_uu, res_ud = first.(res), last.(res)
        return res_uu, res_ud
    end
    rslist_big = rslist

    # Fs = (F↑↑ + F↑↓) / 2, Fa = (F↑↑ - F↑↓) / 2,
    Fs1s_neft, Fa1s_neft = getprop_sa_neft(:F1, neft_factor_tree_level)
    Fs2vs_neft, Fa2vs_neft = getprop_sa_neft(:F2v, neft_factor_one_loop)
    Fs2bs_neft, Fa2bs_neft = getprop_sa_neft(:F2b, neft_factor_one_loop)
    # Fs2bubbles_neft, Fa2bubbles_neft = getprop_sa_neft(:F2bubble, neft_factor_one_loop)
    Fs2ds_neft, Fa2ds_neft = getprop_sa_neft(:F2d, neft_factor_one_loop)
    Fs2cts_neft, Fa2cts_neft = getprop_sa_neft(:F2ct, neft_factor_one_loop)
    # Fs2bubblects_neft, Fa2bubblects_neft = getprop_sa_neft(:F2bubblect, neft_factor_one_loop)
    Fs2zs_neft, Fa2zs_neft = getprop_sa_neft(:F2z, neft_factor_one_loop)
    Fs2s_neft, Fa2s_neft = getprop_sa_neft(:F2, neft_factor_one_loop)
    Fss_neft, Fas_neft = getprop_sa_neft(:F, neft_factor_one_loop)

    # F↑↑ = Fs + Fa, F↑↓ = Fs - Fa
    Fuu1s_neft, Fud1s_neft = getprop_ud_neft(:F1, neft_factor_tree_level)
    Fuu2vs_neft, Fud2vs_neft = getprop_ud_neft(:F2v, neft_factor_one_loop)
    Fuu2bs_neft, Fud2bs_neft = getprop_ud_neft(:F2b, neft_factor_one_loop)
    # Fuu2bubbles_neft, Fud2bubbles_neft = getprop_ud_neft(:F2bubble, neft_factor_one_loop)
    Fuu2ds_neft, Fud2ds_neft = getprop_ud_neft(:F2d, neft_factor_one_loop)
    Fuu2cts_neft, Fud2cts_neft = getprop_ud_neft(:F2ct, neft_factor_one_loop)
    # Fuu2bubblects_neft, Fud2bubblects_neft = getprop_ud_neft(:F2bubblect, neft_factor_one_loop)
    Fuu2zs_neft, Fud2zs_neft = getprop_ud_neft(:F2z, neft_factor_one_loop)
    Fuu2s_neft, Fud2s_neft = getprop_ud_neft(:F2, neft_factor_one_loop)
    Fuus_neft, Fuds_neft = getprop_ud_neft(:F, neft_factor_one_loop)

    # Load our data using jld2
    @load "one_loop_F_ours.jld2" rslist oneloop_sa_ours oneloop_ud_ours
    function getprop_sa_ours(p::Symbol)
        res = getproperty.(oneloop_sa_ours, p)
        res_s, res_a = first.(res), last.(res)
        return res_s, res_a
    end
    function getprop_ud_ours(p::Symbol)
        res = getproperty.(oneloop_ud_ours, p)
        res_uu, res_ud = first.(res), last.(res)
        return res_uu, res_ud
    end
    rslist_small = rslist

    # Fs = (F↑↑ + F↑↓) / 2, Fa = (F↑↑ - F↑↓) / 2,
    Fs1s_ours, Fa1s_ours = getprop_sa_ours(:F1)
    Fs2vs_ours, Fa2vs_ours = getprop_sa_ours(:F2v)
    Fs2bs_ours, Fa2bs_ours = getprop_sa_ours(:F2b)
    # Fs2bubbles_ours, Fa2bubbles_ours = getprop_sa_ours(:F2bubble)
    Fs2ds_ours, Fa2ds_ours = getprop_sa_ours(:F2d)
    Fs2cts_ours, Fa2cts_ours = getprop_sa_ours(:F2ct)
    # Fs2bubblects_ours, Fa2bubblects_ours = getprop_sa_ours(:F2bubblect)
    Fs2zs_ours, Fa2zs_ours = getprop_sa_ours(:F2z)
    Fs2s_ours, Fa2s_ours = getprop_sa_ours(:F2)
    Fss_ours, Fas_ours = getprop_sa_ours(:F)

    # F↑↑ = Fs + Fa, F↑↓ = Fs - Fa
    Fuu1s_ours, Fud1s_ours = getprop_ud_ours(:F1)
    Fuu2vs_ours, Fud2vs_ours = getprop_ud_ours(:F2v)
    Fuu2bs_ours, Fud2bs_ours = getprop_ud_ours(:F2b)
    # Fuu2bubbles_ours, Fud2bubbles_ours = getprop_ud_ours(:F2bubble)
    Fuu2ds_ours, Fud2ds_ours = getprop_ud_ours(:F2d)
    Fuu2cts_ours, Fud2cts_ours = getprop_ud_ours(:F2ct)
    # Fuu2bubblects_ours, Fud2bubblects_ours = getprop_ud_ours(:F2bubblect)
    Fuu2zs_ours, Fud2zs_ours = getprop_ud_ours(:F2z)
    Fuu2s_ours, Fud2s_ours = getprop_ud_ours(:F2)
    Fuus_ours, Fuds_ours = getprop_ud_ours(:F)

    ############################
    ### Fig 1: F↑↑/F↑↓ vs rs ###
    ############################

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    # mean = Measurements.value.(Fuu1s_neft)
    # stddev = Measurements.uncertainty.(Fuu1s_neft)
    # ax.plot(
    #     plotdata(rslist_big, mean, stddev)...;
    #     color=cdict["grey"],
    #     linestyle="--",
    #     label="\$F^{\\uparrow\\uparrow}_1 \\xi\$ (NEFT)",
    # )
    # ax.scatter(rslist_big, mean; color=cdict["grey"], s=4)
    mean = Measurements.value.(Fuus_neft)
    stddev = Measurements.uncertainty.(Fuus_neft)
    ax.plot(
        spline(rslist_big, mean, stddev)...;
        color=cdict["orange"],
        linestyle="--",
        label="\$F^{\\uparrow\\uparrow}_1\\xi + F^{\\uparrow\\uparrow}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["orange"], s=4)
    mean = Measurements.value.(Fuds_neft)
    stddev = Measurements.uncertainty.(Fuds_neft)
    ax.plot(
        spline(rslist_big, mean, stddev)...;
        color=cdict["magenta"],
        linestyle="--",
        label="\$F^{\\uparrow\\downarrow}_1\\xi + F^{\\uparrow\\downarrow}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["magenta"], s=4)
    # Our data
    ax.plot(
        spline(rslist_small, Fuu1s_ours)...;
        color=cdict["black"],
        label="\$F^{\\uparrow\\uparrow}_1 \\xi\$",
        zorder=-1,
    )
    ax.plot(
        spline(rslist_small, Fuus_ours)...;
        color=cdict["blue"],
        label="\$F^{\\uparrow\\uparrow}_1\\xi + F^{\\uparrow\\uparrow}_2 \\xi^2\$",
    )
    ax.plot(
        spline(rslist_small, Fuds_ours)...;
        color=cdict["cyan"],
        label="\$F^{\\uparrow\\downarrow}_1\\xi + F^{\\uparrow\\downarrow}_2 \\xi^2\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\sigma_1 \\sigma_2}\$")
    if isDynamic == false
        ax.set_ylim(-0.9, 0.9)
    end
    ax.set_xlim(0, 10)
    # ax.set_ylabel("\$F^s\$")
    ax.legend(; ncol=2, fontsize=12, loc="upper left", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_F_uu_and_ud$(interactionstr)_vs_rs_$(chan).pdf")
    plt.close(fig)

    #########################
    ### Fig 2: F2↑↑ vs rs ###
    #########################

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    mean = Measurements.value.(Fuu2ds_neft)
    stddev = Measurements.uncertainty.(Fuu2ds_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["orange"],
        linestyle="--",
        label="\$F^{\\uparrow\\uparrow,\\text{d}}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["orange"], s=4)
    # mean = Measurements.value.(Fuu2cts_neft)
    # stddev = Measurements.uncertainty.(Fuu2cts_neft)
    # ax.plot(
    #     plotdata(rslist_big, mean, stddev)...;
    #     color=cdict["red"],
    #     linestyle="--",
    #     label="\$F^{\\uparrow\\uparrow,\\text{ct}}_2 \\xi^2\$ (NEFT)",
    # )
    # ax.scatter(rslist_big, mean; color=cdict["red"], s=4)
    mean = Measurements.value.(Fuu2s_neft)
    stddev = Measurements.uncertainty.(Fuu2s_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["magenta"],
        linestyle="--",
        label="\$F^{\\uparrow\\uparrow}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["magenta"], s=4)
    # Ours
    ax.plot(
        spline(rslist_small, Fuu2cts_ours)...;
        color=cdict["teal"],
        label="\$F^{\\uparrow\\uparrow,\\text{ct}}_2 \\xi^2\$",
    )
    ax.plot(
        spline(rslist_small, Fuu2ds_ours)...;
        color=cdict["blue"],
        label="\$F^{\\uparrow\\uparrow,\\text{d}}_2 \\xi^2\$",
    )
    ax.plot(
        spline(rslist_small, Fuu2s_ours)...;
        color=cdict["cyan"],
        label="\$F^{\\uparrow\\uparrow}_2 \\xi^2\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\sigma_1 \\sigma_2}\$")
    if isDynamic == false
        ax.set_ylim(-0.45, 0.75)
    end
    # ax.set_ylabel("\$F^s\$")
    ax.set_xlim(0, 10)
    ax.legend(; ncol=2, fontsize=12, loc="upper left", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_F2_uu$(interactionstr)_vs_rs_$(chan).pdf")
    plt.close(fig)

    #########################
    ### Fig 3: F2↑↓ vs rs ###
    #########################

    # NOTE: CTs are identically zero for F2↑↓ for both ours/NEFT!
    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    mean = Measurements.value.(Fud2ds_neft)
    stddev = Measurements.uncertainty.(Fud2ds_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["orange"],
        linestyle="--",
        label="\$F^{\\uparrow\\downarrow,\\text{d}}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["orange"], s=4)
    # mean = Measurements.value.(Fud2cts_neft)
    # stddev = Measurements.uncertainty.(Fud2cts_neft)
    # ax.plot(
    #     plotdata(rslist_big, mean, stddev)...;
    #     color=cdict["red"],
    #     linestyle="--",
    #     label="\$F^{\\uparrow\\downarrow,\\text{ct}}_2 \\xi^2\$ (NEFT)",
    #     zorder=100,
    # )
    # ax.scatter(rslist_big, mean; color=cdict["red"], s=4)
    mean = Measurements.value.(Fud2s_neft)
    stddev = Measurements.uncertainty.(Fud2s_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["magenta"],
        linestyle="--",
        label="\$F^{\\uparrow\\downarrow}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["magenta"], s=4)
    # Ours
    ax.plot(
        spline(rslist_small, Fud2cts_ours)...;
        color=cdict["teal"],
        label="\$F^{\\uparrow\\downarrow,\\text{ct}}_2 \\xi^2\$",
    )
    ax.plot(
        spline(rslist_small, Fud2ds_ours)...;
        color=cdict["blue"],
        label="\$F^{\\uparrow\\downarrow,\\text{d}}_2 \\xi^2\$",
    )
    ax.plot(
        spline(rslist_small, Fud2s_ours)...;
        color=cdict["cyan"],
        label="\$F^{\\uparrow\\downarrow}_2 \\xi^2\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\sigma_1 \\sigma_2}\$")
    # if isDynamic == false
    #     ax.set_ylim(-0.9, 0.9)
    # end
    # ax.set_ylabel("\$F^s\$")
    ax.set_xlim(0, 10)
    ax.legend(; ncol=2, fontsize=12, loc="upper left", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_F2_ud$(interactionstr)_vs_rs_$(chan).pdf")
    plt.close(fig)

    ##########################
    ### Fig 4: Fs/Fa vs rs ###
    ##########################

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    # mean = Measurements.value.(Fs1s_neft)
    # stddev = Measurements.uncertainty.(Fs1s_neft)
    # ax.plot(
    #     plotdata(rslist_big, mean, stddev)...;
    #     color=cdict["grey"],
    #     linestyle="--",
    #     label="\$F^{s}_1 \\xi\$ (NEFT)",
    # )
    # ax.scatter(rslist_big, mean; color=cdict["grey"], s=4)
    mean = Measurements.value.(Fss_neft)
    stddev = Measurements.uncertainty.(Fss_neft)
    ax.plot(
        spline(rslist_big, mean, stddev)...;
        color=cdict["orange"],
        linestyle="--",
        label="\$F^{s}_1\\xi + F^{s}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["orange"], s=4)
    mean = Measurements.value.(Fas_neft)
    stddev = Measurements.uncertainty.(Fas_neft)
    ax.plot(
        spline(rslist_big, mean, stddev)...;
        color=cdict["magenta"],
        linestyle="--",
        label="\$F^{a}_1\\xi + F^{a}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["magenta"], s=4)
    # Our data
    ax.plot(
        spline(rslist_small, Fs1s_ours)...;
        color=cdict["black"],
        label="\$F^{s}_1 \\xi = F^{a}_1 \\xi\$",
        zorder=-1,
    )
    ax.plot(
        spline(rslist_small, Fss_ours)...;
        color=cdict["blue"],
        label="\$F^{s}_1\\xi + F^{s}_2 \\xi^2\$",
    )
    ax.plot(
        spline(rslist_small, Fas_ours)...;
        color=cdict["cyan"],
        label="\$F^{a}_1\\xi + F^{a}_2 \\xi^2\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\sigma_1 \\sigma_2}\$")
    if isDynamic == false
        ax.set_ylim(-0.62, 0.06)
    end
    ax.set_xlim(0, 10)
    # ax.set_ylabel("\$F^s\$")
    ax.legend(; ncol=2, fontsize=12, loc="upper right", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_F_s_and_a$(interactionstr)_vs_rs_$(chan).pdf")
    plt.close(fig)

    ########################
    ### Fig 5: F2s vs rs ###
    ########################

    # NOTE: CTs are identically zero for F2↑↓ for both ours/NEFT! ⟹ F2s and F2a CTs are identical
    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    mean = Measurements.value.(Fs2ds_neft)
    stddev = Measurements.uncertainty.(Fs2ds_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["orange"],
        linestyle="--",
        label="\$F^{s,\\text{d}}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["orange"], s=4)
    # mean = Measurements.value.(Fs2cts_neft)
    # stddev = Measurements.uncertainty.(Fs2cts_neft)
    # ax.plot(
    #     plotdata(rslist_big, mean, stddev)...;
    #     color=cdict["red"],
    #     linestyle="--",
    #     label="\$F^{s,\\text{ct}}_2 \\xi^2\$ (NEFT)",
    # )
    # ax.scatter(rslist_big, mean; color=cdict["red"], s=4)
    mean = Measurements.value.(Fs2s_neft)
    stddev = Measurements.uncertainty.(Fs2s_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["magenta"],
        linestyle="--",
        label="\$F^{s}_2 \\xi^2\$ (NEFT)",
    )
    ax.scatter(rslist_big, mean; color=cdict["magenta"], s=4)
    # Ours
    ax.plot(
        spline(rslist_small, Fs2cts_ours)...;
        color=cdict["teal"],
        label="\$F^{s,\\text{ct}}_2 \\xi^2\$",
    )
    ax.plot(
        spline(rslist_small, Fs2ds_ours)...;
        color=cdict["blue"],
        label="\$F^{s,\\text{d}}_2 \\xi^2\$",
    )
    ax.plot(
        spline(rslist_small, Fs2s_ours)...;
        color=cdict["cyan"],
        label="\$F^{s}_2 \\xi^2\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\sigma_1 \\sigma_2}\$")
    if isDynamic == false
        ax.set_ylim(-0.25, 0.6)
    end
    # ax.set_ylabel("\$F^s\$")
    ax.set_xlim(0, 10)
    ax.legend(; ncol=2, fontsize=12, loc="upper left", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_F2_s$(interactionstr)_vs_rs_$(chan).pdf")
    plt.close(fig)

    ########################
    ### Fig 6: F2a vs rs ###
    ########################

    # NOTE: CTs are identically zero for F2↑↓ for both ours/NEFT! ⟹ F2s and F2a CTs are identical
    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    mean = Measurements.value.(Fa2ds_neft)
    stddev = Measurements.uncertainty.(Fa2ds_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["orange"],
        linestyle="--",
        label="\$F^{a,\\text{d}}_2 \\xi^2\$ (NEFT)",
        zorder=100,
    )
    ax.scatter(rslist_big, mean; color=cdict["orange"], s=4)
    mean = Measurements.value.(Fa2s_neft)
    stddev = Measurements.uncertainty.(Fa2s_neft)
    ax.plot(
        plotdata(rslist_big, mean, stddev)...;
        color=cdict["magenta"],
        linestyle="--",
        label="\$F^{a}_2 \\xi^2\$ (NEFT)",
        zorder=100,
    )
    ax.scatter(rslist_big, mean; color=cdict["magenta"], s=4)
    # Ours
    ax.plot(
        spline(rslist_small, Fa2cts_ours)...;
        color=cdict["teal"],
        label="\$F^{a,\\text{ct}}_2 \\xi^2\$",
    )
    ax.plot(
        spline(rslist_small, Fa2ds_ours)...;
        color=cdict["blue"],
        label="\$F^{a,\\text{d}}_2 \\xi^2\$",
        zorder=100,
    )
    ax.plot(
        spline(rslist_small, Fs2s_ours)...;
        color=cdict["cyan"],
        label="\$F^{s}_2 \\xi^2\$",
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\sigma_1 \\sigma_2}\$")
    if isDynamic == false
        ax.set_ylim(-0.22, 0.22)
    end
    # ax.set_ylabel("\$F^s\$")
    ax.set_xlim(0, 10)
    ax.legend(; ncol=2, fontsize=12, loc="upper left", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_F2_a$(interactionstr)_vs_rs_$(chan).pdf")
    plt.close(fig)
end

function main()
    beta = 40.0
    # rslist = [0.01, 5, 10]

    # For fast Fa
    # rslist = [[0.01, 0.1, 0.5]; 1:2:10]

    # # Standard rslist

    save = true
    debug = true
    verbose = true
    z_renorm = false
    show_progress = true

    plot_neft = true
    plot_ours = true
    plot_integrands = false

    # Yukawa interaction
    isDynamic = false

    # ftype = "Fs"  # f^{Di} + f^{Ex} / 2
    # ftype = "Fa"  # f^{Ex} / 2

    # Plot NEFT benchmark data for F↑↑ and F↑↓ vs rs
    plot_F_uu_ud_NEFT(isDynamic, z_renorm)

    # # Plot extras
    # plot_extras(rslist, ftype)
    return
end

main()
