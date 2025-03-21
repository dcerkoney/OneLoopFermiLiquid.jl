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

function plot_F_uu_ud_NEFT(isDynamic, z_renorm)
    # ftypestr = ftype == "Fs" ? "F^{s}" : "F^{a}"

    interactionstr = isDynamic ? "" : "_yukawa"
    zstr = z_renorm ? "_z_renorm" : ""

    # nk ≈ na ≈ 100 is sufficiently converged for all relevant euv/rtol
    Nk, Ok = 7, 6
    Na, Oa = 8, 7

    # DLR parameters for which r(q, 0) is smooth in the q → 0 limit (tested for rs = 1, 10)
    euv = 10.0
    rtol = 1e-7

    # Load NEFT benchmark data using jld2
    @load "one_loop_F_neft.jld2" rslist FsDMCs FaDMCs F1NEFTs Fs2NEFTs Fa2NEFTs FstotalNEFTs FatotalNEFTs Fuu2NEFTs Fud2NEFTs FuutotalNEFTs FudtotalNEFTs F1s_exact F2cts_exact Fuu1s Fud1s Fs2s Fa2s Fuu2vpbs Fud2vpbs Fuu2cts Fud2cts Fuu2s Fud2s
    rslist_big = rslist

    Fuu2totalNEFTs = Fuu1s .+ Fuu2NEFTs
    Fud2totalNEFTs = Fud1s .+ Fud2NEFTs

    # # Load our data using jld2
    # @load "one_loop_Fs_ours.jld2" rslist F1s F2vs F2bs F2cts F2zs F2s
    # rslist_small = rslist
    # Fs2vs = F2vs
    # Fs2bs = F2bs
    # Fs2cts = F2cts
    # Fs2zs = F2zs
    # Fs2s = F2s
    # @load "one_loop_Fa_ours.jld2" rslist F1s F2vs F2bs F2cts F2zs F2s
    # @assert rslist == rslist_small
    # Fa2vs = F2vs
    # Fa2bs = F2bs
    # Fa2cts = F2cts
    # Fa2zs = F2zs
    # Fa2s = F2s

    # # Fs = (F↑↑ + F↑↓) / 2, Fa = (F↑↑ - F↑↓) / 2,
    # # so F↑↑ = Fs + Fa, F↑↓ = Fs - Fa
    # Fuu2s = Fs2s .+ Fa2s
    # Fud2s = Fs2s .- Fa2s

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    ax.errorbar(
        rslist,
        Measurements.value.(F1NEFTs),
        Measurements.uncertainty.(F1NEFTs);
        label="\$F_1 \\xi\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["black"],
    )
    ax.errorbar(
        rslist,
        Measurements.value.(Fuu2NEFTs),
        Measurements.uncertainty.(Fuu2NEFTs);
        label="\$F^{\\uparrow\\uparrow}_2 \\xi^2\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["blue"],
    )
    ax.errorbar(
        rslist,
        Measurements.value.(Fud2NEFTs),
        Measurements.uncertainty.(Fud2NEFTs);
        label="\$F^{\\uparrow\\downarrow}_2 \\xi^2\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["cyan"],
    )
    ax.errorbar(
        rslist,
        Measurements.value.(FuutotalNEFTs),
        Measurements.uncertainty.(FuutotalNEFTs);
        label="\$F_1\\xi + F^{\\uparrow\\uparrow}_2 \\xi^2\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["orange"],
    )
    ax.errorbar(
        rslist,
        Measurements.value.(FudtotalNEFTs),
        Measurements.uncertainty.(FudtotalNEFTs);
        label="\$F_1\\xi + F^{\\uparrow\\downarrow}_2 \\xi^2\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["magenta"],
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\sigma_1 \\sigma_2}\$")
    if isDynamic == false
        ax.set_ylim(-0.05, 1.45)
    end
    # ax.set_ylabel("\$F^s\$")
    ax.legend(; ncol=2, fontsize=12, loc="best", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_F_uu_and_ud_yukawa_vs_rs$(interactionstr)_neft.pdf")
    plt.close(fig)
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
            lindhard.(xgrid) .* (xgrid * rstilde) ./ (xgrid .^ 2 .+ rstilde),
            xgrid,
        )
        B = Interp.integrate1D(lindhard.(xgrid) .* xgrid, xgrid)
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

function main()
    beta = 40.0
    # rslist = [1, 10]

    # For fast Fa
    rslist = [[0.01, 0.1, 0.5]; 1:2:10]

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
    ftype = "Fa"  # f^{Ex} / 2

    # Plot NEFT benchmark data for F↑↑ and F↑↓ vs rs
    plot_F_uu_ud_NEFT(isDynamic, z_renorm)

    # # Plot extras
    # plot_extras(rslist, ftype)
    return
end

main()
