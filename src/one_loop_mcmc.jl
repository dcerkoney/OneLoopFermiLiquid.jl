function get_vertex_graph()
    # Both left and right insertions (factor of 2)
    error("Not yet implemented!")
end

function get_direct_crossed_box_graph()
    error("Not yet implemented!")
end

function get_direct_uncrossed_box_graph()
    error("Not yet implemented!")
end

function get_exchange_crossed_graph()
    error("Not yet implemented!")
end

function get_exchange_uncrossed_graph()
    error("Not yet implemented!")
end

const one_loop_graph_funcs = Dict{OneLoopGraphType,Function}(
    vertex => get_vertex_graph,
    direct_crossed_box => get_direct_crossed_box_graph,
    direct_uncrossed_box => get_direct_uncrossed_box_graph,
    exchange_crossed_box => get_exchange_crossed_graph,
    exchange_uncrossed_box => get_exchange_uncrossed_graph,
)

function get_individual_one_loop_graphs(
    rs,
    beta,
    mass2=0.0,
    Fs=0.0;
    order=1,
    partitions=UEG.partition(order; offset=0),
    leg_convention=:PH,
    channels=[PHr, PHEr, PPr],
    remove_bubble=true,
    optimize_level=1,
)
    vertex_graph = get_vertex_graph()
    direct_crossed_box_graph = get_direct_crossed_box_graph()
    direct_uncrossed_box_graph = get_direct_uncrossed_box_graph()
    exchange_crossed_graph = get_exchange_crossed_graph()
    exchange_uncrossed_graph = get_exchange_uncrossed_graph()
    return (
        vertex_graph,
        direct_crossed_box_graph,
        direct_uncrossed_box_graph,
        exchange_crossed_graph,
        exchange_uncrossed_graph,
    )
end

"""
Load computational grpahs for one or all contributions to the
proper 4-point vertex Γ₄(σ1, σ2) from the three
irreducible channels (PHr, PHExr, PPr).
"""
function get_forward_scattering_graphs(
    rs,
    beta,
    mass2=0.0,
    Fs=0.0;
    order=1,
    partitions=UEG.partition(order; offset=0),
    leg_convention=:PH,
    channels=[PHr, PHEr, PPr],  # NOTE: Alli does not contribute at one-loop order
    remove_bubble=true,
    optimize_level=1,
)
    @assert all(N ≤ order for N in sum.(partitions)) "sum(P) ≤ N not satisfied for all partitions P!"
    dummy_paramc = ElectronLiquid.ParaMC(;
        rs=rs,
        beta=beta,
        Fs=Fs,
        order=order,
        mass2=mass2,
        isDynamic=false,
    )

    # Proper => only exchange and box-type direct diagrams contribute to F₂
    filters = remove_bubble ? [Proper, NoHartree, NoBubble] : [Proper, NoHartree]

    if leg_convention == :PP
        # (k1, k2, Q - k1, ...)
        idx_k1, idx_k2 = 1, 2
    else
        # (k1, k1 + Q, k2, ...)
        idx_k1, idx_k2 = 1, 3
    end

    # Setup forward-scattering extK and zero transfer momentum ((Q,Ω) → 0 limit)
    KinL, KoutL, KinR, KoutR = zeros(16), zeros(16), zeros(16), zeros(16)
    KinL[idx_k1] = KoutL[idx_k1] = 1  # k1  →  k1
    KinR[idx_k2] = KoutR[idx_k2] = 1  # k2 →  k2
    Q = KinL - KoutL        # Q = 0
    extK = (KinL, KoutL, KinR, KoutR)

    diagrams = Diagram.diagram_parquet_response(
        :vertex4,
        dummy_paramc,
        partitions;
        channels=channels,
        filter=filters,
        transferLoop=Q,  # (Q,Ω) → 0
        extK=extK,
        optimize_level=optimize_level,
    )

    # returns: (partition, diagpara, FeynGraphs, extT_labels, spin_conventions)
    # graphs contain: {{↑↑ diags}, {↑↓ diags}}
    return diagrams
end

"""
Use the NEFT toolbox to compute one or all contributions to the
one-loop Fermi liquid parameters F↑↑/F↑↓ from the three
irreducible channels (PHr, PHExr, PPr).
"""
function forward_scattering_mcmc_neft(
    para;
    kamp=[para.kF],
    kamp2=kamp,
    q=[0.0 for k in kamp],
    n=[0, 0, 0],
    l=[0],
    neval=1e6,
    filename::Union{String,Nothing}=nothing,
    reweight_goal=nothing,
    channels=[PHr, PHEr, PPr],  # NOTE: Alli does not contribute at one-loop order
    partitions=UEG.partition(para.order; offset=0),
    leg_convention=:PH,
    remove_bubble=true,
    optimize_level=1,
    verbose=0,
    kwargs...,
)
    @assert all(N ≤ para.order for N in sum.(partitions)) "sum(P) ≤ N not satisfied for all partitions P!"
    kF = para.kF

    diagram = get_forward_scattering_graphs(
        para.rs,
        para.beta,
        para.mass2;
        order=para.order,
        partitions=partitions,
        leg_convention=leg_convention,
        channels=channels,
        remove_bubble=remove_bubble,
        optimize_level=optimize_level,
    )

    # # TODO: code diagram-by-diagram graphs
    # individual_diagram = get_individual_one_loop_graphs()[idx_diagram]

    partition = diagram[1] # diagrams like (1, 1, 0) are absent, so the partitions must be updated
    println(partition)
    neighbor = UEG.neighbor(partition)

    if isnothing(reweight_goal)
        reweight_goal = Float64[]
        for (order, sOrder, vOrder) in partition
            # push!(reweight_goal, 8.0^(order + vOrder - 1))
            push!(reweight_goal, 8.0^(order - 1))
        end
        push!(reweight_goal, 1.0)
    end

    ver4, result = Ver4.lavg(
        para,
        diagram;
        kamp=kamp,
        kamp2=kamp2,
        q=q,
        n=n,
        l=l,
        neval=neval,
        print=verbose,
        neighbor=neighbor,
        reweight_goal=reweight_goal,
        chan=leg_convention,
        kwargs...,
    )

    if isnothing(ver4) == false
        # for (p, data) in ver4
        for p in partition
            data = ver4[p]
            printstyled("partition: $p\n"; color=:yellow)
            for (li, _l) in enumerate(l)
                printstyled("l = $_l\n"; color=:green)
                @printf(
                    "%12s    %16s    %16s    %16s    %16s    %16s    %16s\n",
                    "k/kF",
                    "uu",
                    "ud",
                    "di",
                    "ex",
                    "symmetric",
                    "asymmetric"
                )
                for (ki, k) in enumerate(kamp)
                    factor = 1.0
                    d1, d2 = real(data[1, li, ki]) * factor, real(data[2, li, ki]) * factor
                    s, a = (d1 + d2) / 2.0, (d1 - d2) / 2.0
                    di, ex = (s - a), (a) * 2.0
                    @printf(
                        "%12.6f    %16s    %16s    %16s    %16s    %16s    %16s\n",
                        k / kF,
                        "$d1",
                        "$d2",
                        "$di",
                        "$ex",
                        "$s",
                        "$a"
                    )
                end
            end
        end

        if isnothing(filename) == false
            jldopen(filename, "a+") do f
                key = "$(UEG.short(para))"
                if haskey(f, key)
                    @warn("replacing existing data for $key")
                    delete!(f, key)
                end
                f[key] = (kamp, n, l, ver4)
            end
        end
    end
    return ver4, result
end

"""
Benchmark for the one-loop calculation of F for the Yukawa-type theory using the NEFT toolbox.
To this end, we use the exactly-computed counterterms δ_R of the full theory with the replacement R = VTF,
rather than the NEFT default δ_λ for static interactions.
"""
function get_yukawa_one_loop_neft(
    rslist,
    beta;
    neval=1e6,
    seed=1234,
    leg_convention=:PH,
    channels=[PHr, PHEr, PPr],  # NOTE: Alli does not contribute at one-loop order
    z_renorm=false,
    noTransferMomentum=true,
)
    # We calculate the counterterms and tree-level diagram exactly
    partitions = [(1, 0, 0)]

    # TODO: Determine which of these is correct (matters for dynamic interactions!)
    #       with -1, we have a transfer frequency of iν₁ = i2πT, and with all zeros,
    #       we have iν₀ = 0.
    if noTransferMomentum
        transferFrequencyIndices = [0, 0, 0]
    else
        error("Nonzero transfer momentum depends on Ex/Di—not yet implemented!")
        # transferFrequencyIndices = [-1, 0, 0] for Ex, [-1, -1, 0] for Di
    end

    oneloop_neft_sa = []
    oneloop_neft_ud = []
    for rs in rslist
        println("\nrs = $rs:")
        qTF = Parameter.rydbergUnit(1.0 / beta, rs, 3).qTF
        paramc = ElectronLiquid.ParaMC(;
            rs=rs,
            beta=beta,
            Fs=0.0,
            order=1,
            mass2=qTF^2,  # for a particularly simple dimensionless interaction
            isDynamic=false,
        )
        @unpack e0, NF, mass2, basic = paramc
        # data, result = Ver4.MC_lavg(
        data, result = forward_scattering_mcmc_neft(
            paramc;
            neval=neval,
            n=transferFrequencyIndices,
            channels=channels,
            partitions=partitions,
            leg_convention=leg_convention,
            seed=seed,
            print=-1,
        )
        if isnothing(data) == false
            # one-loop diagrams from NEFT toolbox
            F2d = Tuple(real(data[partitions[1]]))
            # tree-level F1↑↑ and F1↑↓ calculated exactly
            Fs1 = Fa1 = get_F1_TF(rs)
            F1 = measurement.((Ver4.sa2ud(Fs1, Fa1)))  # = (2 * get_F1_TF(rs), 0.0)
            # one-loop R-type counterterms calculated exactly
            Fs2ct, Fa2ct = real.(one_loop_counterterms(paramc))
            F2ct = measurement.((Ver4.sa2ud(Fs2ct, Fa2ct)))
            # z²⟨Γ⟩ = (1 + z₁ξ + ...)²⟨Γ⟩ = 2z₁F₁ξ²
            if z_renorm
                z1 = get_Z1(param)
                Fs2z = Fa2z = 2 * z1 * F1
                F2z = measurement.((Fs2z, Fa2z))
            else
                F2z = measurement.((0.0, 0.0))
            end
            Fuu2 = F2d[1] + F2ct[1] + F2z[1]
            Fud2 = F2d[2] + F2ct[2] + F2z[2]
            F2 = (Fuu2, Fud2)
            Fuu = F1[1] + F2[1]
            Fud = F1[2] + F2[2]
            F = (Fuu, Fud)
            no_meas = measurement.((NaN, NaN))
            oneloop_ud = OneLoopResult(
                F1,
                no_meas,
                no_meas,
                # no_meas,
                F2d,
                F2ct,
                # no_meas,
                F2z,
                F2,
                F,
            )
            oneloop_sa = ud2sa(oneloop_ud)
            push!(oneloop_neft_sa, oneloop_sa)
            push!(oneloop_neft_ud, oneloop_ud)
            GC.gc()
        end
    end
    return oneloop_neft_sa, oneloop_neft_ud
end

"""
Benchmark for the one-loop calculation of F for the R-type theory using the NEFT toolbox.
To this end, we use the exactly-computed counterterms δ_R of the full theory with the replacement R = VTF,
rather than the NEFT default δ_λ for static interactions.
"""
function get_rpa_one_loop_neft(
    rslist,
    beta;
    neval=1e6,
    seed=1234,
    leg_convention=:PH,
    channels=[PHr, PHEr, PPr],  # NOTE: Alli does not contribute at one-loop order
    z_renorm=false,
    noTransferMomentum=true,
)
    # We calculate the counterterms and tree-level diagram exactly
    partitions = [(1, 0, 0)]

    # TODO: Determine which of these is correct (matters for dynamic interactions!)
    #       with -1, we have a transfer frequency of iν₁ = i2πT, and with all zeros,
    #       we have iν₀ = 0.
    if noTransferMomentum
        transferFrequencyIndices = [0, 0, 0]
    else
        error("Nonzero transfer momentum depends on Ex/Di—not yet implemented!")
        # transferFrequencyIndices = [-1, 0, 0] for Ex, [-1, -1, 0] for Di
    end

    oneloop_neft_sa = []
    oneloop_neft_ud = []
    for rs in rslist
        println("\nrs = $rs:")
        qTF = Parameter.rydbergUnit(1.0 / beta, rs, 3).qTF
        paramc = ElectronLiquid.ParaMC(;
            rs=rs,
            beta=beta,
            Fs=0.0,       # RPA
            order=1,
            mass2=0.001,  # in case of divergences
            isDynamic=true,
        )
        @unpack e0, NF, mass2, basic = paramc
        # data, result = Ver4.MC_lavg(
        data, result = forward_scattering_mcmc_neft(
            paramc;
            neval=neval,
            n=transferFrequencyIndices,
            channels=channels,
            partitions=partitions,
            leg_convention=leg_convention,
            seed=seed,
            print=-1,
        )
        if isnothing(data) == false
            # one-loop diagrams from NEFT toolbox
            F2d = Tuple(real(data[partitions[1]]))
            # tree-level F1↑↑ and F1↑↓ calculated exactly
            Fs1 = Fa1 = get_F1_TF(rs)
            F1 = measurement.((Ver4.sa2ud(Fs1, Fa1)))  # = (2 * get_F1_TF(rs), 0.0)
            # one-loop R-type counterterms calculated exactly
            Fs2ct, Fa2ct = real.(one_loop_counterterms(paramc))
            F2ct = measurement.((Ver4.sa2ud(Fs2ct, Fa2ct)))
            # z²⟨Γ⟩ = (1 + z₁ξ + ...)²⟨Γ⟩ = 2z₁F₁ξ²
            if z_renorm
                z1 = get_Z1(param)
                Fs2z = Fa2z = 2 * z1 * F1
                F2z = measurement.((Fs2z, Fa2z))
            else
                F2z = measurement.((0.0, 0.0))
            end
            Fuu2 = F2d[1] + F2ct[1] + F2z[1]
            Fud2 = F2d[2] + F2ct[2] + F2z[2]
            F2 = (Fuu2, Fud2)
            Fuu = F1[1] + F2[1]
            Fud = F1[2] + F2[2]
            F = (Fuu, Fud)
            no_meas = measurement.((NaN, NaN))
            oneloop_ud = OneLoopResult(
                F1,
                no_meas,
                no_meas,
                # no_meas,
                F2d,
                F2ct,
                # no_meas,
                F2z,
                F2,
                F,
            )
            oneloop_sa = ud2sa(oneloop_ud)
            push!(oneloop_neft_sa, oneloop_sa)
            push!(oneloop_neft_ud, oneloop_ud)
            GC.gc()
        end
    end
    return oneloop_neft_sa, oneloop_neft_ud
end

"""
Benchmark for the tree-level calculation of F for the Yukawa-type theory using the NEFT toolbox.
"""
function get_yukawa_tree_level_neft(
    rslist,
    beta;
    neval=1e6,
    seed=1234,
    leg_convention=:PH,
    channels=[PHr, PHEr, PPr],  # NOTE: Alli does not contribute up to one-loop order
    noTransferMomentum=true,
)
    partitions = [(0, 0, 0)]

    # TODO: Determine which of these is correct (matters for dynamic interactions!)
    #       with -1, we have a transfer frequency of iν₁ = i2πT, and with all zeros,
    #       we have iν₀ = 0.
    if noTransferMomentum
        transferFrequencyIndices = [0, 0, 0]
    else
        error("Nonzero transfer momentum depends on Ex/Di—not yet implemented!")
        # transferFrequencyIndices = [-1, 0, 0] for Ex, [-1, -1, 0] for Di
    end

    treelevel_neft_sa = []
    treelevel_neft_ud = []
    for rs in rslist
        println("\nrs = $rs:")
        qTF = Parameter.rydbergUnit(1.0 / beta, rs, 3).qTF
        paramc = ElectronLiquid.ParaMC(;
            rs=rs,
            beta=beta,
            Fs=0.0,
            order=1,
            mass2=qTF^2,  # for a particularly simple dimensionless interaction
            isDynamic=false,
        )
        @unpack e0, NF, mass2, basic = paramc
        # data, result = Ver4.MC_lavg(
        data, result = forward_scattering_mcmc_neft(
            paramc;
            neval=neval,
            n=transferFrequencyIndices,
            channels=channels,
            partitions=partitions,
            leg_convention=leg_convention,
            seed=seed,
            print=-1,
        )
        if isnothing(data) == false
            # # tree-level F1↑↑ and F1↑↓ calculated exactly
            # Fs1 = Fa1 = get_F1_TF(rs)
            # F1 = measurement.((Ver4.sa2ud(Fs1, Fa1)))  # = (2 * get_F1_TF(rs), 0.0)
            
            # tree-level diagram from NEFT toolbox
            F1_ud = Tuple(real(data[partitions[1]]))
            F1_sa = Ver4.ud2sa(F1_ud...)
            push!(treelevel_neft_sa, F1_sa)
            push!(treelevel_neft_ud, F1_ud)
            GC.gc()
        end
    end
    # NOTE: there is an extra factor of (-1)ᵐ = -1 due
    #       to the difference of Feynman rules V ↦ -V
    return treelevel_neft_sa, treelevel_neft_ud
end
