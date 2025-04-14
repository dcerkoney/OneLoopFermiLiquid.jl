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
    partitions=UEG.partition(1; offset=0),
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
function get_one_loop_graphs(
    rs,
    beta,
    mass2=0.0,
    Fs=0.0;
    partitions=UEG.partition(1; offset=0),
    leg_convention=:PH,
    channels=[PHr, PHEr, PPr],  # NOTE: Alli does not contribute at one-loop order
    remove_bubble=true,
    optimize_level=1,
)
    dummy_paramc = ElectronLiquid.ParaMC(;
        rs=rs,
        beta=beta,
        Fs=Fs,
        order=1,
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
function one_loop_mcmc_neft(
    para;
    kamp=[para.kF],
    kamp2=kamp,
    q=[0.0 for k in kamp],
    n=[-1, 0, 0, -1],
    l=[0],
    neval=1e6,
    filename::Union{String,Nothing}=nothing,
    reweight_goal=nothing,
    channels=[PHr, PHEr, PPr],  # NOTE: Alli does not contribute at one-loop order
    partitions=UEG.partition(para.order),
    leg_convention=:PH,
    remove_bubble=true,
    optimize_level=1,
    verbose=0,
    kwargs...,
)
    kF = para.kF

    diagram = get_one_loop_graphs(
        para.rs,
        para.beta,
        para.mass2;
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
        transferFrequencyIndices = [-1, 0, 0]
    end

    # diagrams = get_one_loop_graphs(0.01, beta)
    # graphs = diagrams[3]
    # isUpUp =
    #     Dict(P => [g.properties.response == UpUp for g in graphs[P]] for P in keys(graphs))
    # isUpDown = Dict(
    #     P => [g.properties.response == UpDown for g in graphs[P]] for P in keys(graphs)
    # )

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
        data, result = one_loop_mcmc_neft(
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
