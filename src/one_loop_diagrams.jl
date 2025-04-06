"""
Calculates and returns the specified one-loop box diagram contribution (Fs2bi, Fa2bi).
The box diagram can be crossed or uncrossed, direct or exchange—all four combinations enter F2.
"""
function one_loop_box_diagram(
    param::OneLoopParams;
    is_direct::Bool,
    is_crossed::Bool,
    show_progress=false,
)
    @assert param.initialized "r(q, iνₘ) data not yet initialized!"
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    @unpack qgrid, θgrid, φgrid, θ12, rs, Fs, kF, NF, basic, paramc = param

    # Initialize vertex integrand
    Nq = length(qgrid.grid)
    q_integrand_s = zeros(ComplexF64, Nq)

    # Setup buffers for scatter/gather
    counts = split_count(Nq, comm_size)  # number of values per rank
    data_vbuffer_s = VBuffer(q_integrand_s, counts)
    if rank == root
        length_ubuf = UBuffer(counts, 1)
        # For global indices
        qi_vbuffer = VBuffer(collect(1:Nq), counts)
    else
        length_ubuf = UBuffer(nothing)
        qi_vbuffer = VBuffer(nothing)
    end

    # Scatter the data to all ranks
    local_length = MPI.Scatter(length_ubuf, Int, root, comm)
    local_qi = MPI.Scatterv!(qi_vbuffer, zeros(Int, local_length), root, comm)
    local_data_s =
        MPI.Scatterv!(data_vbuffer_s, zeros(ComplexF64, local_length), root, comm)

    # Compute the integrand over loop momentum magnitude q in parallel
    progress_meter = Progress(
        local_length;
        desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress && rank == root,
    )
    θs_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    φs_integrand = Vector{ComplexF64}(undef, length(φgrid.grid))
    for (i, qi) in enumerate(local_qi)
        # println("rank = $rank: Integrating (q, 0) point $i/$local_length")
        # Get external frequency and momentum at this index
        q = qgrid.grid[qi]
        for (iθ, θ) in enumerate(θgrid)
            for (iφ, φ) in enumerate(φgrid)
                φs_integrand[iφ] = box_matsubara_sum(
                    param,
                    q,
                    θ,
                    φ;
                    is_direct=is_direct,
                    is_crossed=is_crossed,
                )
            end
            θs_integrand[iθ] = Interp.integrate1D(φs_integrand, φgrid)
        end
        local_data_s[i] = Interp.integrate1D(θs_integrand .* sin.(θgrid.grid), θgrid)
        next!(progress_meter)
    end
    finish!(progress_meter)

    # Collect q_integrand subresults from all ranks
    MPI.Allgatherv!(local_data_s, data_vbuffer_s, comm)

    # total integrand ~ NF
    box_integrand_s = q_integrand_s / (NF * (2π)^3)

    # Integrate over q
    Fs2b = Interp.integrate1D(box_integrand_s, qgrid)

    # NOTE: The spin-antisymmetric contribution vanishes for direct-type diagrams,
    # and is identitcal to the spin-symmetric contribution for exchange-type diagrams.
    is_exchange = !is_direct
    Fa2b = is_exchange * Fs2b

    return Fs2b, Fa2b
end

"""
Calculates and returns the contribution from all one-loop box diagrams (Fs2b, Fa2b):

    gg'RR' + exchange counterparts
"""
function one_loop_box_contribution(param::OneLoopParams; show_progress=false)
    @assert param.initialized "r(q, iνₘ) data not yet initialized!"
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    @unpack qgrid, θgrid, φgrid, θ12, rs, Fs, kF, NF, basic, paramc = param

    # Initialize vertex integrand
    Nq = length(qgrid.grid)
    q_integrand_s = zeros(ComplexF64, Nq)
    q_integrand_a = zeros(ComplexF64, Nq)

    # Setup buffers for scatter/gather
    counts = split_count(Nq, comm_size)  # number of values per rank
    data_vbuffer_s = VBuffer(q_integrand_s, counts)
    data_vbuffer_a = VBuffer(q_integrand_a, counts)
    if rank == root
        length_ubuf = UBuffer(counts, 1)
        # For global indices
        qi_vbuffer = VBuffer(collect(1:Nq), counts)
    else
        length_ubuf = UBuffer(nothing)
        qi_vbuffer = VBuffer(nothing)
    end

    # Scatter the data to all ranks
    local_length = MPI.Scatter(length_ubuf, Int, root, comm)
    local_qi = MPI.Scatterv!(qi_vbuffer, zeros(Int, local_length), root, comm)
    local_data_s =
        MPI.Scatterv!(data_vbuffer_s, zeros(ComplexF64, local_length), root, comm)
    local_data_a =
        MPI.Scatterv!(data_vbuffer_a, zeros(ComplexF64, local_length), root, comm)

    # Compute the integrand over loop momentum magnitude q in parallel
    progress_meter = Progress(
        local_length;
        desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress && rank == root,
    )
    θs_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    θa_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    φs_integrand = Vector{ComplexF64}(undef, length(φgrid.grid))
    φa_integrand = Vector{ComplexF64}(undef, length(φgrid.grid))
    for (i, qi) in enumerate(local_qi)
        # println("rank = $rank: Integrating (q, 0) point $i/$local_length")
        # Get external frequency and momentum at this index
        q = qgrid.grid[qi]
        for (iθ, θ) in enumerate(θgrid)
            for (iφ, φ) in enumerate(φgrid)
                φs_integrand[iφ] = box_matsubara_sum(param, q, θ, φ, "Fs")
                φa_integrand[iφ] = box_matsubara_sum(param, q, θ, φ, "Fa")
            end
            θs_integrand[iθ] = Interp.integrate1D(φs_integrand, φgrid)
            θa_integrand[iθ] = Interp.integrate1D(φa_integrand, φgrid)
        end
        local_data_s[i] = Interp.integrate1D(θs_integrand .* sin.(θgrid.grid), θgrid)
        local_data_a[i] = Interp.integrate1D(θa_integrand .* sin.(θgrid.grid), θgrid)
        next!(progress_meter)
    end
    finish!(progress_meter)

    # Collect q_integrand subresults from all ranks
    MPI.Allgatherv!(local_data_s, data_vbuffer_s, comm)
    MPI.Allgatherv!(local_data_a, data_vbuffer_a, comm)

    # total integrand ~ NF
    box_integrand_s = q_integrand_s / (NF * (2π)^3)
    box_integrand_a = q_integrand_a / (NF * (2π)^3)

    # Integrate over q
    Fs2b = Interp.integrate1D(box_integrand_s, qgrid)
    Fa2b = Interp.integrate1D(box_integrand_a, qgrid)
    return Fs2b, Fa2b
end

"""
Calculates and returns the contribution from (identital left/right) one-loop vertex corrections (Fs2v, Fa2v):

    2 Λ₁(θ₁₂) r(|k₁ - k₂|, 0)
"""
function one_loop_vertex_contribution(param::OneLoopParams; show_progress=false)
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    @unpack qgrid, θgrid, φgrid, θ12, rs, Fs, kF, NF, basic, paramc = param

    # Initialize vertex integrand
    Nq = length(qgrid.grid)
    q_integrand = zeros(ComplexF64, Nq)

    # Setup buffers for scatter/gather
    counts = split_count(Nq, comm_size)  # number of values per rank
    data_vbuffer = VBuffer(q_integrand, counts)
    if rank == root
        length_ubuf = UBuffer(counts, 1)
        # For global indices
        qi_vbuffer = VBuffer(collect(1:Nq), counts)
    else
        length_ubuf = UBuffer(nothing)
        qi_vbuffer = VBuffer(nothing)
    end

    # Scatter the data to all ranks
    local_length = MPI.Scatter(length_ubuf, Int, root, comm)
    local_qi = MPI.Scatterv!(qi_vbuffer, zeros(Int, local_length), root, comm)
    local_data = MPI.Scatterv!(data_vbuffer, zeros(ComplexF64, local_length), root, comm)

    # Compute the integrand over loop momentum magnitude q in parallel
    progress_meter = Progress(
        local_length;
        desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress && rank == root,
    )
    θ_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    φ_integrand = Vector{ComplexF64}(undef, length(φgrid.grid))
    for (i, qi) in enumerate(local_qi)
        # println("rank = $rank: Integrating (q, 0) point $i/$local_length")
        # Get external frequency and momentum at this index
        q = qgrid.grid[qi]
        for (iθ, θ) in enumerate(θgrid)
            for (iφ, φ) in enumerate(φgrid)
                φ_integrand[iφ] = vertex_matsubara_sum(param, q, θ, φ)
            end
            θ_integrand[iθ] = Interp.integrate1D(φ_integrand, φgrid)
        end
        local_data[i] = Interp.integrate1D(θ_integrand .* sin.(θgrid.grid), θgrid)
        next!(progress_meter)
    end
    finish!(progress_meter)

    # Collect q_integrand subresults from all ranks
    MPI.Allgatherv!(local_data, data_vbuffer, comm)

    # total integrand ~ NF
    vertex_integrand = q_integrand / (NF * (2π)^3)

    # Integrate over q
    k_m_kp = kF * sqrt(2 * (1 - cos(θ12)))
    # Fᵥ(θ₁₂) = Λ₁(θ₁₂) r(|k₁ - k₂|, 0)
    # NOTE: only exchange-type vertex corrections enter ⟹ Fs2v = Fa2v
    Fs2v = Fa2v = Interp.integrate1D(vertex_integrand, qgrid) * r_interp(param, k_m_kp, 0)
    return Fs2v, Fa2v
end

"""
Separately calculates and returns all four one-loop box-type contributions to Fs2 and Fa2.
The box diagrams can be crossed or uncrossed, direct or exchange—all four combinations enter F2.
"""
function get_one_loop_box_contributions(
    param::OneLoopParams;
    verbose=false,
    show_progress=true,
)
    F2b_terms = []
    for boxtype in boxtypes
        is_direct, is_crossed = boxtype
        F2b_term =
            real.(
                one_loop_box_diagram(
                    param;
                    is_direct=is_direct,
                    is_crossed=is_crossed,
                    show_progress=show_progress,
                )
            )
        push!(F2b_terms, F2b_term)
    end
    boxdiags_sa = OneLoopBoxDiagrams(F2b_terms...)
    boxdiags_ud = sa2ud(boxdiags_sa)
    verbose && println_root("(s, a):\n$boxdiags_sa")
    verbose && println_root("(↑↑, ↑↓):\n$boxdiags_ud")
    return boxdiags_sa, boxdiags_ud
end

"""
Calculates and returns all tree-level (F₁) and one-loop (F₂) contributions to Fs2 and Fa2.
"""
function get_one_loop_Fs(
    param::OneLoopParams;
    verbose=false,
    z_renorm=false,
    show_progress=true,
)
    Fs1 = Fa1 = get_F1(param)
    F1 = (Fs1, Fa1)

    F2v = real.(one_loop_vertex_contribution(param))
    F2b = real.(one_loop_box_contribution(param))
    # F2bubble = real.(one_loop_bubble_diagram(param))

    F2ct = real.(one_loop_counterterms(param))
    # F2bubblect = real.(one_loop_bubble_counterterm(param))

    # z²⟨Γ⟩ = (1 + z₁ξ + ...)²⟨Γ⟩ = 2z₁F₁ξ²
    if z_renorm
        z1 = get_Z1(param)
        Fs2z = Fa2z = 2 * z1 * Fs1
    else
        Fs2z = Fa2z = 0.0
    end
    F2z = (Fs2z, Fa2z)

    # Sum of all diagrams without counterterms
    Fs2d = F2v[1] + F2b[1]
    Fa2d = F2v[2] + F2b[2]
    F2d = (Fs2d, Fa2d)

    # Total contribution for F2 (O(ξ²) term)
    Fs2 = F2v[1] + F2b[1] + F2ct[1] + F2z[1] # + F2bct[1]
    Fa2 = F2v[2] + F2b[2] + F2ct[2] + F2z[2] # + F2bct[2]
    F2 = (Fs2, Fa2)

    # Total result for F to O(ξ²)
    Fs = F1[1] + F2[1]
    Fa = F1[2] + F2[2]
    F = (Fs, Fa)

    oneloop_sa = OneLoopResult(
        F1,
        F2v,
        F2b,
        # F2bubble,
        F2d,
        F2ct,
        # F2bubblect,
        F2z,
        F2,
        F,
    )
    oneloop_ud = sa2ud(oneloop_sa)
    verbose && println_root("(s, a):\n$oneloop_sa")
    verbose && println_root("(↑↑, ↑↓):\n$oneloop_ud")
    return oneloop_sa, oneloop_ud
end

function get_one_loop_graphs(rs, beta, mass2=0.0, Fs=0.0, chan=:PH, remove_bubble=true)
    # One-loop => only interaction-type counterterms
    partitions = UEG.partition(1; offset=0)  # TEST: G-type counterterms should be removed by Diagram.diagram_parquet_response!

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

    if chan == :PP
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
        filter=filters,
        transferLoop=Q,  # (Q,Ω) → 0
        extK=extK,
    )

    # returns: (partition, diagpara, FeynGraphs, extT_labels, spin_conventions)
    # graphs contain: {{↑↑ diags}, {↑↓ diags}}
    return diagrams
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
    chan=:PH,
    z_renorm=false,
    noTransferMomentum=true,
)
    # We calculate the counterterms and tree-level diagram exactly
    partitions = [(1, 0, 0)]

    # Proper => only exchange and box-type direct diagrams contribute to F₂
    # Bubble diagram cancelled exactly in CTs => NoBubble
    filters = [Proper, NoHartree, NoBubble]

    if chan == :PP
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
    transferLoop = KinL - KoutL       # Q = 0
    extK = (KinL, KoutL, KinR, KoutR)

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
        data, result = Ver4.MC_lavg(
            paramc;
            neval=neval,
            n=transferFrequencyIndices,
            l=[0],
            filter=filters,
            partition=partitions,
            transferLoop=transferLoop,
            extK=extK,
            seed=seed,
            print=-1,
            chan=chan,
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
