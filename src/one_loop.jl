

"""
Calculates and returns the one-loop vertex corrections:

    2 Λ₁(θ₁₂) r(|k₁ - k₂|, 0)
"""
function one_loop_vertex_corrections(param::OneLoopParams; show_progress=false)
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
    result = Interp.integrate1D(vertex_integrand, qgrid) * r_interp(param, k_m_kp, 0)
    return result
end

"""
Calculates and returns all one-loop box diagrams:

    gg'RR' + exchange counterparts
"""
function one_loop_box_diagrams(param::OneLoopParams; show_progress=false, ftype="fs")
    @assert param.initialized "r(q, iνₘ) data not yet initialized!"
    @assert ftype in ["Fs", "Fa"]
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
                φ_integrand[iφ] = box_matsubara_sum(param, q, θ, φ; ftype=ftype)
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
    box_integrand = q_integrand / (NF * (2π)^3)

    # Integrate over q
    result = Interp.integrate1D(box_integrand, qgrid)
    return result
end

"""
Calculates and returns all one-loop box diagrams of direct type:

    gg'RR'
"""
function one_loop_direct_box_diagrams(
    param::OneLoopParams;
    show_progress=false,
    which="both",
)
    @assert param.initialized "r(q, iνₘ) data not yet initialized!"
    @assert which in ["both", "ladder", "crossed"]
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
                φ_integrand[iφ] = direct_box_matsubara_sum(param, q, θ, φ; which=which)
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
    box_integrand = q_integrand / (NF * (2π)^3)

    # Integrate over q
    result = Interp.integrate1D(box_integrand, qgrid)
    return result
end

"""
Calculates and returns the total counterterm contribution to F2:

    2R(z1 - f1 Π0) - f1 Π0 f1
"""
function one_loop_counterterms(param::OneLoopParams; kwargs...)
    @unpack rs, kF, EF, NF, Fs, basic, isDynamic = param
    if isDynamic == false
        @assert param.mass2 ≈ param.qTF^2 "Counterterms currently only implemented for the Thomas-Fermi interaction (Yukawa mass = qTF)!"
    end
    rstilde = rs * alpha_ueg / π

    # x = |k - k'| / 2kF
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 16, 1e-8, 16)

    # NF * ⟨R⟩ = -x N_F R(2kF x, 0)
    F1 = get_F1(param)

    # x R(2kF x, 0)
    x_NF_R = isDynamic ? x_NF_R0 : x_NF_VTF
    x_R0 = [x_NF_R(x, rstilde, Fs) / NF for x in xgrid]

    # Z_1(kF)
    z1 = get_Z1(param, 2 * kF * xgrid)

    # Π₀(q, iν=0) = -NF * 𝓁(q / 2kF)
    Π0 = -NF * lindhard.(xgrid)

    # BUGGY!
    # # A = z₁ + 2 ∫₀¹ dx x R(x, 0) Π₀(x, 0)
    # A = z1 + Interp.integrate1D(2 * x_R0 .* Π0, xgrid)

    # A = z₁ + ∫₀¹ dx x R(x, 0) Π₀(x, 0)
    A = z1 + Interp.integrate1D(x_R0 .* Π0, xgrid)

    # B = ∫₀¹ dx x Π₀(x, 0) / NF
    B = Interp.integrate1D(xgrid .* Π0 / NF, xgrid)

    # (NF/2)*⟨2R(z1 - f1 Π0) - f1 Π0 f1⟩ = -2 F1 A - F1² B
    vertex_cts = -(2 * F1 * A + F1^2 * B)
    # vertex_cts = 2 * F1 * A + F1^2 * B
    return vertex_cts
end

"""
Calculates and returns all tree-level (F₁) and one-loop (F₂) contributions to F.
"""
function get_one_loop_Fs(
    param::OneLoopParams;
    verbose=false,
    ftype="Fs",
    z_renorm=false,
    kwargs...,
)
    function one_loop_total(param, verbose; kwargs...)
        if verbose
            F1 = param.isDynamic ? get_F1(param) : get_F1_TF(param.rs)
            println_root("F1 = ($(F1))ξ")

            F2v = real(one_loop_vertex_corrections(param; kwargs...))
            println_root("F2v = ($(F2v))ξ²")

            F2b = real(one_loop_box_diagrams(param; ftype=ftype, kwargs...))
            println_root("F2b = ($(F2b))ξ²")

            F2ct = real(one_loop_counterterms(param; kwargs...))
            println_root("F2ct = ($(F2ct))ξ²")

            # z²⟨Γ⟩ = (1 + z₁ξ + ...)²⟨Γ⟩ = 2z₁F₁ξ²
            if z_renorm
                z1 = get_Z1(param)
                F2z = 2 * z1 * F1
                println_root("F2z = ($(F2z))ξ²")
            else
                F2z = 0.0
            end

            F2 = F2v + F2b + F2ct + F2z
            println_root("F2 = ($(F1))ξ + ($(F2))ξ²")
            return F1, F2v, F2b, F2ct, F2z, F2
        else
            F1 = get_F1(param)
            F2v = real(one_loop_vertex_corrections(param; kwargs...))
            F2b = real(one_loop_box_diagrams(param; kwargs...))
            F2ct = real(one_loop_counterterms(param; kwargs...))
            if z_renorm
                z1 = get_Z1(param)
                F2z = 2 * z1 * F1
            else
                F2z = 0.0
            end
            F2 = F2v + F2b + F2ct + F2z
            return F1, F2v, F2b, F2ct, F2z, F2
        end
    end
    return one_loop_total(param, verbose; kwargs...)
end

"""
Benchmark for the one-loop calculation of F for the Yukawa-type theory using the NEFT toolbox.
"""
function get_yukawa_one_loop_neft(rslist, beta; neval=1e6, seed=1234)
    # One-loop => only interaction-type counterterms
    p = UEG.partition(1; offset=0)  # TEST: G-type counterterms should be removed by Diagram.diagram_parquet_response!

    dummy_paramc = ElectronLiquid.ParaMC(;
        rs=0.01,
        beta=beta,
        Fs=0.0,
        order=1,
        mass2=0.0,
        isDynamic=false,
    )

    # NOTE: we need to include the bubble diagram, as it will be cancelled numerically by the interaction counterterm
    filter = [Proper, NoHartree]  # Proper => only exchange and box-type direct diagrams contribute to F₂

    # returns: (partition, diagpara, FeynGraphs, extT_labels, spin_conventions)
    # graphs contain: {{↑↑ diags}, {↑↓ diags}}
    diagrams = Diagram.diagram_parquet_response(
        :vertex4,
        dummy_paramc,
        p;
        filter=filter,
        transferLoop=zeros(16),  # (Q,Ω) → 0
    )
    graphs = diagrams[3]
    isUpUp =
        Dict(P => [g.properties.response == UpUp for g in graphs[P]] for P in keys(graphs))
    isUpDown = Dict(
        P => [g.properties.response == UpDown for g in graphs[P]] for P in keys(graphs)
    )
    println("Tree-level computational graphs from Diagram.diagram_parquet_response:")
    print_tree(graphs)
    println("isUpUp: $isUpUp")
    println("isUpDown: $isUpDown")

    F1s = []
    Fs2s = []
    Fa2s = []
    Fuu2s = []
    Fud2s = []
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

        Fs_DMC = -get_Fs_DMC(basic)
        Fa_DMC = -get_Fa_DMC(basic)
        println_root("F+ from DMC: $(Fs_DMC), F- from DMC: $(Fa_DMC)")

        ########## test l=0 PH averged Yukawa interaction ############
        data, result = Ver4.lavg(
            paramc,
            diagrams;
            neval=neval,
            l=[0],
            n=[-1, 0, 0],
            seed=seed,
            print=-1,
        )

        # (0, 0, 0)
        obs_tl = real(data[p[1]])

        # (1, 0, 0)
        obs = real(data[p[2]])

        # # (0, 1, 0)
        # obs_gct = real(data[p[3]])
        # println("(G counterterms)\tup-up: $(obs_gct[1]), up-down: $(obs_gct[2])")

        # (0, 0, 1)
        obs_ict = real(data[p[4]])

        @assert length(obs_tl) == 2                  # obs = {↑↑Ins, ↑↓Ins}
        @assert length(obs) == length(obs_ict) == 6  # obs = {PHr ↑↑Dyn, PPr ↑↑Dyn, PHEr ↑↑Dyn, PHr ↑↓Dyn, PPr ↑↓Dyn, PHEr ↑↓Dyn}

        F1uu = -isUpUp' * obs_tl
        F1ud = -isUpDown' * obs_tl

        F2uu = -isUpUp' * obs
        F2ud = -isUpDown' * obs

        F2ctuu = -isUpUp' * obs_ict
        F2ctud = -isUpDown' * obs_ict

        println("(tree-level)\tup-up: $F1uu, up-down: $F1ud")
        println("(one-loop diagrams)\tup-up: $F2uu, up-down: $F2ud")
        println("(one-loop counterterms)\tup-up: $F2ctuu, up-down: $F2ctud")

        Fp(Fuu, Fud) = (Fuu + Fud) / 2
        Fm(Fuu, Fud) = (Fuu - Fud) / 2

        F1p = Fp(F1uu, F1ud)
        F1m = Fm(F1uu, F1ud)
        F1 = F1p
        @assert F1p ≈ F1m "F1p = $F1p, F1m = $F1m (should be equal!)"

        F2p = Fp(F2uu, F2ud)
        F2m = Fm(F2uu, F2ud)
        F2ctp = Fp(F2ctuu, F2ctud)
        F2ctm = Fm(F2ctuu, F2ctud)
        F2s = F2p + F2ctp
        F2a = F2m + F2ctm
        F2totaluu = F2uu + F2ctuu
        F2totalud = F2ud + F2ctud
        println("Fs = ($(F1))ξ + ($(F2s))ξ² + O(ξ³)")
        println("Fa = ($(F1))ξ + ($(F2a))ξ² + O(ξ³)")
        println("F↑↑ = ($(F1uu))ξ + ($(F2totaluu))ξ² + O(ξ³)")
        println("F↑↓ = ($(F1ud))ξ + ($(F2totalud))ξ² + O(ξ³)")

        push!(FsDMCs, Fs_DMC)
        push!(FaDMCs, Fa_DMC)
        push!(F1s, F1)
        push!(Fs2s, F2s)
        push!(Fa2s, F2a)
        push!(Fuu2s, F2totaluu)
        push!(Fud2s, F2totalud)
        GC.gc()
    end
    return FsDMCs, FaDMCs, F1s, Fs2s, Fa2s, Fuu2s, Fud2s
end
