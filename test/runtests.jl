using OneLoopFermiLiquid
using Test

function test_yukawa_tree_level_neft()
    seed = 1234

    # Tree-level => no internal loops or counterterms
    p = (0, 0, 0)

    dummy_paramc = ElectronLiquid.ParaMC(;
        rs=0.01,
        beta=40.0,
        Fs=0.0,
        order=0,
        mass2=0.0,
        isDynamic=false,
    )

    # returns: (partition, diagpara, FeynGraphs, extT_labels, spin_conventions)
    diagrams = Diagram.diagram_parquet_response(
        :vertex4,
        dummy_paramc,
        [p];
        filter=[Proper, NoHartree],  # Proper => only exchange part contribute to F₁
        transferLoop=zeros(3),       # (Q,Ω) → 0
    )
    graphs = diagrams[3]
    println("Tree-level computational graphs from Diagram.diagram_parquet_response:")
    print_tree(graphs)

    rslist = [0.01, 0.1, 1.0, 10.0]
    for rs in rslist
        println("\nTesting for rs = $rs:")
        qTF = Parameter.rydbergUnit(1.0 / 40.0, rs, 3).qTF
        paramc = ElectronLiquid.ParaMC(;
            rs=rs,
            beta=40.0,
            Fs=0.0,
            order=0,
            mass2=qTF^2,  # for a particularly simple dimensionless interaction
            isDynamic=false,
        )
        @unpack e0, NF, mass2 = paramc

        ########## test l=0 PH averged Yukawa interaction ############
        data, result =
            Ver4.lavg(paramc, diagrams; neval=1e5, l=[0], n=[-1, 0, 0], seed=seed, print=-1)
        obs = data[p]
        println("up-up: $(obs[1]), up-down: $(obs[2])")

        # Why are these swapped?
        Fp_MCMC = -real(obs[1] - obs[2]) / 2 # (upup - updn) / 2, extra minus sign, factor of 1/2 already included in lavg
        Fm_MCMC = -real(obs[1] + obs[2]) / 2 # (upup + updn) / 2, extra minus sign, factor of 1/2 already included in lavg
        println("Fp = $Fp_MCMC, Fm = $Fm_MCMC")

        Fp, Fm = Ver4.projected_exchange_interaction(0, paramc, Ver4.exchange_Coulomb)
        println("MCMC for exchange vs NEFT quadrature")
        compare(Fp_MCMC, -Fp)
        compare(Fm_MCMC, -Fm)

        # expect = 0.0
        # compare(real(obs[1]), expect)
        expect = -4π * e0^2 / (mass2) * NF
        println("MCMC for exchange up-down vs exact")
        compare(real(obs[2]), expect)
    end
end

function test_yukawa_one_loop_neft()
    seed = 1234

    # One-loop => only interaction-type counterterms
    p = UEG.partition(1; offset=0)  # TEST: G-type counterterms should be removed by Diagram.diagram_parquet_response!
    # p = [(1, 0, 0), (0, 0, 1)]

    dummy_paramc = ElectronLiquid.ParaMC(;
        rs=0.01,
        beta=40.0,
        Fs=0.0,
        order=1,
        mass2=0.0,
        isDynamic=false,
    )

    # NOTE: we need to include the bubble diagram, as it will be cancelled numerically by the interaction counterterm
    filter = [Proper, NoHartree]  # Proper => only exchange and box-type direct diagrams contribute to F₂

    # returns: (partition, diagpara, FeynGraphs, extT_labels, spin_conventions)
    diagrams = Diagram.diagram_parquet_response(
        :vertex4,
        dummy_paramc,
        p;
        filter=filter,
        transferLoop=zeros(16),  # (Q,Ω) → 0
    )
    graphs = diagrams[3]
    println("Tree-level computational graphs from Diagram.diagram_parquet_response:")
    print_tree(graphs)

    rslist = [0.01, 0.1, 1.0, 10.0]
    for rs in rslist
        println("\nTesting for rs = $rs:")
        qTF = Parameter.rydbergUnit(1.0 / 40.0, rs, 3).qTF
        paramc = ElectronLiquid.ParaMC(;
            rs=rs,
            beta=40.0,
            Fs=0.0,
            order=1,
            mass2=qTF^2,  # for a particularly simple dimensionless interaction
            isDynamic=false,
        )
        @unpack e0, NF, mass2 = paramc

        ########## test l=0 PH averged Yukawa interaction ############
        data, result =
            Ver4.lavg(paramc, diagrams; neval=1e6, l=[0], n=[-1, 0, 0], seed=seed, print=-1)

        # (0, 0, 0)
        obs_tl = real(data[p[1]])

        # (1, 0, 0)
        obs = real(data[p[2]])

        # # (0, 1, 0)
        # obs_gct = real(data[p[3]])
        # println("(G counterterms)\tup-up: $(obs_gct[1]), up-down: $(obs_gct[2])")

        # (0, 0, 1)
        obs_ict = real(data[p[4]])

        # Fp = -real(obs[1] + obs[2]) # upup + updn, extra minus sign, factor of 1/2 already included in lavg
        # Fm = -real(obs[1] - obs[2]) # upup - updn, extra minus sign, factor of 1/2 already included in lavg
        # println("Fp = $Fp, Fm = $Fm")

        exchange_tl = obs_tl[1] - obs_tl[2] # exchange = upup - updn
        exchange = obs[1] - obs[2] # exchange = upup - updn
        exchange_ct = obs_ict[1] - obs_ict[2] # exchange = upup - updn
        println(
            "(tree-level)\tup-up: $(obs_tl[1]), up-down: $(obs_tl[2]), exchange: $(exchange_tl)",
        )
        println(
            "(one-loop diagrams)\tup-up: $(obs[1]), up-down: $(obs[2]), exchange: $(exchange)",
        )
        println(
            "(one-loop counterterms)\tup-up: $(obs_ict[1]), up-down: $(obs_ict[2]), exchange: $(exchange_ct)",
        )

        F1 = exchange_tl
        F2 = exchange + exchange_ct
        println("F = ($(F1))ξ + ($(F2))ξ² + O(ξ³)")

        # TODO: check this—is the test written incorrectly? how can (upup - updn) correspond to Fp?
        Wp, Wm, θgrid = Ver4.exchange_Coulomb(paramc) # Wp = exchanged Coulomb interaction, Wm = 0
        Fp = Ver4.Legrendre(0, Wp, θgrid)
        println("MCMC for tree-level exchange vs NEFT quadrature")
        compare(real(exchange_tl), Fp)

        # expect = 0.0
        # compare(real(obs[1]), expect)
        expect = -4π * e0^2 / (mass2) * NF
        println("MCMC for tree-level up-down vs exact")
        compare(real(obs_tl[2]), expect)
    end
end

@testset "OneLoopFermiLiquid.jl" begin
    # # TODO: refactor tests
    # test_yukawa_tree_level_neft()
    # test_yukawa_one_loop_neft()
end
