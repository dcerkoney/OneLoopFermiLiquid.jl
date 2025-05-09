using FeynmanDiagram

import FeynmanDiagram.FrontEnds: UpUp, Instant
import FeynmanDiagram.Parquet: Interaction, SigmaDiag, DiagPara, mergeby

function recursive_print(diag)
    if typeof(diag) <: FeynmanDiagram.ComputationalGraphs.Graph
        if !isempty(diag.subgraphs)
            print("$(diag.id) $(diag.subgraph_factors)\n")
            for subdiag in diag.subgraphs
                recursive_print(subdiag)
            end
        end
    else
        if !isempty(diag.subdiagram)
            print("$(diag.hash) $(diag.factor) \n")
            for subdiag in diag.subdiagram
                recursive_print(subdiag)
            end
        end
    end
end

function main()
    order = 1
    spin = 2
    KinL, KoutL, KinR = zeros(16), zeros(16), zeros(16)
    KinL[1], KoutL[2], KinR[3] = 1.0, 1.0, 1.0
    # para = GV.diagPara(SigmaDiag, false, spin, order, [NoHartree], KinL - KoutL)
    para = DiagPara(type=SigmaDiag, innerLoopNum=order, interaction=[Interaction(UpUp, [Instant,])], hasTau=true)
    # para = DiagParaF64(type=SigmaDiag, innerLoopNum=2, interaction=[Interaction(ChargeCharge, [Instant,])], hasTau=true)
    parquet_builder = Parquet.build(para)
    diag = parquet_builder.diagram
    # for eachd in diag
    #     print("new diag\n")
    #     recursive_print(eachd)
    # end
    # mergeby version
    d = mergeby(diag[2:end])
    # for eachd in [d[1]]
    #     print("new diag2\n")
    #     recursive_print(eachd)
    # end
    # G = FrontEnds.Graph!(d[1])
    G = [eldest(d[1])]  # drop extraneous Add node at root
    # for d in G
    #     print("graph1\n")
    #     recursive_print(d)
    # end
    print("eval: $(ComputationalGraphs.eval!(G[1]))\n")
    optimize!(G)
    print("eval: $(ComputationalGraphs.eval!(G[1]))\n")
    optimize!(G)
    print("eval: $(ComputationalGraphs.eval!(G[1]))\n")

    # G = [ComputationalGraphs.flatten_prod!(g) for g in G]
    G = [ComputationalGraphs.open_parenthesis(g) for g in G]

    optimize!(G)
    optimize!(G)

    # for d in G
    #     print("graph2\n")
    #     recursive_print(d)
    # end

    # fname = "par_o$(order)_merge_unoptimized"
    fname = "par_o$(order)_merge_opt_spinless_flat"
    filename = "$fname.dot"
    # Compilers.compile_dot(G, filename; diagram_id_map=diagram_id_map)
    Compilers.compile_dot(G, filename)
    run(`dot -Tpdf -o $fname.pdf $filename`)
    # run(`dot -Tpdf -o $fname.pdf -Gnodesep=0.25 -Granksep=0.5 -Gmargin=0 -Gconcentrate=true $filename`)
    # run(`dot -Tpdf -o $fname.pdf -Gnodesep=0.1 -Granksep=0.2 -Gmargin=0 -Gcompound=true $filename`)
    expr, map = Compilers.to_julia_str(G)
    println(expr)
end

main()
