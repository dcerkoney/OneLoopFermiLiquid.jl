function integrand_exchange_crossed_box_diagram_fixed_θ12()
    res_uu = 0
    res_ud = 0
    return res_uu, res_ud
end

function vegas_exchange_crossed_box_diagram_fixed_θ12(param::OneLoopParams)
    # The PHr contribution is solely due to the exchange, crossed box diagram
    @unpack θ12 = param
    
    # sample:    θ12, iνₘ
    # integrate: y (q = y/(1-y)), θ, ϕ

    # ...
    return
end