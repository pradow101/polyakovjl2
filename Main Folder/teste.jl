begin
    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, Optim
    include("parameters.jl")
    include("functions.jl")

    # Switch to plotly backend
    plotly()
end

function gapsolver(mu,T,chuteinit)
    sistema = nlsolve(x->(dM(x[1],x[2],mu,T,x[3]),dphi(x[1],x[2],mu,T,x[3]),dphib(x[1],x[2],mu,T,x[3])),chuteinit)
    return sistema.zero
end

begin
    T_vals = range(0.001, 0.3, length = 50) #range for T
    phi_vals = zeros(length(T_vals)) # Arrays which will store the phi, phib and M solutions
    phib_vals = zeros(length(T_vals))
    M_vals = zeros(length(T_vals))
    mu = 0
    chuteinit = [0.01,0.01,0.3]
    for i in 1:length(T_vals) #Initial guess
        T = T_vals[i]  #Tells the program to use the ith value of the T_vals array
        solution = gapsolver(mu, T, chuteinit) #Call gapsolver function, store it in the variable solution
        phi_vals[i] = solution[1] #solution is a vector of 3 floats, and we are storing the first one in phi_vals[i],
        phib_vals[i] = solution[2] #the second one in phib_vals[i], and the third one in M_vals[i]
        M_vals[i] = solution[3]
        chuteinit = solution #update the initial guess with the previous solution
    end
    plot(T_vals, [M_vals, phi_vals], grid=true, gridalpha=0.5, xlabel = "T", ylabel = "phi", title = "Potential vs M and phi")
end

##debugging
begin
    dmvals = zeros(length(T_vals))
    dphivals = zeros(length(T_vals))
    for i in 1:length(T_vals)
        T = T_vals[i]
        dmvals[i] = abs(dM(phi_vals[i], phib_vals[i], mu, T_vals[i], M_vals[i]))
        dphivals[i] = abs(dphi(phi_vals[i], phib_vals[i], mu, T_vals[i], M_vals[i]))
    end
    df = DataFrame(T=T_vals,dM=dmvals)
    df2 = DataFrame(T=T_vals,dphi=dphivals)
    CSV.write("dphi.csv", df2)
    CSV.write("dM.csv", df)
    plot(T_vals, [dmvals,dphivals], label = ["dM" "dphi"], grid=true, gridalpha=0.5, xlabel = "T", ylabel = "dm/dphi")
end


begin
    # Define the range of M and phi values
    Mi = range(0.01, 0.5, length = 500)  # Range for M
    phi_k = range(-0.5, 0.5, length = 500)  # Range for phi

    # Create a grid of M and phi values
    M_grid = repeat(Mi, 1, length(phi_k))  # Repeat M values along rows
    phi_grid = repeat(phi_k', length(Mi), 1)  # Repeat phi values along columns

    # Compute yi for each pair of (M, phi) on the grid
    yi = [potential(phi, 0.01, 0.35, M, 0.2) for (M, phi) in zip(M_grid[:], phi_grid[:])]

    # Reshape yi to match the grid dimensions
    yi = reshape(yi, length(Mi), length(phi_k))

    # Plot the 3D surface
    plot3d(Mi, phi_k, yi, st=:surface, xlabel = "M", ylabel = "phi", zlabel = "potential", title = "Potential vs M and phi", grid=true, gridalpha=0.5)
end

function gapsolver(mu,T,phi,chuteinit)
    sistema = nlsolve(x->(dM(phi,x[1],mu,T,x[2]),dphib(phi,x[1],mu,T,x[2])),chuteinit)
    return sistema.zero
end

begin
    chuteinit = [0.1,0.3]
    phi_loop = range(0,0.7, length = 100)
    M_valores = zeros(length(phi_loop))
    phib_valores = zeros(length(phi_loop))
    pot_sols = zeros(length(phi_loop))
    mu = 0.34
    T = 0.05
    for i in 1:length(phi_loop)
        phi = phi_loop[i]
        solution = gapsolver(mu, T, phi, chuteinit) #Call gapsolver function, store it in the variable solution
        M_valores[i] = solution[1] #solution is a vector of 3 floats, and we are storing the first one in phi_vals[i],
        phib_valores[i] = solution[2] #the second one in phib_vals[i], and the third one in M_vals[i]
        chuteinit = solution
        
        pot_sols[i] = potential(phi, phib_valores[i], mu, T, M_valores[i])
    end
    plot(phi_loop, pot_sols, grid=true, gridalpha=0.5, xlabel = "M", ylabel = "potential", title = "Potential vs M")
end

begin
    function logarg(phi, phib, mu, T, M, p)
        return 1 + 3*(phi + phib*exp(-(Ep(p,M) - mu)/T))*exp(-(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T)
    end
    phi_loop = range(0,0.7, length = 100)
    M_valores = zeros(length(phi_loop))
    phib_valores = zeros(length(phi_loop))
    pot_sols = zeros(length(phi_loop))
    yi = zeros(length(phi_loop))
    for i in 1:length(phi_loop)
        phi = phi_loop[i]
        phib = phib_valores[i]
        mu = 0.34
        T = 0.05
        M = M_valores[i]
        yi[i] = logarg(phi, phib, mu, T, M, 0.5)    
    end

    plot(phi_loop, yi)
end


begin
    res = optimize(x -> potential(0.01, 0.01, mu, T, x[1]), chuteinit)
    chuteinit = [0.4]
    mu, T = 0, 0.01
    print(res)
    Mrange = range(-0.5, 0.5, length = 100)

    # yi = [potential(0.01,0.01,0,0.01,M) for M in Mrange]
    # plot(Mrange, yi, xlabel = "M", ylabel = "potential", title = "Potential vs M")

    a = nlsolve(x-> potential(0.01,0.01,0,0.01,x[1]) + 2.802303e-02, chuteinit)
    println(a.zero)

    b = dM(0.01,0.01,0,0.01,0.3261988478194129)
    println(b)

    #extrair resultado do optimize, calcular os valores da massa, phi e phib pois
    #optimize retorna valores do potencial
end




##PRESSÃO DE STEFAN BOLTZMANN
begin
    trange = range(0.3, 2, length = 100)
    pf(T) = T^4*(8π^2/45 + 42π^2/180)
    phisol = zeros(length(trange))
    phibsol = zeros(length(trange))
    Msol = zeros(length(trange))
    p = zeros(length(trange))
    mu = 0
    chuteinit = [0.01,0.01,0.3]
    for i in 1:length(trange)
        T = trange[i]/0.19
        solsarray = gapsolver(mu, T, chuteinit)
        phisol[i] = solsarray[1]
        phibsol[i] = solsarray[2]
        Msol[i] = solsarray[3]
        chuteinit = solsarray
        p[i] = -potential(phisol[i], phibsol[i], mu, T, Msol[i])/pf(T)
    end
    plot(trange, p)
end



#Plotar p = -Ω/pf e comparar com figura 7 do Ratti. Checar a parametrização


begin
    function CEPfinder(chuteinit)
        sistema = nlsolve(x->(dM(x[1],x[2],x[3],x[4],x[5]),dphi(x[1],x[2],x[3],x[4],x[5]),dphib(x[1],x[2],x[3],x[4],x[5]),eq1(x[1],x[2],x[3],x[4],x[5]),eq2(x[1],x[2],x[3],x[4],x[5])),chuteinit)
        return sistema.zero
    end
    a = CEPfinder([0.01,0.01,0.3,0.1,0.1])
    println(a)

end