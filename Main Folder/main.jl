begin
    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations

    include("parameters.jl")
    include("functions.jl")

    plotly()
end

function gapsolver(mu,T,chutealto)
    sistema1 = nlsolve(x->(dM(x[1],x[2],mu,T,x[3]),dphi(x[1],x[2],mu,T,x[3]),dphib(x[1],x[2],mu,T,x[3])),chutealto)
    return sistema1.zero
end


function maxfind(y, T) #Here x will play the role of the derivatives of phi, phib and M solutions for a given μ
    for i in 100:length(y)-1
        if y[i+1] < y[i] && y[i-1] < y[i]  
            return T[i], y[i]
        end
    end
    return NaN, NaN # Return NaN if no maximum is found
end

begin
    function Trangesolver(mu, T_vals)
        phi_vals = zeros(length(T_vals)) # Arrays which will store the phi, phib and M solutions
        phib_vals = zeros(length(T_vals))
        M_vals = zeros(length(T_vals))
        chutealto = [0.01,0.01,0.4]
        for i in 1:length(T_vals) #Initial guess
            T = T_vals[i]  #Tells the program to use the ith value of the T_vals array
            solution = gapsolver(mu, T, chutealto) #Call gapsolver function, store it in the variable solution
            phi_vals[i] = solution[1] #solution is a vector of 3 floats, and we are storing the first one in phi_vals[i],
            phib_vals[i] = solution[2] #the second one in phib_vals[i], and the third one in M_vals[i]
            M_vals[i] = solution[3]
            #here I need to implement a condition to switch initial guess values
            chutealto = solution
        end
        return T_vals, phi_vals, phib_vals, M_vals
    end

    function Interp(T_vals, phi_vals, phib_vals, M_vals)
        itpM = interpolate((T_vals,), M_vals, Gridded(Linear()))
        itpphi = interpolate((T_vals,), phi_vals, Gridded(Linear()))
        itpphib = interpolate((T_vals,), phib_vals, Gridded(Linear()))
        interp = zeros(length(T_vals), 4)
        derinterp = zeros(length(T_vals), 4)
        interp[:,1] = T_vals
        derinterp[:,1] = T_vals
        for i in range(1, length(T_vals))
            interp[i,2] = itpM(T_vals[i])
            interp[i,3] = itpphi(T_vals[i])
            interp[i,4] = itpphib(T_vals[i])
            derinterp[i,2] = only(Interpolations.gradient(itpphi, T_vals[i]))
            derinterp[i,3] = only(Interpolations.gradient(itpphib, T_vals[i]))
            derinterp[i,4] = -only(Interpolations.gradient(itpM, T_vals[i]))  
        end
        return derinterp
    end

    function murangesolver(T_vals)
        mu_vals = range(0,0.331,length=150)
        solutions = zeros(length(mu_vals), length(T_vals), 4)
        println(size(solutions))
        println(size(mu_vals))

        Threads.@threads for i in eachindex(mu_vals)
            solsi = Trangesolver(mu_vals[i], T_vals)
            #println(size(solsi))
            solutions[i,:,1] = solsi[1]
            solutions[i,:,2] = solsi[2]
            solutions[i,:,3] = solsi[3]
            solutions[i,:,4] = solsi[4]
        end
        return solutions, mu_vals
    end
end


function gapsolvedensidade(T, chuteinit, nb)
    sistema = nlsolve(x->(dM(x[1],x[2],x[3],T,x[4]),dphi(x[1],x[2],x[3],T,x[4]),dphib(x[1],x[2],x[3],T,x[4]),densidade(x[1],x[2],x[3],T,x[4],nb)),chuteinit)
    return sistema.zero
end


begin
    CEP = nlsolve(x -> (dM(x[1],x[2],x[3],x[4],x[5]), dphi(x[1],x[2],x[3],x[4],x[5]), dphib(x[1],x[2],x[3],x[4],x[5]), eq1(x[1],x[2],x[3],x[4],x[5]), eq2(x[1],x[2],x[3],x[4],x[5])), [0.1, 0.1, 0.4, 0.4, 0.32]).zero
    println("phi = $(CEP[1]), phib = $(CEP[2]), mu = $(CEP[3]), T = $(CEP[4]), M = $(CEP[5])")
    println("CEP = $(CEP)")
end


@time begin
    T_vals = range(0.04,0.4,1500)
    murange, muvalores = murangesolver(T_vals)
end

begin
    function Trange_density(T)
        Nbvals = range(0.0001,0.01,length=100)
        phi_vals = zeros(length(Nbvals)) # Arrays which will store the phi, phib and M solutions
        phib_vals = zeros(length(Nbvals))
        M_vals = zeros(length(Nbvals))
        mu_vals = zeros(length(Nbvals))
        potential_vals = zeros(length(Nbvals))
        chuteinit = [0.01,0.01,0.4,0.4]
        for i in 1:length(Nbvals) #Initial guess
            nb = Nbvals[i]  #Tells the program to use the ith value of the T_vals array
            solution = gapsolvedensidade(T, chuteinit, nb) #Call gapsolver function, store it in the variable solution
            phi_vals[i] = solution[1] #solution is a vector of 3 floats, and we are storing the first one in phi_vals[i],
            phib_vals[i] = solution[2] #the second one in phib_vals[i], and the third one in M_vals[i]
            mu_vals[i] = solution[3]
            M_vals[i] = solution[4]
            chuteinit = solution
            potential_vals[i] = potential(phi_vals[i], phib_vals[i], mu_vals[i], T, M_vals[i])
        end
        return Nbvals, mu_vals, phi_vals, phib_vals, M_vals, potential_vals
    end
end



begin
    #=aqui, preciso dar um jeito de achar o valor para qual o potencial começa a voltar
    vou usar a condição de quando o valor de mu aumenta pela primeira vez, de depois quando volta a cair. =#
    function interpot(pvals, muvals)
        firstcurvex = []
        firstcurvey = []
        secondcurvex = []
        secondcurvey = []
        for i in 2:length(muvals)
            if muvals[i] < muvals[i-1]
                break
            end
            append!(firstcurvey, pvals[i])
            append!(firstcurvex, muvals[i])
        end
        for i in length(muvals)-1:-1:2
            if muvals[i] < muvals[i-1]
                break
            end
            append!(secondcurvey, pvals[i])
            append!(secondcurvex, muvals[i])
        end
        return firstcurvex, firstcurvey, secondcurvex, secondcurvey
    end
end

let
    _,muvalores,_,_,_,potentialvalores = Trange_density(0.04)
    curva1x, curva1y, curva2x, curva2y = interpot(potentialvalores, muvalores)

    x1 = Vector{Float64}(curva1x)
    y1 = Vector{Float64}(curva1y)

    x2 = reverse(Vector{Float64}(curva2x))
    y2 = reverse(Vector{Float64}(curva2y))

    interpolacao1 = DataInterpolations.LinearInterpolation(y1, x1; extrapolate=true)
    interpolacao2 = DataInterpolations.LinearInterpolation(y2, x2; extrapolate=true)

    yi1 = [interpolacao1(y) for y in muvalores]
    yi2 = [interpolacao2(y) for y in muvalores]

    plot(muvalores,yi2)
    plot!(muvalores, potentialvalores)
    scatter!(x2,y2)

    # df = DataFrame(mu=x2, potential=y2)
    # CSV.write("potential_data.csv", df)
end

let 
    a = range(1,10,10)
    for (ai,i) in enumerate(a)
        println(ai, i)
    end    
end





begin
    function fofinder(T, chuteinit)
        Nbvals, mu_vals, phi_vals, phib_vals, M_vals, potential_vals = Trange_density(T)
        firstcurvex, firstcurvey, secondcurvex, secondcurvey = interpot(potential_vals, mu_vals)

        x1 = Vector{Float64}(firstcurvex)
        y1 = Vector{Float64}(firstcurvey)
        x2 = reverse(Vector{Float64}(secondcurvex))
        y2 = reverse(Vector{Float64}(secondcurvey))
        
        interp1 = DataInterpolations.LinearInterpolation(y1, x1; extrapolate=true)
        interp2 = DataInterpolations.QuadraticInterpolation(y2, x2; extrapolate=true)
        
        # return interp1, interp2
        diferenca(mu) = interp1(mu) - interp2(mu)
        
        mucritico = nlsolve(x -> [diferenca(x[1])], [chuteinit], method=:newton)
        return mucritico.zero[1], interp2(mucritico.zero[1]), interp1, interp2, x2, y2, x1, y1
    end
end
# begin
#     T_vals = range(0.01,0.1,150)
#     mu = 0.34
#     T_vals, phi_vals, phib_vals, M_vals = Trangesolver(mu, T_vals)
#     plot(T_vals, [M_vals,phi_vals, phib_vals], grid=true, gridalpha=0.5, xlabel = "T", ylabel = "phi, phib", title = "phi and phib solutions", linewidth = 2, label = ["M" "ϕ" "ϕ*"])

# end


begin
    T_valores = zeros(length(murange[1,:,1]),length(muvalores))
    phi_valores = zeros(length(murange[1,:,1]),length(muvalores))
    phib_valores = zeros(length(murange[1,:,1]),length(muvalores))
    M_valores = zeros(length(murange[1,:,1]),length(muvalores))
    for i in eachindex(muvalores)
        T_vals, phi_vals, phib_vals, M_vals = murange[i,:,1], murange[i,:,2], murange[i,:,3], murange[i,:,4]
        interploop = Interp(T_vals, phi_vals, phib_vals, M_vals)
        T_valores[:,i] = interploop[:,1]
        phi_valores[:,i] = interploop[:,2]
        phib_valores[:,i] = interploop[:,3]
        M_valores[:,i] = interploop[:,4]
    end
end 

let 
    
end 







#plot(T_vals, [M_vals], grid=true, gridalpha=0.5, xlabel = "T", ylabel = "phi, M", title = "M and phi solutions")end

        
# begin       #calculating and plotting the pressure
#     T_vals = range(0.1,0.4,500)
#     mu = 0
#     T_vals, phi_vals, phib_vals, M_vals = Trangesolver(mu, T_vals)
#     pf_vals = zeros(length(T_vals))
#     for i in 1:length(T_vals)
#         T = T_vals[i]
#         phi = phi_vals[i]
#         phib = phib_vals[i]
#         M = M_vals[i]
#         pf_vals[i] = -(potential(phi, phib, 0, T, M) - potential(phi, phib, 0, 0.001, M))/pf(T) #pressure
#     end
#     plot(T_vals, pf_vals, grid = true, gridalpha=0.5, xlabel = "T", ylabel = "Pressure", title = "Pressure vs T", xrange = (0.1,0.395))
# end

begin
    spin1x = []
    spin1y = []
    spin2x = []
    spin2y = []
    Ti = range(0.01, 0.065, length=100)
    for i in 1:length(Ti)
        T = Ti[i]
        _, muvalsspin, _, _, _, pvalsspin = Trange_density(T)
        x1spin, y1spin, x2spin, y2spin = interpot(pvalsspin, muvalsspin)
        append!(spin1x, x1spin[end])
        append!(spin1y, Ti[i])
        append!(spin2x, x2spin[end])
        append!(spin2y, Ti[i])
    end

    plot(spin1x, spin1y)
    plot!(spin2x, spin2y)
end

begin
    Ttransitionphi = zeros(length(muvalores))
    Ttransitionphib = zeros(length(muvalores))
    actualphi = zeros(length(muvalores))
    TtransitionM = zeros(length(muvalores))
    Mutransition = muvalores
    Threads.@threads for i in 1:length(muvalores)
        Ttransitionphi[i] = maxfind(phi_valores[:,i], T_valores[:,1])[1]
        Ttransitionphib[i] = maxfind(phib_valores[:,i], T_valores[:,1])[1]
        TtransitionM[i] = maxfind(M_valores[:,i], T_valores[:,1])[1]
        actualphi[i] = (Ttransitionphi[i] + Ttransitionphib[i])/2
    end
    Trange = range(0.01, 0.065, length=50)  # Replace with your desired range
    mucriticos = zeros(length(Trange))
    Threads.@threads for i in 1:length(Trange)
        chuteinit = 0.34
        mucriticos[i] = fofinder(Trange[i], chuteinit)[1]
        chuteinit = mucriticos[i]
    end
    p = plot(mucriticos, Trange, label = "First Order", linewidth = 3)
    plot!(p, Mutransition, [Ttransitionphi, actualphi], 
    label = ["ϕ crossover" "M crossover"],
    xlabel = "μ [GeV]",
    ylabel = "T [GeV]",
    title = "PNJL Phase Diagram", dpi=800, linewidth = 3, linestyle = :dash)
    scatter!(p, [0.331795119246286],[0.06557512526501373], label = "CEP")
    plot!(p, spin1x, spin1y, linestyle = :dashdot, label = "Spinodal 1")
    plot!(p, spin2x, spin2y, linestyle = :dashdot, label = "Spinodal 2")
    plot!(p, mu_quark, T_quark, label = "Quarkyonic Phase", linestyle = :dot)
    #savefig(p, "PNJL_Phase_Diagram.png")
end




begin
    function gapsolvermu(mu, T, chute)
        sistema = nlsolve(x->(dM(x[1],x[2],mu,T,x[3]),dphi(x[1],x[2],mu,T,x[3]),dphib(x[1],x[2],mu,T,x[3])),chute)
        return sistema.zero
    end

    function musolver(mu_r, T)
        μphi = zeros(length(mu_r))
        μphib = zeros(length(mu_r))
        μM = zeros(length(mu_r))
        chute = [0.01,0.01,0.4]
        for i in 1:length(mu_r)
            mu = mu_r[i]
            solution = gapsolvermu(mu, T, chute)
            μphi[i] = solution[1]
            μphib[i] = solution[2]
            μM[i] = solution[3]
            chute = solution
        end
        return mu_r, μphi, μphib, μM
    end

    function musolvertrange(mu_r, T_r)
        solsarray = zeros(length(T_r), length(mu_r), 4)
        Threads.@threads for i in eachindex(T_r)
            solarrayi = musolver(mu_r, T_r[i])
            solsarray[i,:,1] = solarrayi[1]
            solsarray[i,:,2] = solarrayi[2]
            solsarray[i,:,3] = solarrayi[3]
            solsarray[i,:,4] = solarrayi[4]
        end
        return solsarray, mu_r, T_r
    end
end

#==
ok, here sols[i,j,1] = T, sols[i,j,2] = ϕ, sols[i,j,3] = ϕ*, sols[i,j,4] = M
i is the ith value of T-r, and j is the jth value of mu_r. So, here, 
i'm plotting ϕ, ϕ* and M against μ, given the 50th value of T (out of 100).
==#

begin
    murange = range(0,1,length=5000)
    trange = range(0.065,0.15,length=150)
    sols, mu_r, t_r = musolvertrange(murange, trange)

    plot(mu_r, [sols[50,:,2], sols[50,:,3], sols[50,:,4]], xlabel = "μ [GeV]", ylabel = "ϕ, ϕ*, M")
end

#==
for each value of T, i want to check for which value of μ
is bigger than the value of M, correspondant to that value of T.
==#

begin
    function quarkyonic(mu_r, t_r, sols)
        T_quark = zeros(length(t_r))
        mu_quark = zeros(length(t_r))
        for i in 1:length(t_r)
            for j in 1:length(mu_r)
                if sols[i,j,4] < mu_r[j]
                    T_quark[i] = t_r[i]
                    mu_quark[i] = mu_r[j]
                    break
                end
            end
        end
        return mu_quark, T_quark
    end
end


begin
    mu_quark, T_quark = quarkyonic(mu_r, t_r, sols)
    plot(mu_quark, T_quark)
end