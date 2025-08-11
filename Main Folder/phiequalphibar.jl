begin
    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations
    include("parameters.jl")
end

Ep(p, M) = sqrt(p^2 + M^2)
pf(T) = (T^4)*(8π^2/45 + 7π^2/30)

potplkv(phi, T) = T^4*((-(a0 + a1*(t0/T) + a2*(t0/T)^2 + a3*(t0/T)^3)/2)*(phi^2) - (b3/3)*(phi^3) + (b4/4)*(phi)^4)

zminus(phi, M, mu, T, p) = log(1 + 3*(phi + phi*exp(-(Ep(p,M) - mu)/T))*exp(-(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T))
zplus(phi, M, mu, T, p) = log(1 + 3*(phi + phi*exp(-(Ep(p,M) + mu)/T))*exp(-(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T))

function Imed(phi, M, mu, T)
    quadgk(p -> p^2 * (zminus(phi,M,mu,T,p) + zplus(phi,M,mu,T,p)), 0, Inf)[1]
end

function Ivac(M)
    quadgk(p -> p^2 * Ep(p,M), 0, lamb)[1]
end

function potential(phi, mu, T, M)
    (M-m)^2 / (4G) - T*Nf*Imed(phi, M, mu, T)/π^2 - 3*Nf*Ivac(M)/π^2 + potplkv(phi, T)
end

function dphi(phi, mu, T, M)
    ForwardDiff.derivative(phix -> potential(phix, mu, T, M), phi)
end

function dM(phi, mu, T, M)
    ForwardDiff.derivative(Mi -> potential(phi, mu, T, Mi), M)
end

function dM2(phi, mu, T, M)
    ForwardDiff.derivative(Mi -> dM(phi, mu, T, Mi), M)
end

function dM3(phi, mu, T, M)
    ForwardDiff.derivative(Mi -> dM2(phi, mu, T, Mi), M)
end

function dMu(phi, mu, T, M)
    ForwardDiff.derivative(mui -> dM(phi, mui, T, M), mu)
end

function eq1(phi, mu, T, M)
    a = dM2(phi, mu, T, M)
    b = dMu(phi, mu, T, M)
    return a/b
end

function eq2(phi, mu, T, M)
    a = dM3(phi, mu, T, M)
    b = dMu(phi, mu, T, M)
    return a/b
end

function densidade(phi, mu, T, M, nb)
    a = ForwardDiff.derivative(mui -> potential(phi, mui, T, M), mu)
    return a + nb
end

function gapsolver(mu, T, chutealto)
    sistema1 = nlsolve(x -> [
        dM(x[1], mu, T, x[2]),
        dphi(x[1], mu, T, x[2])
    ], chutealto)
    return sistema1.zero
end

function maxfind(y, T)
    for i in 100:length(y)-1
        if y[i+1] < y[i] && y[i-1] < y[i]
            return T[i], y[i]
        end
    end
    return NaN, NaN
end

begin
    function Trangesolver(mu, T_vals)
        phi_vals = zeros(length(T_vals))
        M_vals = zeros(length(T_vals))
        chutealto = [0.01, 0.4]
        for i in 1:length(T_vals)
            T = T_vals[i]
            solution = gapsolver(mu, T, chutealto)
            phi_vals[i] = solution[1]
            M_vals[i] = solution[2]
            chutealto = solution
        end
        return T_vals, phi_vals, M_vals
    end

    function Interp(T_vals, phi_vals, M_vals)
        itpM = interpolate((T_vals,), M_vals, Gridded(Linear()))
        itpphi = interpolate((T_vals,), phi_vals, Gridded(Linear()))
        interp = zeros(length(T_vals), 3)
        derinterp = zeros(length(T_vals), 3)
        interp[:,1] = T_vals
        derinterp[:,1] = T_vals
        for i in 1:length(T_vals)
            interp[i,2] = itpM(T_vals[i])
            interp[i,3] = itpphi(T_vals[i])
            derinterp[i,2] = -only(Interpolations.gradient(itpM, T_vals[i]))
            derinterp[i,3] = only(Interpolations.gradient(itpphi, T_vals[i]))
        end
        return derinterp
    end

    function murangesolver(T_vals)
        mu_vals = range(0, 0.331, length=150)
        solutions = zeros(length(mu_vals), length(T_vals), 4)
        println(size(solutions))
        println(size(mu_vals))

        Threads.@threads for i in eachindex(mu_vals)
            solsi = Trangesolver(mu_vals[i], T_vals)
            solutions[i,:,1] = solsi[1]
            solutions[i,:,2] = solsi[2]
            solutions[i,:,3] = solsi[3]
        end
        return solutions, mu_vals
    end
end

begin
    function gapsolvedensidade(T, chuteinit, nb)
        sistema = nlsolve(x->(dM(x[1],x[2],x[3],T,x[4]),dphi(x[1],x[2],x[3],T,x[4]),dphib(x[1],x[2],x[3],T,x[4]),densidade(x[1],x[2],x[3],T,x[4],nb)),chuteinit)
        return sistema.zero
    end

    function Trange_density(T)
        Nbvals = range(0.0001,0.01,length=100)
        phi_vals = zeros(length(Nbvals)) # Arrays which will store the phi, phib and M solutions
        M_vals = zeros(length(Nbvals))
        mu_vals = zeros(length(Nbvals))
        potential_vals = zeros(length(Nbvals))
        chuteinit = [0.01,0.01,0.4,0.4]
        for i in 1:length(Nbvals) #Initial guess
            nb = Nbvals[i]  #Tells the program to use the ith value of the T_vals array
            solution = gapsolvedensidade(T, chuteinit, nb) #Call gapsolver function, store it in the variable solution
            phi_vals[i] = solution[1] #solution is a vector of 3 floats, and we are storing the first one in phi_vals[i],
            mu_vals[i] = solution[2]
            M_vals[i] = solution[3]
            chuteinit = solution
            potential_vals[i] = potential(phi_vals[i], mu_vals[i], T, M_vals[i])
        end
        return Nbvals, mu_vals, phi_vals, M_vals, potential_vals
    end

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

    function fofinder(T, chuteinit)
        Nbvals, mu_vals, phi_vals, M_vals, potential_vals = Trange_density(T)
        firstcurvex, firstcurvey, secondcurvex, secondcurvey = interpot(potential_vals, mu_vals)

        x1 = Vector{Float64}(firstcurvex)
        y1 = Vector{Float64}(firstcurvey)
        x2 = Vector{Float64}(secondcurvex)
        y2 = Vector{Float64}(secondcurvey)

        interp1 = DataInterpolations.LinearInterpolation(y1, x1; extrapolate=true)
        interp2 = DataInterpolations.LinearInterpolation(y2, x2; extrapolate=true)

        # return interp1, interp2
        diferenca(mu) = interp1(mu) - interp2(mu)

        mucritico = nlsolve(x -> [diferenca(x[1])], [chuteinit], method=:newton)
        return mucritico.zero[1]
    end
end

@time begin
    T_vals = collect(range(0.04, 0.4, length=1500))
    murange, muvalores = murangesolver(T_vals)
end

begin
    T_valores = zeros(length(murange[1,:,1]), length(muvalores))
    phi_valores = zeros(length(murange[1,:,1]), length(muvalores))
    M_valores = zeros(length(murange[1,:,1]), length(muvalores))

    for i in eachindex(muvalores)
        T_vals_loop, phi_vals, M_vals = murange[i,:,1], murange[i,:,2], murange[i,:,3]
        interploop = Interp(T_vals_loop, phi_vals, M_vals)
        T_valores[:,i] = interploop[:,1]
        phi_valores[:,i] = interploop[:,2]
        M_valores[:,i] = interploop[:,3]
    end
end

begin
    Ttransitionphi = zeros(length(muvalores))
    TtransitionM = zeros(length(muvalores))
    Mutransition = muvalores
    Threads.@threads for i in 1:length(muvalores)
        Ttransitionphi[i] = maxfind(phi_valores[:,i], T_valores[:,1])[1]
        TtransitionM[i] = maxfind(M_valores[:,i], T_valores[:,1])[1]
    end
    Trange = range(0.01, 0.065, length=50)  # Replace with your desired range
    mucriticos = zeros(length(Trange))
    Threads.@threads for i in 1:length(Trange)
        chuteinit = 0.32
        mucriticos[i] = fofinder(Trange[i], chuteinit)
        chuteinit = mucriticos[i]
    end
    scatter(mucriticos, Trange, legend = false)
    scatter!(Mutransition, [Ttransitionphi, TtransitionM], 
    label = ["ϕ Transition" "M Transition"],
    xlabel = "μ [GeV]",
    ylabel = "T [GeV]",
    title = "PNJL Phase Diagram", dpi=800, linewidth = 3, legend = false)
    scatter!([0.331795119246286],[0.06557512526501373], legend = false)
end
   

begin
    Nbvals, muvals, phivals, Mvals, potentialvals = Trange_density(0.075)
    plot(muvals, potentialvals, label = "Potential", xlabel = "μ [GeV]", ylabel = "Potential [GeV]", title = "Potential at T=0.075 GeV", dpi=800, linewidth = 3)
end