begin
    include("parameters.jl")
end

Ep(p,M) = sqrt(p^2 + M^2)

pf(T) = (T^4)*(8π^2/45 + 7π^2/30)

potplkv(phi,phib,T) = T^4*((-(a0 + a1*(t0/T) + a2*(t0/T)^2 + a3*(t0/T)^3)/2)*phib*phi - (b3/6)*(phi^3 + phib^3) + (b4/4)*(phib*phi)^2)

zminus(phi,phib,M,mu,T,p) = log(1 + 3*(phi + phib*exp(-(Ep(p,M) - mu)/T))*exp(-(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T))

zplus(phi,phib,M,mu,T,p) = log(1 + 3*(phib + phi*exp(-(Ep(p,M) + mu)/T))*exp(-(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T))

function Imed(phi,phib,M,mu,T)
    quadgk(p -> p^2 * (zminus(phi,phib,M,mu,T,p) + zplus(phi,phib,M,mu,T,p)), 0, Inf)[1]
end

function Ivac(M)
    quadgk(p -> p^2 * Ep(p,M), 0, lamb)[1]
end

function potential(phi,phib,mu,T,M)
    (M-m)^2/4G - T*Nf*Imed(phi,phib,M,mu,T)/π^2 - 3*Nf*Ivac(M)/π^2 + potplkv(phi,phib,T)
end

function dphi(phi,phib,mu,T,M)
    ForwardDiff.derivative(phix -> potential(phix,phib,mu,T,M), phi)
end

function dphib(phi,phib,mu,T,M)
    ForwardDiff.derivative(phibx -> potential(phi,phibx,mu,T,M), phib)
end

function dM(phi, phib, mu, T, M)
    ForwardDiff.derivative(Mi -> potential(phi, phib, mu, T, Mi), M)
end

function dM2(phi, phib, mu, T, M)
    ForwardDiff.derivative(Mi -> dM(phi, phib, mu, T, Mi), M)
end

function dM3(phi, phib, mu, T, M)
    ForwardDiff.derivative(Mi -> dM2(phi, phib, mu, T, Mi), M)
end

function dMu(phi, phib, mu, T, M)
    ForwardDiff.derivative(mui -> dM(phi, phib, mui, T, M), mu)
end

function eq1(phi, phib, mu, T, M)
    a = dM2(phi, phib, mu, T, M)
    b = dMu(phi, phib, mu, T, M)
    return a/b
end

function eq2(phi, phib, mu, T, M)
    a = dM3(phi, phib, mu, T, M)
    b = dMu(phi, phib, mu, T, M)
    return a/b
end

function densidade(phi,phib,mu,T,M,nb)
    a = ForwardDiff.derivative(mui -> potential(phi, phib, mui, T, M), mu)
    return a + nb
end
