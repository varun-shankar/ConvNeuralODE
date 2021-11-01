using Flux, DiffEqFlux, DiffEqSensitivity

basic_tgrad(u,p,t) = zero(u)

struct NODE{M,P,RE,T,S,A,K}
    model::M
    p::P
    re::RE
    tspan::T
    solver::S
    args::A
    kwargs::K
end

function NODE(model,tspan,solver=nothing,args...;kwargs...)
    p,re = Flux.destructure(model)
    NODE{typeof(model),typeof(p),typeof(re),
        typeof(tspan),typeof(solver),typeof(args),typeof(kwargs)}(
        model,p,re,tspan,solver,args,kwargs)
end

Flux.@functor NODE

function (n::NODE)(x,p=n.p)
    dudt_(u,p,t) = n.re(p)(u)
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,x,n.tspan,p)
    concrete_solve(prob,n.solver,x,p,n.args...;n.kwargs...)
end

Flux.trainable(m::NODE) = (m.p,)
