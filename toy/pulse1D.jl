using DifferentialEquations
using Flux
using DiffEqFlux
using BSON

using CuArrays
using CUDAnative
CuArrays.allowscalar(false)
CUDAnative.device!(1)

################################################################################
### Functions ###
shiftf(x) = vcat(x[end:end], x[1:end-1])

shiftb(x) = vcat(x[2:end], x[1:1])

# Analytical soln of impulse @x,t=0
function phi_a(x,t,u,Γ)
    phi = (1 ./sqrt.(4 .*pi.*Γ.*t)).*exp.(-(x.-u.*t).^2 ./(4 .*Γ.*t))
    return phi
end

# FD function
function dphidt_fd(phin,u,Γ,delx)
    dphidt = -(u.*(shiftb(phin)-shiftf(phin))./(2 .*delx)) +
    Γ.*((shiftb(phin)-2 .*phin+shiftf(phin))./(delx.^2))
    return dphidt
end

################################################################################
### Constants ###
Γ = 0.1
u = 2.
datasize = 50
numpts = 2^7

## x and t
x = collect(range(0,stop=6,length=numpts))
delx = x[2]-x[1]

tspan = (0.5f0,1.5f0)
t = range(tspan[1],tspan[2],length=datasize)
tmat = repeat(reshape(t, 1, :), outer=(numpts,1))
delt = t[2]-t[1]

################################################################################
### Calculate analytical and FD ###
phi_anal = convert(Array{Float32,2},phi_a(x,tmat,u,Γ)) |> gpu
phi0 = phi_anal[:,1] |> gpu

#=
# Alternate phi0
phi0[:] = zeros(numpts,1)
phi0[21:30] = 2*ones(10)
phi0[31:40] = 1*ones(10)
phi0[41:50] = 0.5*ones(10)
phi0[71:80] = 0.5*ones(10)
=#

phi_fd = similar(phi_anal)
phi_fd[:,1] = phi0
for i = 2:datasize
    phin = phi_fd[:,i-1]
    phi_fd[:,i] = phin + delt.*dphidt_fd(phin,u,Γ,delx)
end

weights_opt = [-u./(2 .*delx), Γ./(delx.^2)]

################################################################################
### Define architecture ###

## Deep
dudt = Chain(x -> reshape(x,numpts,1,1),
             Conv((3,), 1=>2, pad=1, stride=1),
             Conv((3,), 2=>4, pad=1, stride=1),
             Conv((1,), 4=>2, pad=0, stride=1),
             Conv((1,), 2=>1, pad=0, stride=1),
             x -> reshape(x,numpts)) |> gpu
# weights = BSON.load("params.bson")[:params]
# Flux.loadparams!(dudt, weights)

#=
## FD Informed
wfd = reshape([1.f0 0.f0 -1.f0;1.f0 -2.f0 1.f0]',3,1,2)
bfd = [0.0f0,0.0f0]
fdConv(x) = Tracker.data(Conv(param(wfd),param(bfd),pad=1)(x))
dudt = Chain(x -> reshape(x,numpts,1,1),
             Conv((3,), 1=>2, pad=1, stride=1),
             Conv((1,), 2=>1, pad=0, stride=1),
             x -> reshape(x,numpts))
@load "weights-fd.bson" weights
Flux.loadparams!(dudt, weights)
opt_pred = Flux.data(predict_n_ode())
@load "weights-checkpoint.bson" weights
Flux.loadparams!(dudt, weights)
cur_pred = Flux.data(predict_n_ode())
=#

################################################################################
### NODE Setup ###
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)

loss() = sum(abs,phi_anal .- n_ode(phi0))

### Training ###
data = Iterators.repeated((), 10)
opt = ADAM(0.001)

cb = function () #callback function to observe training
  display(loss())
  weights = cpu.(params(dudt))
  println(weights)
  println(" ")
  flush(stdout)
  bson("params.bson", params=weights)
  up = cpu(n_ode(phi0))
  write("upred", up[:])
  #=
  display(plot(x,cur_pred[:,[1,10,25,40,datasize]],
               ylims=(-.5,1.5),layout=(5,1),legend=false,size=(1000,2000)))
  display(plot!(x,phi_anal[:,[1,10,25,40,datasize]],
                ylims=(-.5,1.5),layout=(5,1),legend=false,size=(1000,2000)))
  =#
end

println("Training...")
println(" ")

ps = Flux.params(dudt)
Flux.train!(loss, ps, data, opt, cb = cb)

################################################################################
### Plotting ###
#=
pyplot()
for i = 1:datasize
    #fig = plot(x,phi_anal[:,i],ylims=(-.5,2.5),label="Analytical")
    #fig = plot!(x,phi_fd[:,i],ylims=(-.5,1.5),label="FD")
    fig = plot(x,cur_pred[:,i],ylims=(-.5,2.5),label="NN")
    fig = plot!(x,cur_pred_deep[:,i],ylims=(-.5,2.5),label="deep-NN")
    fig = plot!(x,opt_pred[:,i],ylims=(-.5,2.5),label="opt-NN")
    display(fig)
    sleep(.001)
end
=#
