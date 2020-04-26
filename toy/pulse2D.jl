using DifferentialEquations
using Flux
using Flux: @epochs
using DiffEqFlux
using NPZ
using BSON: @save
using BSON: @load
using Statistics
################################################################################
### Setup ###

use_gpu = false
plotting = false
train = true
load_weights = false
load_from = "pulse2D-1"
save_to = "pulse2D-1"
epochs = 2000
println("Pulse 2D")

################################################################################
if use_gpu
  using CuArrays
  CuArrays.allowscalar(false)
end
if plotting
  using Plots
end
################################################################################
### Functions ###

# Analytical soln
function phi_a(x,y,t,ux,uy,Γ)
  phi = (1 ./(4 .*t.+1)).*exp.(.-(x.-ux.*t.-0.5).^2 ./(Γ.*(4 .*t.+1)).-
  (y.-uy.*t.-0.5).^2 ./(Γ.*(4 .*t.+1)))
  return phi
end

function bi_G(x,y,μ,ρ,σxy)
  out = (1 ./(2 .*π.*σxy[1].*σxy[2].*sqrt.(1 .-ρ.^2))).*
        exp.(-1 ./(2 .*(1 .-ρ.^2)).*
        ((x.-μ[1]).^2 ./σxy[1].^2 .+ (y.-μ[2]).^2 ./σxy[2].^2) .-
        2 .*ρ.*(x.-μ[1]).*(y.-μ[2])./(σxy[1].*σxy[2]))
end
################################################################################
### Constants ###
Γ = 0.01
ux = 0.8
uy = 0.8
tsize = 50
numpts = 2^7

## x and t
x = range(0,stop=2,length=numpts)
Y = repeat(x, outer=(1, numpts))
X = Y'
dx = x[2]-x[1]

tspan = (0.0f0,1.0f0)
t = range(tspan[1],tspan[2],length=tsize)
tmat = repeat(reshape(t, 1, 1, :), outer=(numpts,numpts,1))

################################################################################
### Calculate analytical ###

phi_anal = reshape(Float32.(phi_a(X,Y,tmat,ux,uy,Γ)),numpts,numpts,1,tsize,1)
if use_gpu
  phi_anal = gpu(phi_anal)
end
phi0 = phi_anal[:,:,:,1,:]
# Test phi0
# phi0[:,:,1,1] = bi_G(X,Y,[.7;.5],.1,[.1;.1]).+
#                 bi_G(X,Y,[.45;.5],.12,[.3;.15]).+
#                 bi_G(X,Y,[.9;.6],0,[.2;.3]).+
#                 bi_G(X,Y,[.6;.65],.5,[.15;.1])
# phi0[:,:,1,1] = phi0[:,:,1,1]./maximum(phi0[:,:,1,1])
data = Iterators.repeated((),1)

################################################################################
### Define architecture ###

# Model
dudt = Chain(Conv((5,5), 1=>4, pad=2), swish),
             Conv((3,3), 4=>4, pad=1, swish),
             Conv((1,1), 4=>1, pad=0))

# True FD
dudt_true = Chain(Conv((3,3), 1=>1, pad=1))
wt = Float32.([0                 -uy/(2*dx)+Γ/dx^2 0               ;
               -ux/(2*dx)+Γ/dx^2 -4*Γ/dx^2         ux/(2*dx)+Γ/dx^2;
               0                 uy/(2*dx)+Γ/dx^2  0               ])
wt = reshape(wt,3,3,1,1)
bt = Float32.([0])
weights_true = [wt, bt]
Flux.loadparams!(dudt_true,weights_true)
dudt_true = Flux.mapleaves(Flux.data, dudt_true)


if use_gpu
 dudt = gpu(dudt)
 dudt_true = gpu(dudt_true)
end

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
n_ode_true = NeuralODE(dudt_true,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)

n_ode(x) = permutedims(neural_ode(dudt,x,tspan,
                       Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9),[1,2,3,5,4])

n_ode_true(x) = permutedims(neural_ode(dudt_true,x,tspan,
                       Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9),[1,2,3,5,4])

model = Chain(x -> n_ode(x))
model_true = Chain(x -> n_ode_true(x))

if load_weights
  @load "params-"*load_from*".bson" weights
  Flux.loadparams!(dudt,weights)
end

################################################################################
### Training ###

loss() = mean(abs,model(phi0) .- phi_anal)
loss_true() = mean(abs,model_true(phi0) .- phi_anal)
opt = ADAM(0.001)


cb = function ()
  @show(loss())
  flush(stdout)
  weights = Tracker.data.(Flux.params(cpu(dudt)))
  @save "params-"*save_to*".bson" weights
end

@show(use_gpu)
@show(dudt)
@show(opt)
@show(epochs)
@show(save_to)
ps = Flux.params(dudt)
throttlemin = 0
cb()
if train
  @time @epochs epochs Flux.train!(loss, ps, data, opt, cb = cb)
end

################################################################################
### Plotting ###

if plotting
  cur_pred = model(phi0).data
  true_pred = model_true(phi0)
  for i = 1:tsize
      display(contour(x,x,true_pred[:,:,1,i,1],
      fill=false,clims=(0.,maximum(true_pred))))
      display(contour!(x,x,cur_pred[:,:,1,i,1],
      fill=false,clims=(0.,maximum(true_pred))))
      sleep(.001)
  end
end
