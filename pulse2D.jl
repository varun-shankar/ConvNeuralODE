using DifferentialEquations
using Flux
using DiffEqFlux
using Plots
using NPZ
using BSON: @save
using BSON: @load

################################################################################
### Functions ###

# Analytical soln
function phi_a(x,y,t,ux,uy,Γ)
    phi = (1 ./(4 .*t.+1)).*exp.(.-(x.-ux.*t.-0.5).^2 ./(Γ.*(4 .*t.+1)).-
    (y.-uy.*t.-0.5).^2 ./(Γ.*(4 .*t.+1)))
    return phi
end

################################################################################
### Constants ###
Γ = 0.01
ux = 0.8
uy = 0.8
datasize = 50
numpts = 2^7

## x and t
x = range(0,stop=2,length=numpts)
Y = repeat(x, outer=(1, numpts))
X = Y'

tspan = (0.0f0,1.0f0)
t = range(tspan[1],tspan[2],length=datasize)
tmat = repeat(reshape(t, 1, 1, :), outer=(numpts,numpts,1))

################################################################################
### Calculate analytical and FD ###
phi_anal = convert(Array{Float32,3},phi_a(X,Y,tmat,ux,uy,Γ))
phi0 = reshape(phi_anal[:,:,1],numpts^2,1)

################################################################################
### Define architecture ###

dudt = Chain(x -> reshape(x,numpts,numpts,1,1),
             Conv((5,5), 1=>4, pad=2),
             Conv((1,1), 4=>1, pad=0),
             x -> reshape(x,numpts^2,1))
#@load "weights-2D-checkpoint.bson" weights
#Flux.loadparams!(dudt, weights)
#cur_pred = reshape(Flux.data(predict_n_ode()),numpts,numpts,datasize)

################################################################################
### NODE Setup ###
n_ode(x) = neural_ode(dudt,x,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)

function predict_n_ode()
  n_ode(phi0)
end
loss_n_ode() = sum(abs2,phi_anal .- reshape(predict_n_ode(),numpts,numpts,datasize))

### Training ###
data = Iterators.repeated((), 10000)
opt = ADAM(0.001)


cb = function ()
  display(loss_n_ode())
  weights = Tracker.data.(Flux.params(dudt))
  println(weights)
  println(" ")
  flush(stdout)
  @save "weights-2D-wide-checkpoint.bson" weights
  cur_pred = reshape(Flux.data(predict_n_ode()),numpts,numpts,datasize)
  npzwrite("cur_pred_wide_2D.npz",cur_pred)

  #=
  display(contour(x,x,cur_pred[:,:,datasize],clims=(0.,maximum(phi_anal))))
  display(contour!(x,x,phi_anal[:,:,datasize],clims=(0.,maximum(phi_anal))))
  =#
end

ps = Flux.params(dudt)
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

####################################################c############################
### Plotting ###

#=
pyplot()
for i = 1:datasize
    display(contour(x,x,phi_anal[:,:,i],fill=false,clims=(0.,maximum(phi_anal))))
    display(contour!(x,x,cur_pred[:,:,i],fill=false,clims=(0.,maximum(phi_anal))))
    sleep(.001)
end
=#
