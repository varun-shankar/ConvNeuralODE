using DifferentialEquations
using Flux
using Flux: @epochs
using DiffEqFlux
using Statistics
using NPZ
using BSON: @save
using BSON: @load
################################################################################
### Setup ###

use_gpu = false
plotting = false
train = false
load_weights = true
load_from = "CNODE2"
save_to = "CNODE2"
train_size = 1
test_size = 1
epochs = 2
println("Testing")

################################################################################
if use_gpu
  using CUDAnative
  CUDAnative.device!(0)
  using CuArrays
  CuArrays.allowscalar(false)
end
if plotting
  using Plots
end
################################################################################
### Load Data ###

u_in = Float32.(
       npzread("../data/u_64x64x3x100x50_small.npz")["arr_0"][:,:,:,1:20,1:10])
if use_gpu
  u_in = gpu(u_in)
end
tsize = size(u_in)[4]
numpts = size(u_in)[1]
datasize = size(u_in)[5]
channels = size(u_in)[3]
u0 = u_in[:,:,:,1:5,:]
u_in = permutedims(u_in,[1,2,4,3,5])
u0 = permutedims(u0,[1,2,4,3,5])
t = 0.002.*10 .*(0:tsize-1)
tspan = Float32.((t[1],t[end]))
u_data = [u_in[:,:,:,:,i:i] for i in 1:datasize]
u0_data = [u0[:,:,:,:,i:i] for i in 1:datasize]
u_data_train = u_data[1:train_size]
u0_data_train = u0_data[1:train_size]
u_data_test = u_in[:,:,:,:,(datasize-test_size+1):datasize]
u0_data_test = u0[:,:,:,:,(datasize-test_size+1):datasize]
data_train = zip(u0_data_train, u_data_train)

################################################################################
### Define architecture ###

spFilt = Conv((5,5,1), 3=>3, pad=(2,2,0))
tFilt = Conv((1,1,5), 3=>3, pad=0)
dudt = Chain(Conv((3,3), 6=>32, pad=1, swish),
             Conv((3,3), 32=>6, pad=1))
node(x) = neural_ode_rd(dudt,x,(0f0,0.002f0),Tsit5(),reltol=1e-7,abstol=1e-9)
combine = Chain(Conv((3,3), 6=>32, pad=1, swish),
                Conv((3,3), 32=>3, pad=1))

function predict1step(u)
  ubar = spFilt(u)
  ufluc = u .- ubar
  ubar0 = ubar[:,:,end,:,:]
  ufluc0 = tFilt(ufluc)[:,:,1,:,:]
  uu = cat(ubar0,ufluc0,dims=3)
  unext = combine(node(uu)[:,:,:,:,end])
end

function predict_all(u, horizon)
  upred = u
  for i = 1:horizon
    unext = predict1step(upred[:,:,end-4:end,:,:])
    upred = cat(upred,reshape(unext,64,64,1,3,:),dims=3)
  end
  return upred
end

model(x) = predict_all(x,5)


if use_gpu
  encode = gpu(encode)
  decode = gpu(decode)
  dudt = gpu(dudt)
end

if load_weights
  @load "params-"*load_from*".bson" w1
  Flux.loadparams!(spFilt,w1[1])
  Flux.loadparams!(tFilt,w1[2])
  Flux.loadparams!(dudt,w1[3])
  Flux.loadparams!(combine,w1[4])
end

################################################################################
### Training ###

loss(u0,u) = sum(abs,model(u0) .- u)
opt = NADAM(0.005)


cb = function ()
  @show(loss(u0_data_test,u_data_test))
  flush(stdout)
  w1 = [Tracker.data.(Flux.params(cpu(spFilt))),
        Tracker.data.(Flux.params(cpu(tFilt))),
        Tracker.data.(Flux.params(cpu(dudt))),
        Tracker.data.(Flux.params(cpu(combine)))]
  @save "params-"*save_to*".bson" w1
end

@show(use_gpu)
@show(model)
@show(dudt)
@show(opt)
@show(train_size)
@show(test_size)
@show(epochs)
@show(save_to)
ps = Flux.params(spFilt,tFilt,dudt,combine)
throttlemin = 0
if train
  @time @epochs epochs Flux.train!(loss, ps, data_train, opt,
                                   cb = Flux.throttle(cb,60*throttlemin))
end

################################################################################
