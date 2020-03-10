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
train = false
load_weights = false
load_from = "z24-1"
save_to = "z24-1"
train_size = 1
test_size = 1
epochs = 1e4
println("Testing 2 z=24")

################################################################################
if use_gpu
  using CUDAnative
  CUDAnative.device!(0)
  using CuArrays
  CuArrays.allowscalar(false)
end
################################################################################
### Load Data ###

encoded = Float32.(npzread("../CAE/encoded-1-sample-z24.npz")["encoded"])
if use_gpu
  encoded = gpu(encoded)
end
tsize = size(encoded)[4]
ldims = size(encoded)[1]
datasize = size(encoded)[5]
lchannels = size(encoded)[3]
encoded0 = encoded[:,:,:,1,:]
t = 0.002.*10 .*(0:tsize-1)
tspan = Float32.((t[1],t[end]))
encoded_data = [encoded[:,:,:,:,i:i] for i in 1:datasize]
encoded0_data = [encoded0[:,:,:,i:i] for i in 1:datasize]
encoded_data_train = encoded_data[1:train_size]
encoded0_data_train = encoded0_data[1:train_size]
encoded_data_test = encoded[:,:,:,:,(datasize-test_size+1):datasize]
encoded0_data_test = encoded0[:,:,:,(datasize-test_size+1):datasize]
data_train = zip(encoded0_data_train, encoded_data_train)

################################################################################
### Define architecture ###

## Latent derivative function
dldt = Chain(Conv((3,3), ldims=>16, pad=1, swish),
             Conv((3,3), 16=>32, pad=1, swish),
             Conv((3,3), 32=>16, pad=1, swish),
             Conv((3,3), 16=>ldims, pad=1))

if use_gpu
  dldt = gpu(dldt)
end

n_ode(x) = permutedims(neural_ode(dldt,x,tspan,
                       Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9),[1,2,3,5,4])
model = Chain(x -> n_ode(x))

if load_weights
  @load "params-"*load_from*".bson" weights
  Flux.loadparams!(dldt,weights)
end

################################################################################
### Training ###

loss(l0,l) = mean(abs,model(l0) .- l)
opt = NADAM(0.001)


cb = function ()
  @show(loss(encoded0_data_test,encoded_data_test))
  flush(stdout)
  weights = Tracker.data.(Flux.params(cpu(dldt)))
  @save "params-"*save_to*".bson" weights
end

@show(use_gpu)
@show(dldt)
@show(opt)
@show(train_size)
@show(test_size)
@show(epochs)
@show(save_to)
ps = Flux.params(dldt)
throttlemin = 0
if train
  @time @epochs epochs Flux.train!(loss, ps, data_train, opt,
                                   cb = Flux.throttle(cb,60*throttlemin))
end

################################################################################
