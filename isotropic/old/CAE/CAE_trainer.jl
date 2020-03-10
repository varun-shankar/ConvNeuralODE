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

use_gpu = true
train = true
load_weights = true
load_from = "cae-z12"
save_to = "cae-z12"
epochs = 1e2
println("CAE z12")

################################################################################
if use_gpu
  using CUDAnative
  CUDAnative.device!(0)
  using CuArrays
  CuArrays.allowscalar(false)
end
################################################################################
### Load Data ###

u_in = Float32.(
       npzread("../data/u_64x64x3x400x100_final.npz")["arr_0"][:,:,:,1:50,:])
if use_gpu
  u_in = gpu(u_in)
end
tsize = size(u_in)[4]
channels = size(u_in)[3]
numpts = size(u_in)[1]
datasize = size(u_in)[5]
u_data = [u_in[:,:,:,i,:] for i in 1:tsize]
u_data_train = u_data[1:end-1]
u_data_test = u_data[end]
data_train = zip(u_data_train)

################################################################################
### Define architecture ###

encode = Chain(Conv((4,4), 3=>16, pad=1, stride=2, swish),
               Conv((4,4), 16=>32, pad=1, stride=2, swish),
               Conv((2,2), 32=>16, pad=0, stride=2))

decode = Chain(ConvTranspose((2,2), 16=>32, pad=0, stride=2, swish),
               ConvTranspose((4,4), 32=>16, pad=1, stride=2, swish),
               ConvTranspose((4,4), 16=>3, pad=1, stride=2))

if use_gpu
 encode = gpu(encode)
 decode = gpu(decode)
end

model = Chain(encode, decode)
if load_weights
  @load load_from*".bson" weights
  Flux.loadparams!(model, weights)
end

################################################################################
### Training ###

loss(u) = sum(abs,(model(u).-u))
opt = NADAM(.001)

cb = function ()
  mean_loss_test = loss(u_data_test)/prod(size(u_data_test))
  @show(mean_loss_test)
  flush(stdout)
  weights = Tracker.data.(Flux.params(cpu(model)))
  @save save_to*".bson" weights
end

z = prod(size(u_data_test))/prod(size(encode(u_data_test)))

@show(use_gpu)
@show(model)
@show(z)
@show(opt)
@show(epochs)
@show(save_to)
ps = Flux.params(model)
throttlemin = 1
cb()
if train
  @time @epochs epochs Flux.train!(loss, ps, data_train, opt,
                                   cb = Flux.throttle(cb,60*throttlemin))
end

################################################################################
