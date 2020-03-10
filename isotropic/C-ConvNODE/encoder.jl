using DifferentialEquations
using Flux
using Flux: @epochs
using DiffEqFlux
using DiffEqSensitivity
using Statistics
using NPZ
using BSON
using BSON: @save
using BSON: @load
################################################################################
### Setup ###

use_gpu = true
load_data = false
train = true
load_weights = true
load_from = "encoder"
save_to = "encoder"
train_size = 25
test_size = 1
mbs = 1
epochs = 100
write_pred = true
println("resnet, mse")

################################################################################
if use_gpu
  using CUDAnative
  CUDAnative.device!(1)
  using CuArrays
  CuArrays.allowscalar(false)
end
################################################################################
### Load Data ###

include("../data/3D/read_data.jl")
u_data_train = [u_train[:,:,:,:,1:10:end,(i-1)*mbs+1:i*mbs] for i in 1:Int(train_size/mbs)]
u_data_test = u_test[:,:,:,:,1:10:end,:]
tsize=10
data_train = zip(u_data_train,u_data_train)

################################################################################
### Define architecture ###

encode = Chain(x -> reshape(x,size(x,1),size(x,2),size(x,3),size(x,4),
               prod(size(x)[5:6])),
               Conv((4,4,4), 3=>16, pad=1, stride=2, swish),
               Conv((4,4,4), 16=>32, pad=1, stride=2, swish),
               Conv((2,2,2), 32=>16, pad=0, stride=2))

decode = Chain(ConvTranspose((2,2,2), 16=>32, pad=0, stride=2, swish),
               ConvTranspose((4,4,4), 32=>16, pad=1, stride=2, swish),
               ConvTranspose((4,4,4), 16=>3, pad=1, stride=2),
               x -> reshape(x,numpts,numpts,numpts,channels,tsize,
               Int(size(x,5)/tsize)))

if load_weights
  weights = BSON.load("params-"*load_from*".bson")[:params]
  Flux.loadparams!([encode,decode], weights)
end

if use_gpu
  encode = gpu(encode)
  decode = gpu(decode)
end

model = Chain(encode, decode)

################################################################################
### Training ###

lossMAE(up,u) = mean(abs,up.-u)
lossMSE(up,u) = sum(sum(abs2,up.-u, dims=[1,2,3,5,6])./prod(size(u)[[1,2,3,5,6]]))

lossNorm(up,u) = mean(sum(mean(abs2,up.-u, dims=[1,2,3,5]),dims=4)./
                      sum(mean(abs2,u, dims=[1,2,3,5]),dims=4))

loss(u0,u) = lossMAE(model(u0),u)

opt = Flux.Optimiser(ADAM(.00001), WeightDecay(0))

cb = function ()
  up = model(u_data_test)
  loss = lossMSE(up,u_data_test)
  lossN = lossNorm(up,u_data_test)
  println("MSE: ", loss)
  println("   Norm: ", lossN)
  flush(stderr)
  flush(stdout)
  weights = cpu.(params(encode,decode))
  bson("params-"*save_to*".bson", params=weights)
end

@show(use_gpu)
@show(model)
@show(opt)
@show(train_size)
@show(mbs)
@show(epochs)
@show(save_to)
ps = params(encode,decode)
throttlesec = 3

@show(loss(u_data_test,u_data_test))
if train
  @time @epochs epochs Flux.train!(loss, ps, data_train, opt,
                                   cb = Flux.throttle(cb,throttlesec))
end

if write_pred
  utrue = cpu(u_data_test)[:,:,:,:,1:1:end,1]
  upred = cpu(model(u_data_test))[:,:,:,:,1:1:end,1]
  npzwrite("cur_pred.npz", Dict("utrue"=>utrue, "upred"=>upred))
end

################################################################################
### Plotting ###
