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
#import Zygote: Params, gradient
################################################################################
### Setup ###

use_gpu = true
train = true
load_weights = true
load_from = "1"
save_to = "1"
train_size = 48
test_size = 1
mbs = 4
epochs = 50
write_pred = true
println("conv 2, pretrained encoder")

################################################################################
if use_gpu
  using CUDAnative
  CUDAnative.device!(1)
  using CuArrays
  CuArrays.allowscalar(false)
end
################################################################################
### Data ###

numpts = 64
channels = 3
tsize = 100

include("/data/vshankar/ConvNODE/3D/read_data.jl")

dt = .02
tspan = (0f0, Float32(dt*(tsize-1)))
t = range(tspan[1],tspan[2],length=tsize)

dataOb = DataLoader(train_size, test_size, mbs)

################################################################################
### Define architecture ###

encode = Chain(Conv((4,4,4), 3=>16, pad=1, stride=2, swish),
               Conv((4,4,4), 16=>32, pad=1, stride=2, swish),
               Conv((2,2,2), 32=>16, pad=0, stride=2))

decode = Chain(x -> reshape(x,size(x,1),size(x,2),size(x,3),size(x,4),
               prod(size(x)[5:6])),
               ConvTranspose((2,2,2), 16=>32, pad=0, stride=2, swish),
               ConvTranspose((4,4,4), 32=>16, pad=1, stride=2, swish),
               ConvTranspose((4,4,4), 16=>3, pad=1, stride=2),
               x -> reshape(x,numpts,numpts,numpts,channels,tsize,
               Int(size(x,5)/tsize)))

## dldt
dldt = Chain(Conv((5,5,5), 16=>32, pad=2, swish),
             Conv((3,3,3), 32=>64, pad=1, swish),
             Conv((3,3,3), 64=>64, pad=1, swish),
             Conv((3,3,3), 64=>64, pad=1, swish),
             Conv((3,3,3), 64=>64, pad=1, swish),
             Conv((3,3,3), 64=>64, pad=1, swish),
             Conv((3,3,3), 64=>64, pad=1, swish),
             Conv((3,3,3), 64=>64, pad=1, swish),
             Conv((3,3,3), 64=>32, pad=1, swish),
             Conv((3,3,3), 32=>16, pad=1))

if use_gpu
  encode = gpu(encode)
  decode = gpu(decode)
  dldt = gpu(dldt)
end

n_ode = NeuralODE(dldt,tspan,#sensealg=InterpolatingAdjoint(),
                 Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)

if load_weights
  weights = BSON.load("params-"*load_from*".bson")[:params]
  Flux.loadparams!([encode,n_ode,decode], weights)
end
weightsE = BSON.load("params-encoder.bson")[:params]
Flux.loadparams!([encode,decode], weightsE)

if use_gpu
 node(x) = CuArray(n_ode(x))
else
 node(x) = Array(n_ode(x))
end

model = Chain(encode, x -> permutedims(node(x),[1,2,3,4,6,5]), decode)

################################################################################
### Training ###

# Loss
lossMAE(up,u) = mean(abs,up.-u)
lossMSE(up,u) = sum(sum(abs2,up.-u, dims=[1,2,3,5,6])./prod(size(u)[[1,2,3,5,6]]))

lossNorm(up,u) = mean(sum(mean(abs2,up.-u, dims=[1,2,3,5]),dims=4)./
                      sum(mean(abs2,u, dims=[1,2,3,5]),dims=4))

loss(u0,u) = lossMSE(model(u0),u)

# Optimiser
opt = Flux.Optimiser(ADAM(.0002), WeightDecay(0))

# Train function
function train_fn(epochs, loss, ps, dataOb, opt; cb = () -> ())
  println("")
  for i = 1:epochs
    println("[Epoch: ", i,"]")
    reshuffle!(dataOb)
    for j = 1:Int(dataOb.train_size/dataOb.mbs)
      train_batch = get_train(dataOb, j, use_gpu)
      Flux.train!(loss, ps, [train_batch], opt)
      cb()
    end
  end
end

cb = function ()
  test_batch = get_test(dataOb, use_gpu)
  up = model(test_batch[1])
  lossM = lossMSE(up,test_batch[2])
  lossN = lossNorm(up,test_batch[2])
  println("MSE: ", lossM)
  println("   Norm: ", lossN)
  flush(stderr)
  flush(stdout)
  weights = cpu.(params(encode,n_ode,decode))
  bson("params-"*save_to*".bson", params=weights)
  if lossM < best_loss
    global best_loss = lossM
    println("   Best test")
    bson("params-best-"*string(nb+1)*".bson", params=weights)
    utrue = cpu(test_batch[2])[:,:,:,:,1:20:end,1]
    upred = cpu(up)[:,:,:,:,1:20:end,1]
    npzwrite("cur_pred.npz", Dict("utrue"=>utrue, "upred"=>upred))
  end
end

@show(use_gpu)
@show(model)
@show(dldt)
@show(opt)
@show(train_size)
@show(mbs)
@show(epochs)
@show(save_to)
ps = params(n_ode)
throttlesec = 3

test_batch = get_test(dataOb, use_gpu)
up = model(test_batch[1])
global best_loss = lossMSE(up,test_batch[2])
lossN = lossNorm(up,test_batch[2])
test_batch = nothing
println("MSE: ", best_loss)
println("   Norm: ", lossN)
flush(stderr)
flush(stdout)
global nb = length(filter(x->occursin("params-best",x), readdir()))

if train
  @time train_fn(epochs, loss, ps, dataOb, opt, cb = cb)
end

if write_pred
  test_batch = get_test(dataOb, use_gpu)
  utrue = cpu(test_batch[2])[:,:,:,:,1:5:end,1]
  upred = cpu(up)[:,:,:,:,1:5:end,1]
  npzwrite("cur_pred.npz", Dict("utrue"=>utrue, "upred"=>upred))
end

################################################################################
