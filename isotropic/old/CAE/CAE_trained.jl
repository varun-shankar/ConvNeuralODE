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
save_data = true
load_from = "cae-z12"
println("CAE z=24")

################################################################################
if use_gpu
  using CUDAnative
  CUDAnative.device!(1)
  using CuArrays
  CuArrays.allowscalar(false)
end
################################################################################
### Load Data ###

u_in = Float32.(
        npzread("../data/u_64x64x3x100x50_small.npz")["arr_0"][:,:,:,:,1:1])
if use_gpu
  u_in = gpu(u_in)
end
tsize = size(u_in)[4]
channels = size(u_in)[3]
numpts = size(u_in)[1]
datasize = size(u_in)[5]

u_in_batched = reshape(u_in,numpts,numpts,channels,:)

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

@load load_from*".bson" weights
Flux.loadparams!(model, weights)

encode = Flux.mapleaves(Flux.data, encode)
decode = Flux.mapleaves(Flux.data, decode)
model = Flux.mapleaves(Flux.data, model)

################################################################################
### Apply ###

loss(u) = mean(abs,(model(u).-u))

function zcalc()
  u_test = reshape(u_in_batched[:,:,:,rand(1:size(u_in_batched,4))],
                   numpts,numpts,channels,1)
  z = prod(size(u_test))/prod(size(encode(u_test)))
  @show(z)
end

@show(model)

zcalc()

print("Encoding")
@time u_encoded_batched = encode(u_in_batched)
u_encoded_unbatched = reshape(u_encoded_batched,
                      size(u_encoded_batched,1),size(u_encoded_batched,2),
                      size(u_encoded_batched,3),tsize,datasize)

print("Decoding")
@time u_out_batched = decode(u_encoded_batched)
u_out = reshape(u_out_batched,
        size(u_out_batched,1),size(u_out_batched,2),
        size(u_out_batched,3),tsize,datasize)

loss_total = mean(abs,(u_in.-u_out))
@show(loss_total)

################################################################################
### Saving ###

if save_data
  npzwrite("encoded-1-sample-z12.npz",
           Dict("encoded" => cpu(u_encoded_unbatched)))
end
