@everywhere using Statistics
using BSON
using Flux
import Flux.Optimise: update!
################################################################################
### Data ###

@everywhere tsize = 100

include("/home/vshanka2/research/ml-cfd/ConvNODE/data/read_data.jl")

dataOb = DataLoader(train_size, test_size, mbs, tsize)

dt = dataOb.globDict["dt"]*dataOb.globDict["dtStat"]*(dataOb.stepsτ÷dataOb.tsize)
@eval @everywhere tspan = (0f0, Float32($dt*(tsize-1)))
@everywhere t = range(tspan[1],tspan[2],length=tsize)

nx = dataOb.globDict["nx"]
ny = dataOb.globDict["ny"]
nz = dataOb.globDict["nz"]
channels = dataOb.channels

################################################################################
### Architecture ###

@everywhere include("architecture.jl")

################################################################################
### Serial or parallel ###

if length(gpu_ids)<2
  println("### Serial ###")
  include("serial.jl")
else
  println("### Parallel ###")
  include("parallel.jl")
end
@show(gpu_ids)
println("")

################################################################################
### Training ###

# Train loop
function train_fn(ms, dataOb, loss, opt, epochs; cb = () -> ())
  println("")
  for i = 1:epochs
    println("[Epoch: ", i,"]")
    if dataOb.shuff
      reshuffle!(dataOb)
    end
    for j = 1:Int(dataOb.train_size/dataOb.mbs)

      ps, gs = train_b(workers(), dataOb, j, ms, loss)
      update!(opt, ps, gs)

      if (j-1)%1==0
        cb()
      end
    end
  end
  println("")
  println("Done")
  cb()
end

# Callback
cb = function ()
 up, lossM, lossN = get_loss(ms, test_batch)
 println("MSE: ", lossM)
 println("   Norm: ", lossN)
 weights = cpu.(params(ms))
 bson("params-"*save_to*".bson", params=weights)
 if lossM < best_loss
   global best_loss = lossM
   println("   Best test")
   bson("params-best-"*string(nb+1)*".bson", params=weights)
   upred = cpu(denormalize(test_consts,up))
   write("upred", upred[:])
 end
 flush(stderr)
 flush(stdout)
end

################################################################################
### Functions ###
@everywhere begin

# Loss
lossMAE(up,u) = mean(abs,up.-u)
lossMSE(up,u) = sum(sum(abs2,up.-u, dims=[1,2,3,5,6])./
                prod(size(u)[[1,2,3,5,6]]))

lossNorm(up,u) = mean(reshape(
                 sum(mean(abs2,up.-u, dims=[1,2,3,5]),dims=4),
                 size(u,6))./(2 .*batch_ke(u)))

# Helpers
function snap_ke(u)
  0.5*(var(u[:,:,:,1],corrected=false)
      +var(u[:,:,:,2],corrected=false)
      +var(u[:,:,:,3],corrected=false))
end

function time_ke(u)
  0.5*(var(u[:,:,:,1,:],corrected=false)
      +var(u[:,:,:,2,:],corrected=false)
      +var(u[:,:,:,3,:],corrected=false))
end

function batch_ke(u_batch)
  out = [time_ke(u_batch[:,:,:,:,:,i]) for i in 1:size(u_batch,6)]
  if !isa(u_batch, Array)
    out = gpu(out)
  end
  return eltype(u_batch).(out)
end

function normalize(data_tuple)
  norm_consts = 4 .*sqrt.(2 .*batch_ke(data_tuple[2])./3)
  norm_data_tuple = deepcopy(data_tuple)
  for i in 1:length(norm_consts)
    norm_data_tuple[1][:,:,:,:,i] ./= norm_consts[i]
    norm_data_tuple[2][:,:,:,:,:,i] ./= norm_consts[i]
  end
  return norm_consts, norm_data_tuple
end

function denormalize(norm_consts, norm_data)
  data = deepcopy(norm_data)
  for i in 1:length(norm_consts)
    data[:,:,:,:,:,i] .*= norm_consts[i]
  end
  return data
end

end
