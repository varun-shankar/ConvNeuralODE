@everywhere using Statistics
using BSON
using DelimitedFiles
using Flux
import Flux.Optimise: update!
################################################################################
### Data ###

include(base_dir*"/read_data.jl")

dataOb = DataLoader(train_size, test_size, mbs, tsize, data_dir)

dt = dataOb.globDict["dt"]*dataOb.globDict["dtStat"]*(dataOb.stepsτ÷dataOb.tsize)
@eval @everywhere tsize = $tsize
@eval @everywhere tspan = (0f0, Float32($dt*(tsize-1)))
@everywhere t = range(tspan[1],tspan[2],length=tsize)

nx = dataOb.globDict["nx"]
ny = dataOb.globDict["ny"]
nz = dataOb.globDict["nz"]
channels = dataOb.channels

################################################################################
### Architecture ###

@eval @everywhere include($arch_dir*"/architecture.jl")

################################################################################
### Serial or parallel ###

if gpu_num<2
  println("### Serial ###")
  include(base_dir*"/serial.jl")
else
  println("### Parallel ###")
  include(base_dir*"/parallel.jl")
end
print("gpu_ids: ")
display(gpu_ids)
println("")
flush(stderr)
flush(stdout)

################################################################################
### Training ###

# Train loop
loss_h = []
loss_save = []
global λ = .001f0 # initial value
λ_up_hist_length = 5
λ_up_tol = 0.001
λ_up_factor = 1.1
min_ep = 2
function train_fn(ms, dataOb, loss, opt, epochs; testit=1, cb=()->())
  println("")
  for i = 1:epochs
    println("[Epoch: ", i,"]")
    if dataOb.shuff
      reshuffle!(dataOb)
    end
    for j = 1:Int(dataOb.train_size/dataOb.mbs)
      # λ update
      if i > min_ep
        λ_crit = mean(diff(loss_h[end-λ_up_hist_length+1:end]))
        println(λ_crit)
        if λ_crit >= λ_up_tol#(i-1)%2==0 && j==1#
          global λ *= λ_up_factor
          println("λ: ", round(λ,digits=4))
        end
      end

      l, ps, gs = train_b(workers(), dataOb, j, ms, loss, λ)
      update!(opt, ps, gs)
      push!(loss_h, l)

      if (j-1)%testit==0
        cb()
      end

    end
  end
  println("")
  println("Done")
  println("")
  cb()
end

# Callback
cb = function ()
 up, lossM, lossN = get_loss(ms, test_batch)
 println("MSE: ", lossM)
 println("   Norm: ", lossN)
 weights = cpu.(params(ms))
 bson("params-"*save_to*".bson", params=weights)
 push!(loss_save, [λ, loss_h[end], lossM, lossN])
 writedlm("loss-"*save_to_loss, loss_save)
 if lossN < best_loss
   rm("params-best-"*string(round(best_loss,digits=4))*".bson")
   global best_loss = lossN
   println("   Best test")
   bson("params-best-"*string(round(best_loss,digits=4))*".bson", params=weights)
   upred = cpu(denormalize(test_consts,up))
   write("upred", upred[:])
 end
 flush(stderr)
 flush(stdout)
 return up, lossM, lossN
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
