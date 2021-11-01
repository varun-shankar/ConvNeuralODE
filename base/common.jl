@everywhere using Statistics
using BSON
using DelimitedFiles
using Flux
import Flux.Optimise: update!
using BayesianOptimization, GaussianProcesses, Distributions
using Logging, TensorBoardLogger

################################################################################
### Training ###

function initialize_training(dataOb, ms)
  ## Parameters ##
  ps = gen_ps(ms)
  no_node_ps = length([Iterators.flatten(cpu.(Flux.params(ms[2])))...])
  no_tot_ps = length([Iterators.flatten(cpu.(ps))...])

  ## Initial loss ##
  global test_consts, test_batch = get_test(dataOb)
  up, lossM, lossN = initialize_gpus(ms, test_batch)
  global best_loss = lossN
  weights = cpu.(Flux.params(ms))
  BSON.@save "state-best-"*string(round(best_loss,digits=4))*".bson" opt weights
  # bson("params-best-"*string(round(best_loss,digits=4))*".bson", params=weights)

  # Print setup
  println("### Setup ###")
  @show(train)
  @show(load_from)
  @show(save_to)
  @show(train_size)
  @show(test_size)
  @show(mbs)
  @show(epochs)
  @show(write_out)
  @show(tsize)
  @show(decay_data)
  println("")
  println("### Model ###")
  @show(ms[1])
  @show(ms[2].model)
  @show(ms[3])
  println("")
  println("# of NODE params: ", no_node_ps)
  println("# of trainable params: ", no_tot_ps)
  println("")

  # Print initial loss
  println("MSE: ", lossM)
  println("   Norm: ", lossN)
  flush(stderr)
  flush(stdout)
end

# Train loop
global λ = 0f0 # initial value
# λ_up_hist_length = 10
# λ_up_tol = 0.001
# λ_up_factor = 1.0
# min_ep = 20
function train_fn(ms, dataOb, loss, opt, epochs; testit=1, cb=()->())
  global loss_h = []
  println("")
  for i = 1:epochs
    println("[Epoch: ", i,"]")

    if i>1 && (i-1)%10==0
      println("Writing prediction")
      write_pred(ms)
      weights = cpu.(Flux.params(ms))
      BSON.@save "state-"*save_to*".backup.bson" opt weights
    end

    if decay_data && dataOb.train_size>dataOb.mbs && i>1 && (i-1)%2==0
      dataOb.train_size -= dataOb.mbs
      dataOb = DataLoader(dataOb.train_size, test_size,
                          mbs, tsize, data_dir, dataOb.stepsτ, dataOb.shuff)
    end
    println("Train size: ", dataOb.train_size)
    if dataOb.shuff
      println("Shuffling")
      reshuffle!(dataOb)
    end

    for j = 1:Int(dataOb.train_size/dataOb.mbs)
      # λ update
      # if i > min_ep
      #   λ_crit = mean(diff(loss_h[end-λ_up_hist_length+1:end]))
      #   println(λ_crit)
      #   if λ_crit >= λ_up_tol#(i-1)%2==0 && j==1#
      #     global λ *= λ_up_factor
      #     println("λ: ", round(λ,digits=4))
      #   end
      # end

      # Dynamic LR
      # if i == 201; opt.eta /= 2;elseif i == 301; opt.eta = .0001;end

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
 weights = cpu.(Flux.params(ms))
 BSON.@save "state-"*save_to*".bson" opt weights
 # bson("params-"*save_to*".bson", params=weights)

 test_div = get_div(denormalize(test_consts,up))
 train_loss = loss_h[end]
 test_loss = lossM
 test_ke = snap_ke(up[:,:,:,:,end,1])
 true_ke = snap_ke(test_batch[2][:,:,:,:,end,1])

 cost = 0.4*train_loss*10 + 0.2*test_loss*20 +
        0.1*test_div*1000 + 0.3*abs2(test_ke-true_ke)*100
 println("     KE: ", test_ke)
 push!(loss_save, Float32.([lossN, loss_h[end], lossM, test_div, test_ke, cost]))
 writedlm("loss-"*save_to, loss_save)
 if lossN < best_loss
   rm("state-best-"*string(round(best_loss,digits=4))*".bson")
   # rm("params-best-"*string(round(best_loss,digits=4))*".bson")
   global best_loss = lossN
   println("   Best test")
   BSON.@save "state-best-"*string(round(best_loss,digits=4))*".bson" opt weights
   # bson("params-best-"*string(round(best_loss,digits=4))*".bson", params=weights)
   upred = cpu(denormalize(test_consts,up))
   write("upred", upred[:])
 end
 flush(stderr)
 flush(stdout)

 # Logging
 # kes = zeros(Float32,tsize)
 # for i=1:1#tsize
 #   kes[i] = snap_ke(up[:,:,:,:,i,1])
 # end
 # with_logger(logger) do
 #    param_dict = Dict{String, Any}()
 #    fill_param_dict!(param_dict, Chain(ms[1],ms[2].re(ms[2].p),ms[3]), "")
 #    @info "model" params=param_dict cost=cost log_step_increment=0
 #    @info "train" loss=train_loss log_step_increment=0
 #    @info "test" loss=test_loss div=test_div tke=kes
 #  end

 return up, lossM, lossN
end

function fill_param_dict!(dict, m, prefix)
if m isa Chain
  for (i, layer) in enumerate(m.layers)
    fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
  end
else
  for fieldname in fieldnames(typeof(m))
    val = getfield(m, fieldname)
    if val isa AbstractArray
      val = vec(val)
      dict[prefix*string(fieldname)] = val
    end
  end
 end
end

function train_wrapper(x)
  Nh, Nz, Na, Nk, Nw, Nl = x
  oNs = (Nh, Nz, Na, Nk, Nw, Nl)
  Nh, Nz, Na, Nw, Nl = round.(Int, [Nh, Nz, Na, Nw, Nl])
  Nk = round_odd(Nk)
  println("")
  println(Nh, "\t", Nz, "\t", Na, "\t", Nk, "\t", Nw, "\t", Nl)

  dataOb = DataLoader(train_size, test_size,
                      mbs, tsize, data_dir, false)

  @everywhere @eval begin
  ms = arch($Nh, $Nz, $Na, $Nk, $Nw, $Nl)
  gen_mod(ms) = Chain(ms[1],
                      x -> add_p(x, $Na),
                      ms[2],
                      x -> (isa(x,Array)) ? cpu(x) : gpu(x),
                      x -> permutedims(x,[1,2,3,4,6,5]),
                      x -> rm_p(x, $Na),
                      t2b,
                      ms[3], specProj3Db,
                      b2t)
  end
  encode, dldt, decode = ms

  initialize_training(dataOb, ms)

  up, lossM, lossN = train_fn(ms, dataOb, loss, opt, epochs,
                                    testit = 2, cb = cb)
  test_div = get_div(denormalize(test_consts,up))
  train_loss = loss_h[end]
  test_loss = lossM
  test_ke = snap_ke(up[:,:,:,:,end,1])
  true_ke = snap_ke(test_batch[2][:,:,:,:,end,1])
  @show(train_loss)
  @show(test_loss)
  @show(test_div)
  @show(test_ke)
  @show(true_ke)
  @show(abs2(test_ke-true_ke))
  cost = 0.4*train_loss*10 + 0.2*test_loss*20 +
         0.1*test_div*1000 + 0.3*abs2(test_ke-true_ke)*100

  println("")
  open("search-"*save_to, "a") do io
    writedlm(io, zip(oNs..., cost))
  end
  @show(cost)
  return cost
end

function write_pred(ms, wtrue=false; idx=1)
  up, _, _ = get_loss(ms, test_batch)
  if wtrue
    utrue = cpu(denormalize(test_consts,test_batch[2]))[:,:,:,:,:,idx]
    write("utrue", utrue[:])
  end
  upred = cpu(denormalize(test_consts,up))[:,:,:,:,:,idx]
  write("upred", upred[:])
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

lossDiv(up,u) = lossMSE(up,u) +
                50 .*sum(abs,div3D(t2b(up)))./prod(size(u)[[1,2,3,5,6]])

function lossSpec(up,u)
  out = 0
    for i in 1:size(u,6)
      U1 = rfft(u[:,:,:,1,end,i]); U2 = rfft(u[:,:,:,2,end,i]); U3 = rfft(u[:,:,:,3,end,i])
      UP1 = rfft(up[:,:,:,1,end,i]); UP2 = rfft(up[:,:,:,2,end,i]); UP3 = rfft(up[:,:,:,3,end,i])
      out = out + mean(abs2,U1.-UP1) + mean(abs2,U2.-UP2) + mean(abs2,U3.-UP3)
    end
    return out
end

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

# function all_ke(u)
#   0.5*(var(u[:,:,:,1,:,:],corrected=false)
#       +var(u[:,:,:,2,:,:],corrected=false)
#       +var(u[:,:,:,3,:,:],corrected=false))
# end

function all_ke(u)
  # mapslices(snap_ke, u, dims=[1,2,3,4])
  out = Zygote.Buffer(u, size(u,5), size(u,6))
  for i = 1:size(u,5)
    for j = 1:size(u,6)
      out[i:i,j:j] = snap_ke(u[:,:,:,:,i,j])
    end
  end
  return copy(out)
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

round_odd(x) = Int(2*floor(x/2)+1)
