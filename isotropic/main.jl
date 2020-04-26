using Distributed
gpu_ids = [0,1,2,3]
gpu_num = length(gpu_ids)
if gpu_num > 1 && nprocs() == 1
  addprocs(gpu_num)
  def_worker = 3
end
################################################################################
### Setup ###
train = true
load_from = "1"
save_to = "1"
train_size = 40
test_size = 1
mbs = 4
epochs = 50
write_pred = true

# Print description
println("### Description ###")
println("z=6, ANODE (4L/64C/+2C), pretrained-joint")
println("")

################################################################################

include("common.jl")

################################################################################

if !isempty(load_from)
  weights = BSON.load("params-"*load_from*".bson")[:params]
  @eval @everywhere Flux.loadparams!(ms, $weights)
end

################################################################################

## Optimiser ##
opt = Flux.Optimiser(ADAM(.0002), WeightDecay(0))

## Parameters ##
ps = gen_ps(ms)
no_node_ps = length([Iterators.flatten(cpu.(params(dldt)))...])
no_tot_ps = length([Iterators.flatten(cpu.(ps))...]) - no_node_ps

## Initial loss ##
test_consts, test_batch = get_test(dataOb)
up, lossM, lossN = initialize_gpus(ms, test_batch)
global best_loss = lossM

## Check no. of best params ##
if length(filter(x->occursin("params-best",x), readdir()))>0
  global nb = parse(Int,filter(x->occursin("params-best",x),
                    readdir())[end][13])
else
  global nb = 0
end

################################################################################

# Print setup
println("### Setup ###")
@show(train)
@show(load_from)
@show(save_to)
@show(train_size)
@show(test_size)
@show(mbs)
@show(epochs)
@show(write_pred)
println("")
println("### Model ###")
@show(encode)
@show(dldt)
@show(decode)
@show(opt)
println("")
println("# of NODE params: ", no_node_ps)
println("# of trainable params: ", no_tot_ps)
println("")

# Print initial loss
println("MSE: ", best_loss)
println("   Norm: ", lossN)
flush(stderr)
flush(stdout)

# Train
if train
  @time train_fn(ms, dataOb, lossMSE, opt, epochs, cb = cb)
end

# Write out
if write_pred
  up, _, _ = get_loss(ms, test_batch)
  utrue = cpu(denormalize(test_consts,test_batch[2]))
  write("utrue", utrue[:])
  upred = cpu(denormalize(test_consts,up))
  write("upred", upred[:])
end

################################################################################
