gpu_ids = [1,2]
def_worker = 3
################################################################################
### Setup ###

train = true
load_from = "sw-curl-resf"
save_to = "sw-curl2-resf"
save_to_loss = "sw-curl2-resf"
train_size = 4
test_size = 1
mbs = 2
epochs = 500
write_pred = true
tsize = 100

# Directories
base_dir = "../base"
data_dir = "/data/vshankar/ConvNODE/3D_LANL"
arch_dir = "."

# Print description
println("### Description ###")
println("z=6, ANODE (4L/64C/+2C), joint, Res")
println("")
flush(stdout)

################################################################################
### Initialize ###

include(base_dir*"/distributed.jl")
gpu_num, pds = distributed_setup(gpu_ids)
include(base_dir*"/common.jl")

################################################################################
### Load weights ###

if !isempty(load_from)
  weights = BSON.load("params-"*load_from*".bson")[:params]
  @eval @everywhere Flux.loadparams!(ms, $weights)
end

################################################################################

## Optimiser ##
opt = ADAM(.0005)

## Parameters ##
ps = gen_ps(ms)
no_node_ps = length([Iterators.flatten(cpu.(params(dldt)))...])
no_tot_ps = length([Iterators.flatten(cpu.(ps))...])

## Initial loss ##
test_consts, test_batch = get_test(dataOb)
up, lossM, lossN = initialize_gpus(ms, test_batch)
global best_loss = lossN
weights = cpu.(params(ms))
bson("params-best-"*string(round(best_loss,digits=4))*".bson", params=weights)

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
@show(tsize)
println("")
println("### Model ###")
@show(encode)
@show(dldt)
@show(decode)
# @show(opt)
println("")
println("# of NODE params: ", no_node_ps)
println("# of trainable params: ", no_tot_ps)
println("")

# Print initial loss
println("MSE: ", lossM)
println("   Norm: ", lossN)
flush(stderr)
flush(stdout)

# Train
if train
  @time up, lossM, lossN = train_fn(ms, dataOb, lossMSE, opt, epochs,
                                    testit = 2, cb = cb)
end

# Write out
if write_pred
  utrue = cpu(denormalize(test_consts,test_batch[2]))
  write("utrue", utrue[:])
  upred = cpu(denormalize(test_consts,up))
  write("upred", upred[:])
end

################################################################################
