# Directories
base_dir = "base"
data_dir = "/ocean/projects/asc200013p/vshankar/3D_LANL"
arch_dir = "."
log_dir = "log"

# Print description
println("### Description ###")
println("")
println("")
flush(stdout)
################################################################################
### Setup ###

gpu_ids = [0]
def_worker = 3

train = true
load_from = ""
save_to = "1"
train_size = 8
test_size = 2
mbs = 8
epochs = 300
write_out = true
tpdp = 100
tskip = 1
decay_data = false
η = 0.001

################################################################################
### Processes ###

include(base_dir*"/distributed.jl")
gpu_num, pds = distributed_setup(gpu_ids)
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
### Data ###

@eval @everywhere include($base_dir*"/read_data.jl")

tsize = Int(tpdp/tskip)
dataOb = DataLoader(train_size, test_size,
                    mbs, tsize, data_dir, tpdp, true)

dt = dataOb.globDict["dt"]*dataOb.globDict["dtStat"]*(dataOb.stepsτ÷dataOb.tsize)
@eval @everywhere tsize = $tsize
@eval @everywhere tspan = (0f0, Float32($dt*(tsize-1)))
@everywhere t = range(tspan[1],tspan[2],length=tsize)
@everywhere ts = range(tspan[1],tspan[2],length=tsize*4)


################################################################################
### Architecture ###

@eval @everywhere include($base_dir*"/gradient.jl")
@eval @everywhere include($base_dir*"/NODE.jl")
@eval @everywhere include($arch_dir*"/architecture.jl")

################################################################################
### Optimiser and Loss ###

opt = ADAM(η)
@everywhere loss(up,u) = lossMSE(up,u) 

################################################################################
### Load state ###

loss_save = []
load_opt, load_save = true, false
if !isempty(load_from)
 weights = cpu.(Flux.params(ms))
 BSON.@load "state-"*load_from*".bson" weights
 @eval @everywhere Flux.loadparams!(ms, $weights)
 if load_opt
   println("Loading opt...")
   BSON.@load "state-"*load_from*".bson" opt weights
   @show(opt.eta)
   for (p1, p2) in zip(weights, Flux.params(ms))
     if haskey(opt.state, p1)
       opt.state[p2] = opt.state[p1]
       println("Found param")
     end
     delete!(opt.state, p1)
   end
 end
 if load_save
   loss_save = readdlm("loss-"*load_from)
   loss_save = [Float32.(loss_save[i,:]) for i in 1:size(loss_save,1)]
 end
end
opt.eta = η
@show(opt.eta)

################################################################################
### Training ###

include(base_dir*"/common.jl")
# logger = TBLogger(log_dir, tb_overwrite)

initialize_training(dataOb, ms)

# Train
if train
  @time up, lossM, lossN = train_fn(ms, dataOb, loss, opt, epochs,
                                    testit = 2, cb = cb)
end

# Write out
if write_out
  write_pred(ms, true)
end

################################################################################
