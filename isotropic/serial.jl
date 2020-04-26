using Flux
################################################################################
### Initialize the GPU ###
if gpu_num==1
  using CuArrays
  using CUDAnative
  CuArrays.allowscalar(false)
  CUDAnative.device!(gpu_ids[1])
end
################################################################################
### Get loss ###

# Standard
function get_loss(ms, batch)
  if gpu_num==1
    batch = gpu(batch)
    ms = gpu(ms)
  end
  model = gen_mod(ms)
  up = model(batch[1])
  lossM = lossMSE(up, batch[2])
  lossN = lossNorm(up, batch[2])
  return cpu(up), lossM, lossN
end

# Initialize
function initialize_gpus(ms, batch)
  if gpu_num==1
    batch = gpu(batch)
    ms = gpu(ms)
    CuArrays.reclaim()
  end
  model = gen_mod(ms)
  up = model(batch[1])
  if gpu_num==1
    CuArrays.reclaim()
  end
  lossM = lossMSE(up, batch[2])
  lossN = lossNorm(up, batch[2])
  return cpu(up), lossM, lossN
end

################################################################################
### Training ###

# Batch backprop
function train_b(w, DL, batch_idx, ms, loss)
  consts, data = get_train(DL, batch_idx)
  ps = gen_ps(ms)
  if gpu_num==1
    data = gpu(data)
    ms = gpu(ms)
  end
  model = gen_mod(ms)
  pss = gen_ps(ms)
  l(u0,u) = loss(model(u0),u)
  gs = gradient(pss) do
    l(data...)
  end
  gs = Dict(cpu(p)=>cpu(gs[p]) for p in pss)
  return ps, gs
end
