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
  testmode!(model)
  up = model(batch[1])
  lossM = lossMSE(up, batch[2])
  lossN = lossNorm(up, batch[2])
  return cpu(up), lossM, lossN
end

# Standard
function get_div(u)
  if gpu_num==1
    u = gpu(u)
  end
  mean(div3D(t2b(u))[2:end-1,2:end-1,2:end-1,:,:])
end

# Initialize
function initialize_gpus(ms, batch)
  if gpu_num==1
    batch = gpu(batch)
    ms = gpu(ms)
    CuArrays.reclaim()
  end
  model = gen_mod(ms)
  testmode!(model)
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
function train_b(w, DL, batch_idx, ms, loss, λ)
  consts, data = get_train(DL, batch_idx)
  ps = gen_ps(ms)
  if gpu_num==1
    data = gpu(data)
    ms = gpu(ms)
  end
  model = gen_mod(ms)
  pss = gen_ps(ms)
  l(u0,u) = loss(model(u0),u)*(1+λ)
  lossM = lossMSE(model(data[1]), data[2])
  gs = gradient(pss) do
    l(data...)
  end
  gs = Dict(cpu(p)=>cpu(gs[p]) for p in pss)
  return lossM, ps, gs
end
