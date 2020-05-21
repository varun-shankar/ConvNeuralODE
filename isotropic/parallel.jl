using Flux
################################################################################
### Initialize the GPUs ###
asyncmap(pds) do (p, d)
   remotecall_wait(p) do
       h = gethostname()
       @eval using CuArrays
       @eval using CUDAnative
       CuArrays.allowscalar(false)
       CUDAnative.device!(d)
       println("Worker $p uses GPU:$d on $h")
   end
end
################################################################################
### Get loss ###

# From specific worker
function get_loss(ms, batch, p=def_worker)
  remotecall_fetch(p) do
    batch = gpu(batch)
    ms = gpu(ms)
    model = gen_mod(ms)
    CuArrays.reclaim() #
    up = model(batch[1])
    lossM = lossMSE(up, batch[2])
    lossN = lossNorm(up, batch[2])
    return cpu(up), lossM, lossN
  end
end

# Weird bug to initialize forward pass?
function initialize_gpus(ms, batch, w=workers())
  out = asyncmap(w) do p
    remotecall_fetch(p) do
      batch = gpu(batch)
      ms = gpu(ms)
      model = gen_mod(ms)
      CuArrays.reclaim()
      up = model(batch[1])
      CuArrays.reclaim()
      lossM = lossMSE(up, batch[2])
      lossN = lossNorm(up, batch[2])
      CuArrays.reclaim()
      return cpu(up), lossM, lossN
    end
  end
  return out[1]
end
################################################################################
### Training ###

# Get grads
@everywhere function return_gs(ms, data, loss, λ)
  data = gpu(data)
  ms = gpu(ms)
  model = gen_mod(ms)
  ps = gen_ps(ms)
  l(u0,u) = loss(model(u0),u)*(1+λ)
  lossM = lossMSE(model(data[1]), data[2])
  gs = gradient(ps) do
    l(data...)
  end
  flush(stderr)
  out = Dict(cpu(p)=>cpu(gs[p]) for p in ps)
  return lossM, out
end

# Parallel backprop
function train_b(w, DL, batch_idx, ms, loss, λ)
  consts, train_batch = get_train(DL, batch_idx)
  ps = gen_ps(ms)
  grads = Vector{Any}(undef,gpu_num)
  ls = Vector{Any}(undef,gpu_num)
  data =
    [(train_batch[1][:,:,:,:,(i-1)*mbs÷gpu_num+1:i*mbs÷gpu_num],
       train_batch[2][:,:,:,:,:,(i-1)*mbs÷gpu_num+1:i*mbs÷gpu_num])
       for i in 1:gpu_num]
  asyncmap(w) do p
    ls[p-1], grads[p-1] = remotecall_fetch(return_gs, p, ms, data[p-1], loss, λ)
  end
  gs = merge((x...)->(any(x.==nothing)) ? nothing : +(x...)./gpu_num, grads...)
  return mean(ls), ps, gs
end
