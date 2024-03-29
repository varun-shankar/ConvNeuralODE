using Random
using DelimitedFiles
using Statistics
################################################################################

mutable struct DataLoader
  train_size
  test_size::Int
  tsize::Int
  mbs::Int
  shuff::Bool
  dir::String
  data_idxs
  train_idxs
  test_idxs
  globDict::Dict
  channels::Int
  stepsτ::Int

  function DataLoader(train_size_, test_size_, minibatch_size_, tsize_,
                      dir_=data_dir, stepsτ_=100, shuff_=true)

    dir = dir_
    train_size = train_size_
    test_size = test_size_
    mbs = minibatch_size_
    tsize = tsize_
    shuff = shuff_
    channels = 3
    stepsτ = stepsτ_

    temp = readdlm(dir*"/global",comments=true,comment_char='/')
    globDict = Dict(temp[i,1]=>temp[i,2] for i in 1:size(temp,1))

    data_idxs = 0:globDict["dtStat"]*stepsτ:globDict["tsEnd"]
    data_idxs = Array(data_idxs[1:end-1])

    # File lists
    if isa(train_size, Int)
      train_idxs = data_idxs[1:train_size]
    else
      train_idxs = data_idxs[train_size]
      train_size = length(train_size)
    end
    if isa(test_size, Int)
      test_idxs = reverse(data_idxs)[1:test_size]
    else
      test_idxs = data_idxs[test_size]
      test_size = length(test_size)
    end

    #batching
    train_idxs = [train_idxs[(i-1)*mbs+1:i*mbs] for i in 1:Int(train_size/mbs)]

    new(train_size, test_size, tsize, mbs, shuff, dir,
        data_idxs, train_idxs, test_idxs, globDict, channels, stepsτ)
  end
end

# Reading a binary file
function readbin(filename, T)
  out = Vector{T}(undef, filesize(filename)÷sizeof(T))
  read!(filename, out)
end

# Get training batch
function get_train(DL::DataLoader, idx, bs=nothing, norm=true)
  if bs == nothing; bs = 1:DL.mbs; end
  u_train_batch = Float32.(zeros(
                  DL.globDict["nx"],DL.globDict["ny"],DL.globDict["nz"],
                  DL.channels,DL.tsize,length(bs)))
  for b = 1:length(bs)
    ts = range(DL.train_idxs[idx][bs[b]],length=DL.tsize,step=DL.globDict["dtStat"]*DL.stepsτ÷DL.tsize)
    for t = 1:length(ts)
      for v = 1:3
        snap = readbin(DL.dir*"/data_single/u"*string(v)*"."*string(ts[t]), Float32)
        snap = reshape(snap,2+DL.globDict["nx"],DL.globDict["ny"],DL.globDict["nz"])
        snap = snap[1:end-2,:,:]
        u_train_batch[:,:,:,v,t,b] = snap
      end
    end
  end
  u0_train_batch = u_train_batch[:,:,:,:,1,:]
  if norm
    return normalize((u0_train_batch, u_train_batch))
  else
    return ones(length(bs)), (u0_train_batch, u_train_batch)
  end
end

# Get the test batch
function get_test(DL::DataLoader, norm=true)
  u_test = Float32.(zeros(
           DL.globDict["nx"],DL.globDict["ny"],DL.globDict["nz"],
           DL.channels,DL.tsize,DL.test_size))
  for b = 1:DL.test_size
    ts = range(DL.test_idxs[b],length=DL.tsize,step=DL.globDict["dtStat"]*DL.stepsτ÷DL.tsize)
    for t = 1:length(ts)
      for v = 1:3
        snap = readbin(DL.dir*"/data_single/u"*string(v)*"."*string(ts[t]), Float32)
        snap = reshape(snap,2+DL.globDict["nx"],DL.globDict["ny"],DL.globDict["nz"])
        snap = snap[1:end-2,:,:]
        u_test[:,:,:,v,t,b] = snap
      end
    end
  end
  u0_test = u_test[:,:,:,:,1,:]
  if norm
    return normalize((u0_test, u_test))
  else
    return ones(DL.test_size), (u0_test, u_test)
  end
end

function reshuffle!(DL::DataLoader, test=false)
  DL.train_idxs = collect(Iterators.flatten(DL.train_idxs))
  shuffle!(DL.train_idxs)
  if test
    DL.test_idxs = collect(Iterators.flatten(DL.test_idxs))
    shuffle!(DL.test_idxs)
  end
  #batching
  DL.train_idxs =
  [DL.train_idxs[(i-1)*DL.mbs+1:i*DL.mbs] for i in 1:Int(DL.train_size/DL.mbs)]
end
