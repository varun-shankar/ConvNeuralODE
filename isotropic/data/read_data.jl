using HDF5
using Random
################################################################################

mutable struct DataLoader
  numpts::Int
  channels::Int
  tsize::Int
  train_size::Int
  test_size::Int
  mbs::Int
  dir::String
  datafiles
  trainfiles
  testfiles
  shuffle::Bool

  function DataLoader(train_size_, test_size_, minibatch_size_,
                      tsize_=tsize, numpts_=numpts, channels_=channels,
                      shuffle_=false)

    dir = "/data/vshankar/ConvNODE/3D/"

    train_size = train_size_
    test_size = test_size_
    mbs = minibatch_size_
    numpts = numpts_
    channels = channels_
    tsize = tsize_
    shuffle = shuffle_

    datafiles = readdir(dir)
    datafiles = datafiles[occursin.(".h5", datafiles)]

    # File lists
    trainfiles = datafiles[1:train_size]
    testfiles = reverse(datafiles)[1:test_size]

    if shuffle
      shuffle!(trainfiles)
      shuffle!(testfiles)
    end

    #batching
    trainfiles = [trainfiles[(i-1)*mbs+1:i*mbs] for i in 1:Int(train_size/mbs)]

    new(numpts,channels,tsize,train_size,test_size,mbs,
        dir,datafiles,trainfiles,testfiles,shuffle)
  end
end

function get_train(DL::DataLoader, idx, use_gpu=false)
  u_train_batch = Float32.(zeros(DL.numpts,DL.numpts,DL.numpts,
                                 DL.channels,DL.tsize,DL.mbs))
  for j = 1:DL.mbs
    fid = h5open(DL.dir*DL.trainfiles[idx][j],"r")
    i = 1
    for obj in fid
      data = read(obj)
      if length(size(data)) == 4
        u_train_batch[:,:,:,:,i,j] = permutedims(data,[2,3,4,1])
      end
      i+=1
    end
    close(fid)
  end
  u0_train_batch = u_train_batch[:,:,:,:,1,:]
  if use_gpu
    u0_train_batch = cu(u0_train_batch)
    u_train_batch = cu(u_train_batch)
  end
  return u0_train_batch, u_train_batch
end

function get_test(DL::DataLoader, use_gpu=false)
  u_test = Float32.(zeros(DL.numpts,DL.numpts,DL.numpts,
                          DL.channels,DL.tsize,DL.test_size))
  for j = 1:DL.test_size
    fid = h5open(DL.dir*DL.testfiles[j],"r")
    i = 1
    for obj in fid
      data = read(obj)
      if length(size(data)) == 4
        u_test[:,:,:,:,i,j] = permutedims(data,[2,3,4,1])
      end
      i+=1
    end
    close(fid)
  end
  u0_test = u_test[:,:,:,:,1,:]
  if use_gpu
    u0_test = cu(u0_test)
    u_test = cu(u_test)
  end
  return u0_test, u_test
end

function reshuffle!(DL::DataLoader)
  DL.trainfiles = DL.datafiles[1:DL.train_size]
  DL.testfiles = reverse(DL.datafiles)[1:DL.test_size]

  shuffle!(DL.trainfiles)
  shuffle!(DL.testfiles)

  #batching
  DL.trainfiles =
  [DL.trainfiles[(i-1)*DL.mbs+1:i*DL.mbs] for i in 1:Int(DL.train_size/DL.mbs)]
end
