using DifferentialEquations
using Flux
using Zygote
using DiffEqFlux
using DiffEqSensitivity
using BSON
################################################################################

# HPs
Nz, Nk = 4, 3
Nh, Na, Nw, Nl = 16, 2, 16, 4

## Helpers

# Reshaping
t2b(x) = reshape(x,size(x,1),size(x,2),size(x,3),size(x,4),prod(size(x)[5:6]))
b2t(x) = reshape(x,size(x,1),size(x,2),size(x,3),size(x,4),tsize,Int(size(x,5)/tsize))

add_p(x, Na) = begin
            y = zeros(size(x,1),size(x,2),size(x,3),Na,size(x,5))
            (isa(x,Array)) ? cat(x,cpu(y),dims=4) : cat(x,gpu(y),dims=4)
           end
rm_p(x, Na) = x[:,:,:,1:end-Na,:,:]

function n_ode(x, dldt)
  out = Zygote.Buffer(x, size(x)...,tsize)
  out[:,:,:,:,:,1] = x
  for i = 2:tsize
    out[:,:,:,:,:,i] = out[:,:,:,:,:,i-1].+dldt(out[:,:,:,:,:,i-1]).*(t[2]-t[1])
  end
  return copy(out)
end

# Serial decoder
function decode_serial(x, decode)
  out = Zygote.Buffer(x, 64,64,64,3,tsize,size(x,6))
  for i = 1:tsize
    out[:,:,:,:,i,:] = decode(x[:,:,:,:,i,:])
  end
  return copy(out)
end

## Create model
function arch(Nh, Nz, Na, Nk, Nw, Nl)
  encode = Chain(Conv((4,4,4), 3=>Nh, pad=1, stride=2, swish),
                 Conv((3,3,3), Nh=>Nz, pad=1, stride=1))
  decode = Chain(ConvTranspose((3,3,3), Nz=>Nh, pad=1, stride=1, swish),
                 ConvTranspose((4,4,4), Nh=>3, pad=1, stride=2))

  layers = []
  for i = 1:Nl
    if i==1
      dldt = push!(layers, Conv((Nk,Nk,Nk), Nz+Na=>Nw, pad=floor(Int,Nk/2), swish))
    elseif i==Nl
      dldt = push!(layers, Dropout(0.3), Conv((Nk-2,Nk-2,Nk-2), Nw=>Nz+Na, pad=floor(Int,(Nk-2)/2)))
    else
      dldt = push!(layers, Conv((Nk-2,Nk-2,Nk-2), Nw=>Nw, pad=floor(Int,(Nk-2)/2), swish))
    end
  end
  dldt = Chain(layers...)
  node = NODE(dldt,tspan,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                                  #BacksolveAdjoint(checkpointing=true),
                   Euler(),tstops=t,saveat=t,reltol=1e-4,abstol=1e-6)

  ms = [encode,node,decode]
  # gen_mod(ms) = Chain(ms[1],
  #                     x -> add_p(x, Na),
  #                     x -> n_ode(x, ms[2]),
  #                     x -> (isa(x,Array)) ? cpu(x) : gpu(x),
  #                     x -> permutedims(x,[1,2,3,4,6,5]),
  #                     x -> rm_p(x, Na),
  #                     t2b,
  #                     ms[3], specProj3Db,
  #                     b2t)
  # gen_ps(ms) = params(ms)

  return ms#, gen_mod, gen_ps
end

# ms, _, _ = arch(Nh, Nz, Na, Nk, Nw, Nl)


encode = Chain(pad3D(1), Conv((4,4,4), 3=>Nh, pad=0, stride=2, swish),
               pad3D(1), Conv((3,3,3), Nh=>Nz, pad=0, stride=1))
decode = Chain(ConvTranspose((3,3,3), Nz=>Nh, pad=1, stride=1, swish),
               ConvTranspose((4,4,4), Nh=>3, pad=1, stride=2))


dldt = Chain(pad3D(1), Conv((3,3,3), Nz+Na=>Nw, pad=0, swish),
             pad3D(1), Conv((3,3,3), Nw=>Nw, pad=0, swish),
             pad3D(1), Conv((3,3,3), Nw=>Nw, pad=0, swish),
             pad3D(1), Conv((3,3,3), Nw=>Nw, pad=0, swish),
             pad3D(1), Conv((3,3,3), Nw=>Nw, pad=0, swish),
             pad3D(1), Conv((3,3,3), Nw=>Nw, pad=0, swish),
             pad3D(1), Conv((3,3,3), Nw=>Nw, pad=0, swish),
             pad3D(1), Conv((3,3,3), Nw=>Nw, pad=0, swish),
	     pad3D(1), Conv((3,3,3), Nw=>Nz+Na, pad=0))

node = NODE(dldt,tspan,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                                #BacksolveAdjoint(checkpointing=true),
                                #DiffEqBase.SensitivityADPassThrough(),
                 Tsit5(),#tstops=ts,
                 saveat=t,reltol=1e-4,abstol=1e-6)

gen_mod(ms) = Chain(
                    ms[1],
                    x -> add_p(x, Na),
                    # x -> n_ode(x, ms[2]),
                    ms[2],
                    x -> (isa(x, Array)) ? cpu(x) : gpu(x),
                    x -> permutedims(x,[1,2,3,4,6,5]),
                    x -> rm_p(x, Na),
                    t2b,
                    ms[3],
                    # x -> reshape(x,size(x,1),size(x,2),size(x,3),size(x,4),1,size(x,5))
                    specProj3Db,
                    b2t
                    )

ms = [encode,node,decode]

gen_ps(ms) = Flux.params(ms)
