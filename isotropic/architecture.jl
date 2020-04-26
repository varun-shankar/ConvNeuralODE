using DifferentialEquations
using Flux
using Zygote
using DiffEqFlux
using DiffEqSensitivity
################################################################################

# Reshaping
t2b(x) = reshape(x,size(x,1),size(x,2),size(x,3),size(x,4),prod(size(x)[5:6]))
b2t(x) = reshape(x,size(x,1),size(x,2),size(x,3),size(x,4),tsize,size(x,5)÷tsize)

# Channels
Dz, Dh, Da = 4, 50, 2

# Encoder
encode = Chain(Conv((4,4,4), 3=>Dh, pad=1, stride=2, relu),
               Conv((3,3,3), Dh=>Dz, pad=1, stride=1))

# Decoder
decode = Chain(ConvTranspose((3,3,3), Dz=>Dh, pad=1, stride=1, relu),
               ConvTranspose((4,4,4), Dh=>3, pad=1, stride=2))

function decode_serial(x, decode)
  out = Zygote.Buffer(x, 64,64,64,3,tsize,size(x,6))
  for i = 1:tsize
    out[:,:,:,:,i,:] = decode(x[:,:,:,:,i,:])
  end
  return copy(out)
end

## dldt
dldt = Chain(Conv((3,3,3), 6=>32, pad=1, relu),
             Conv((3,3,3), 32=>64, pad=1, relu),
             Conv((3,3,3), 64=>64, pad=1, relu),
             Conv((3,3,3), 64=>32, pad=1, relu),
             Conv((3,3,3), 32=>6, pad=1))

# Augment
add_p(x) = begin
            a = zeros(size(x,1),size(x,2),size(x,3),Da,size(x,5))
            (isa(x,Array)) ? cat(x,cpu(a),dims=4) : cat(x,gpu(a),dims=4)
           end
rm_p(x) = x[:,:,:,1:Dz,:,:]

n_ode = NeuralODE(dldt,tspan,#sensealg=InterpolatingAdjoint(),
                 Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)

# Final model
ms = [encode,n_ode,decode]
# Generate the full model
gen_mod(ms) = Chain(ms[1],
                    add_p,
                    ms[2],
                    x -> (isa(x,Array)) ? cpu(x) : gpu(x),
                    x -> permutedims(x,[1,2,3,4,6,5]),
                    rm_p,
                    x -> decode_serial(x, ms[3])) #t2b, decode, b2t
# Generate params for training
gen_ps(ms) = params(ms)
