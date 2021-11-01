using Zygote
using Flux
using FFTW
using VectorizedRoutines

function laplaceFlux(N, periodic=false)
    kernel = zeros(Float32, fill(3, N)...)
    kernel[fill(2, N)...] = -2*N
    for i=1:N
        idxs = fill([2], N)
        idxs[i] = [1,3]
        kernel[idxs...] = ones(2)
    end
    kernel = reshape(kernel,size(kernel)...,1,1)
    if !periodic
        lapFilt = Conv(kernel, Float32.([0]), pad = 1)
        out = lapFilt
    else
        lapFilt = Conv(kernel, Float32.([0]), pad = 0)
        out = Chain(pad3D(1), lapFilt)
    end
    return out
end

function gradFlux(N, periodic=false)
    kernel = zeros(Float32, fill(3, N)..., 1, N)
    for i=1:N
        idxs = fill([2], N)
        idxs[i] = [1,3]
        kernel[idxs...,1,i] = [1,-1]./2
    end
    if !periodic
        gradFilt = Conv(kernel, Float32.([0]), pad = 1)
        out = gradFilt
    else
        gradFilt = Conv(kernel, Float32.([0]), pad = 0)
        out = Chain(pad3D(1), gradFilt)
    end
    return out
end

gf = gradFlux(3)
function curl3D(F, g=gpu(gf))
    out = Zygote.Buffer(F)
    dFx = g(F[:,:,:,1:1,:])
    dFy = g(F[:,:,:,2:2,:])
    dFz = g(F[:,:,:,3:3,:])
    out[:,:,:,1,:] = dFz[:,:,:,2,:].-dFy[:,:,:,3,:]
    out[:,:,:,2,:] = dFx[:,:,:,3,:].-dFz[:,:,:,1,:]
    out[:,:,:,3,:] = dFy[:,:,:,1,:].-dFx[:,:,:,2,:]
    return copy(out)
end

function div3D(F, g=gpu(gf))
    out = Zygote.Buffer(F[:,:,:,1:1,:])
    dFx = g(F[:,:,:,1:1,:])
    dFy = g(F[:,:,:,2:2,:])
    dFz = g(F[:,:,:,3:3,:])
    out = dFx[:,:,:,1:1,:].+dFy[:,:,:,2:2,:].+dFz[:,:,:,3:3,:]
    return copy(out)
end

function periodic_pad(pad)
    padder = function (x)
        N = length(size(x))
        out = Zygote.Buffer(x, 2 .*pad.+size(x))
        for i = 1:N
            p = pad[i]
            if p != 0
                s = repeat(Any[:], N)
                s[i] = size(x, i)-p+1:size(x, i)
                left = x[s...]
                s[i] = 1:p
                right = x[s...]
                x = cat(left, x, right, dims=i)
            end
        end
        out[:] = x
        return copy(out)
    end
    return padder
end

function pad3D(p)
    padder = function (x)
        left = x[end-p+1:end,:,:,:,:]
        right = x[1:p,:,:,:,:]
        x = cat(left, x, right, dims=1)
        left = x[:,end-p+1:end,:,:,:]
        right = x[:,1:p,:,:,:]
        x = cat(left, x, right, dims=2)
        left = x[:,:,end-p+1:end,:,:]
        right = x[:,:,1:p,:,:]
        x = cat(left, x, right, dims=3)
        return x
    end
    return padder
end

lfp3 = laplaceFlux(3,true)
gfp3 = gradFlux(3,true)
# pad3D(p) = periodic_pad((p,p,p,0,0))

kh = rfftfreq(64,64)
k = fftfreq(64,64)
kgrid = Matlab.meshgrid(kh,k,k)
k_1 = permutedims(kgrid[1], [2,1,3])
k_2 = permutedims(kgrid[2], [2,1,3])
k_3 = permutedims(kgrid[3], [2,1,3])

function specProj3D(u)
    k1, k2, k3 = gpu((k_1,k_2,k_3))
    U1 = rfft(u[:,:,:,1]); U2 = rfft(u[:,:,:,2]); U3 = rfft(u[:,:,:,3])
    kdotU = (k1 .*U1 .+ k2 .*U2 .+ k3 .*U3)
    kdotk = (k1.^2 .+ k2.^2 .+ k3.^2).+eps(Float32)
    norm = kdotU./kdotk
    Uh1 = U1 .- norm.*k1
    Uh2 = U2 .- norm.*k2
    Uh3 = U3 .- norm.*k3
    uhat = cat(irfft(Uh1,64),
               irfft(Uh2,64),
               irfft(Uh3,64),dims=4)
    out = uhat
end

function specProj3Db(u)
    out = Zygote.Buffer(u)
    for i in 1:size(u,5)
        out[:,:,:,:,i] = specProj3D(u[:,:,:,:,i])
    end
    return copy(out)
end

Zygote.@adjoint function irfft(xs, d)
  return AbstractFFTs.irfft(xs, d), function(Δ)
    total = length(Δ)
    fullTransform = AbstractFFTs.rfft(real.(Δ)) / total
    return (fullTransform, nothing)
  end
end
