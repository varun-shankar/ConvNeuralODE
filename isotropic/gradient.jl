using Zygote
using Flux

function laplaceFlux(N)
    kernel = zeros(Float32, fill(3, N)...)
    kernel[fill(2, N)...] = -2*N
    for i=1:N
        idxs = fill([2], N)
        idxs[i] = [1,3]
        kernel[idxs...] = ones(2)
    end
    kernel = reshape(kernel,size(kernel)...,1,1)
    lapFilt = Conv(kernel, Float32.([0]), pad = 1)
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
        out = Chain(periodic_pad((fill(1, N)...,0,0)), gradFilt)
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

pad3D(p) = periodic_pad((p,p,p,0,0))
