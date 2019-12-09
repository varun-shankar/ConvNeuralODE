using DifferentialEquations
using Plots
using Interpolations
using CuArrays
using Flux
using DiffEqFlux
using NPZ
using BSON: @save
using BSON: @load
CuArrays.allowscalar(false)
################################################################################
### Functions ###

# Blasius ODE
function blasius(η_max, dη)
    function blasiusODE(du,u,p,t)
        du[1] = u[2]
        du[2] = u[3]
        du[3] = -u[1].*u[3]
    end

    function b_bc!(residual,u,p,t)
        residual[1] = u[1][1]
        residual[2] = u[1][2]
        residual[3] = u[end][2] - 1
    end

    tspan = (0.0,η_max)
    bvp = TwoPointBVProblem(blasiusODE, b_bc!, zeros(3), tspan)
    sol = solve(bvp, MIRK4(), dt=dη)
    return sol
end

# Calculate velocity field
function plate_flow(X, Y, U, ν, blasius_sol)
    η_range = range(blasius_sol.t[1],stop=blasius_sol.t[end],length=length(blasius_sol.t))
    η_grid = Y.*sqrt.(U./(2 .*ν.*X))

    f0_int = extrapolate(scale(interpolate(blasius_sol[1,:], BSpline(Linear())), η_range), Line())
    f1_int = extrapolate(scale(interpolate(blasius_sol[2,:], BSpline(Linear())), η_range), Line())

    u = U.*f1_int.(η_grid)
    v = 0.5.*sqrt.(U.*ν./X).*(η_grid.*f1_int.(η_grid) - f0_int.(η_grid))

    return cat(u,v,dims=3)
end

################################################################################
### Constants ###
numpts = 2^7
datasize = 50
start_i = 1

U = 1; ν = 0.001

x = range(0.1,stop=1,length=numpts)
y = range(0.00001,stop=0.3,length=numpts)
Y = repeat(y, outer=(1, numpts))
X = repeat(x, outer=(1, numpts))'

tspan = (Float32(x[start_i]),Float32(x[datasize+start_i-1]))
t = range(tspan[1],tspan[2],length=datasize)

################################################################################
### Calculate Blasius ###

bsol = blasius(50., 0.1)
u = gpu(Float32.(plate_flow(X, Y, U, ν, bsol)))
u0 = u[:,start_i,:]
u_train = u[:,start_i:datasize+start_i-1,:]

################################################################################
### Define architecture ###

## Latent
dudt = Chain(x -> reshape(x,8,1,32,1),
             Conv((1,1), 32=>50, pad=0, relu),
             Conv((1,1), 50=>32, pad=0),
             x -> reshape(x,8*32,1))

n_ode(x) = neural_ode(gpu(dudt),x,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)

encode = Chain(x -> reshape(x,numpts,1,2,1),
               Conv((4,1), 2=>4, pad=(1,1,0,0), stride=(2,1), relu),
               Conv((2,1), 4=>8, pad=0, stride=(2,1), relu),
               Conv((2,1), 8=>16, pad=0, stride=(2,1), relu),
               Conv((2,1), 16=>32, pad=0, stride=(2,1), relu),
               x -> reshape(x,8*32,1))

decode = Chain(x -> permutedims(reshape(x,8,32,datasize,1),[1,3,2,4]),
               ConvTranspose((2,1), 32=>16, pad=0, stride=(2,1), relu),
               ConvTranspose((2,1), 16=>8, pad=0, stride=(2,1), relu),
               ConvTranspose((2,1), 8=>4, pad=0, stride=(2,1), relu),
               ConvTranspose((4,1), 4=>2, pad=(1,1,0,0), stride=(2,1)),
               x -> reshape(x,numpts,datasize,2))

model = Chain(gpu(encode), x -> n_ode(x), gpu(decode))

function predict_n_ode(x)
    model(x)
end
loss_n_ode() = sum(abs,u_train .- reshape(predict_n_ode(u0),numpts,datasize,2))
#@load "ed-gpu-checkpoint.bson" w1
#@load "du-gpu-checkpoint.bson" w2
#Flux.loadparams!(model, w1)
#Flux.loadparams!(dudt, w2)

################################################################################
### Training ###

data = Iterators.repeated((), 10)
opt = ADAM(0.001)


cb = function ()
    display(loss_n_ode())
    #=
    w1 = Tracker.data.(Flux.params(model))
    w2 = Tracker.data.(Flux.params(dudt))
    println(" ")
    flush(stdout)
    @save "ed-gpu-checkpoint.bson" w1
    @save "du-gpu-checkpoint.bson" w2
    cur_pred = reshape(Flux.data(predict_n_ode(u0)),numpts,datasize,2)

    cp = contour(x[start_i:datasize+start_i-1],y,cur_pred[:,:,1],fill=true,
               c=:plasma,clims=(0,U),levels=range(0,stop=U,length=100))
    cu = contour(x[start_i:datasize+start_i-1],y,u_train[:,:,1],fill=true,
               c=:plasma,clims=(0,U),levels=range(0,stop=U,length=100))
    display(plot(cp, cu, layout = (2,1), size = (600,800)))
    =#
end

ps = Flux.params(model,dudt);
@time Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
#display(loss_n_ode())

####################################################c############################
### Plotting ###

#=
pyplot()
cu = contour(x,y,u[:,:,1],fill=true, c = :plasma,
             clims=(0,U),levels=range(0,stop=U,length=100))
cv = contour(x,y,u[:,:,2],fill=true, c = :plasma,
             clims=(0,maximum(u[:,2,2])),levels=range(0,stop=maximum(u[:,2,2]),length=100))
plot(cu, cv, layout = (2,1), size = (600,800))
contour!(x,y,u[:,:,1],levels=[0.99*U,2*U],c=:black)
=#
