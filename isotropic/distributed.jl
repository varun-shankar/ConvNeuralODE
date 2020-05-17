using Distributed

function distributed_setup(gpu_ids)
    gpu_num = length([Iterators.flatten(gpu_ids)...])
    if isa(gpu_ids,Array{Array{Int64,1},1}) && nprocs() == 1
        # Multiple host distributed
        @info "Multiple host distributed"
        hosts = [x for x in ARGS]
        host_num = length(hosts)
        if host_num!=length(gpu_ids); error("Host/GPU mismatch"); end
        for i in 1:host_num; addprocs([(hosts[i],length(gpu_ids[i]))]); end
        pis = [[] for i=1:host_num]
        for p in workers()
            h, pid = remotecall_fetch(p) do
                h = findall(gethostname().==hosts)[1]
                return h, myid()
            end
            push!(pis[h],pid)
        end
        pds = zip(Iterators.flatten(pis),Iterators.flatten(gpu_ids))
        return gpu_num, pds
    elseif isa(gpu_ids,Array{Int64,1}) && gpu_num > 1
        # Single host distributed
        @info "Single host distributed"
        if nprocs() == 1 # If you're on the REPL
            addprocs(gpu_num)
        end
        pds = zip(workers(), gpu_ids)
        return gpu_num, pds
    else
        # Serial
        @info "Single host serial"
        return gpu_num, nothing
    end
end
