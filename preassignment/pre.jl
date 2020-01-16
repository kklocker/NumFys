using Random
using Plots;pyplot()

N=1000000
n= 100
t = [1:n]

r_dist =  randn!(MersenneTwister(), zeros(N,n))
println(typeof(r_dist)) # Array{Float64,2}
println(size(r_dist))
@time random_walks = cumsum(r_dist,dims = 2)
println(size(random_walks))

println(typeof(random_walks))
@time function checkstuff(index::Int,prob,walk::Array{Float64,2})
    for j=1:n
        if walk[index,1] >0
            if random_walks[index,j] <0
                prob[j] += 1
                return
            end
        elseif walk[index,1] <0
            if random_walks[index,j] >0
                prob[j] += 1
                return
            end
        end
    end
    return
end

p = zeros(n)

@time for i=1:N
    checkstuff(i,p,random_walks)
end

a = plot(p)
display(a)
readline()
