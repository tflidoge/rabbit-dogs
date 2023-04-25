using PyPlot
using Random
using LinearAlgebra
using Distributed
using Base.Threads
using Distances
using Polynomials
#using DataStructures

# Define parameters
n = 1  # number of rabbits
m = 1  # number of dogs
dt = 0.001  # time step
N = round(Int,100/dt) #the total steps to catch all rabbtis
sigma = 2  # standard deviation of the Gaussian distribution for rabbits
vr = 2
v = 1  # speed of the dogs


# Initialize position arrays for the rabbits and the dogs
pos_r = zeros(n, 2)
pos_d = zeros(m, 2)
pos_d[:, 1] = cos.(range(0, stop=2*pi, length=m+1)[1:end-1])
pos_d[:, 2] = sin.(range(0, stop=2*pi, length=m+1)[1:end-1])
caught = zeros(Bool, n)  # boolean array indicating whether a rabbit is caught
nearest_rabbit=zeros(m)
# # Set up the plot
# fig, ax = PyPlot.subplots()
# L = 2
# ax.set_xlim(-L, L)
# ax.set_ylim(-L, L)
# ax.set_aspect("equal")
# lines_r = []
# lines_d = []

# for i in 1:n
#     line_r, = ax.plot([], [], "o", color="C0")
#     push!(lines_r, line_r)
# end

# for i in 1:m
#     line_d, = ax.plot([], [], "x", color="C1")
#     push!(lines_d, line_d)
# end

function capture(Ns, hatD)
    T = 0
    for rounds in 1:Ns
        # get the current time in seconds
        t = time()

        # convert the time to an integer seed
        seed = Int(floor(1e9 * t))

        # set the random seed
        Random.seed!(seed)
        pos_r = zeros(n, 2)
        pos_d = zeros(m, 2)
        pos_d[:, 1] = cos.(range(0, stop=2*pi, length=m+1)[1:end-1])
        pos_d[:, 2] = sin.(range(0, stop=2*pi, length=m+1)[1:end-1])
        caught = zeros(Bool, n)  # boolean array indicating whether a rabbit is caught
        
        for k in 1:N
            liveN = 0
            # Update rabbit positions with Brownian motion
            for j in 1:n
                if !caught[j]
                    #print("alive")
                    liveN += 1
                    pos_r[j, :] .+= sqrt(2*dt *hatD) .* randn(2)
                end
            end
            # Update dog positions and find nearest rabbit
            if liveN == 0
                T += k 
                break  
            end
            
            for j in 1:m
                # compute the distances between the j-th B particle and all A particles
                distances = [euclidean(pos_d[j, :], pos_r[ii, :, :]) for ii in findall(!, caught)]
                
                # find the index of the A particle with the minimum distance to the j-th B particle
                nearest_rabbit[j] = argmin(distances)
            end
            
            for j in 1:m
                
                nearest=Int(nearest_rabbit[j])
                direction = pos_r[nearest, :] - pos_d[j, :]
                direction /= norm(direction)
                pos_d[j, :] .+= v * direction * dt
                if euclidean(pos_d[j, :], pos_r[nearest, :, :])<0.1
                    caught[nearest] = true
                end
            end
        end
    end
    return T/Ns*dt
end
Ns = 10
hatD_vals = range(0, stop=5, length=10)
avg_times = []
for hatD in hatD_vals
    times = Float64[]
    @threads for i in 1:36
        push!(times, capture(Ns, hatD))
    end
    avg_time = sum(times) / length(times)
    push!(avg_times, avg_time)
    println("hatD: ", round(hatD, digits=2), ", average time: ", round(avg_time, digits=2), "s")
end

# # 进行线性拟合
# fit = polyfit(hatD_vals, avg_times, 1)
# # 获取斜率和截距
# slope, intercept = fit

# # 输出拟合结果
# println("斜率：", slope)
# println("截距：", intercept)

# # Plot the results
# fig, ax = PyPlot.subplots()
# ax.plot(hatD_vals, avg_times)
# ax.set_xlabel("hatD")
# ax.set_ylabel("Average time to catch all rabbits")
# PyPlot.show()