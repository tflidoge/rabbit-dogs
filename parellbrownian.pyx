import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from joblib import Parallel, delayed

# Define parameters
n = 1  # number of rabbits
m = 1 # number of dogs
dt = 0.001  # time step
N=(int)(100/dt)
sigma = 2  # standard deviation of the Gaussian distribution for rabbits
vr=2
v = 1  # speed of the dogs

# Initialize position arrays for the rabbits and the dogs
pos_r = np.zeros((n, 2))
pos_d = np.zeros((m, 2))
pos_d[:, 0] = np.cos(np.linspace(0, 2*np.pi, m, endpoint=False)) 
pos_d[:, 1] = np.sin(np.linspace(0, 2*np.pi, m, endpoint=False)) 
caught = np.zeros(n, dtype=bool)  # boolean array indicating whether a rabbit is caught

# Set up the plot
fig, ax = plt.subplots()
L=2
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_aspect('equal')
lines_r = []
for i in range(n):
    line_r, = ax.plot([], [], 'o', color='C0')
    lines_r.append(line_r)
lines_d = []
for i in range(m):
    line_d, = ax.plot([], [], 'x', color='C1')
    lines_d.append(line_d)

def capture(Ns, hatD):
    T=0
    for rounds in range(Ns):
        pos_r = np.zeros((n, 2))
        pos_d = np.zeros((m, 2))
        pos_d[:, 0] = np.cos(np.linspace(0, 2*np.pi, m, endpoint=False)) 
        pos_d[:, 1] = np.sin(np.linspace(0, 2*np.pi, m, endpoint=False)) 
        caught = np.zeros(n, dtype=bool)  # boolean array indicating whether a rabbit is caught
        k=0
        for k in range(N):
            liveN=0
            # Update rabbit positions with Brownian motion
            for j in range(n):
                if not caught[j]:
                    liveN+=1
                    #pos_r[j, :] += np.sqrt(dt) * sigma * np.random.randn(2)
                    pos_r[j, :] +=  np.sqrt(2*dt *hatD) * np.random.randn(2)
            # Update dog positions and find nearest rabbit
            if liveN==0:
                T+=k 
                break  
            for j in range(m):
                if not np.all(caught):
                    # Calculate distance from dog to each rabbit
                    dist = np.linalg.norm(pos_r - pos_d[j, :], axis=1)
                    # Find nearest rabbit that hasn't been caught yet
                    nearest = np.argmin(dist[~caught])
                    # Move dog towards nearest rabbit
                    if not np.isnan(nearest):
                        direction = pos_r[nearest, :] - pos_d[j, :]
                        direction /= np.linalg.norm(direction)
                        pos_d[j, :] += v * direction * dt
            # Check if any rabbits are caught
            for j in range(n):
                if not caught[j]:
                    if np.any(np.linalg.norm(pos_r[j, :] - pos_d, axis=1) < 0.1):
                        caught[j] = True
                        lines_r[j].set_color('C3')
                        break
    return T/Ns*dt

Ns=20
hatD_vals = np.linspace(0, 5, 50)
avg_times = []
for hatD in hatD_vals:
    times = Parallel(n_jobs=-1)(delayed(capture)(Ns, hatD) for i in range(20))
    #print(times)
    avg_time = sum(times) / len(times)
    avg_times.append(avg_time)
    print(f'hatD: {hatD:.2f}, average time: {avg_time:.2f}s')
# 进行线性拟合
fit = np.polyfit(hatD_vals, avg_times, 1)
# 获取斜率和截距
slope, intercept = fit

# 输出拟合结果
print("斜率：", slope)
print("截距：", intercept)

# Plot the results
fig, ax = plt.subplots()
ax.plot(hatD_vals, avg_times)
ax.set_xlabel('hatD')
ax.set_ylabel('Average time to catch all rabbits')
plt.show()