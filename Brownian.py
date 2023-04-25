import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define parameters
n = 1  # number of rabbits
m = 3 # number of dogs
dt = 0.01  # time step
N=(int)(100/dt)
hatD=0.2
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
def capture(Ns):
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
                    pos_r[j, :] +=  2*np.sqrt(dt *hatD) * np.random.randn(2)
            # Update dog positions and find nearest rabbit
            if liveN==0:
                T+=k 
                break  
            for j in range(m):
                if not np.all(caught):
                    # Calculate distance from dog to each rabbit
                    dist = np.linalg.norm(pos_r - pos_d[j], axis=1)
                    # Find index of nearest rabbit
                    idx = np.argmin(dist)
                    while caught[idx]:
                        # If rabbit is already caught, find the next nearest one
                        dist[idx] = np.inf
                        idx = np.argmin(dist)
                    # Calculate direction vector from dog to rabbit
                    dir_vec = pos_r[idx, :] - pos_d[j, :]
                    # Calculate distance from dog to rabbit
                    dist = np.linalg.norm(dir_vec)
                    # Normalize direction vector and update dog position
                    if dist > 0:
                        dir_vec /= dist 
                    pos_d[j, :] += v * dir_vec * dt
                    # Check if dog caught a rabbit
                    if dist <= 0.1:
                        caught[idx] = True
            
    return T/Ns*dt        
# Define the update function for the animation
def update(i):
    # Update rabbit positions with Brownian motion
    for j in range(n):
        if not caught[j]:
            #pos_r[j, :] += np.sqrt(dt) * sigma * np.random.randn(2)
            pos_r[j, :] += 2*np.sqrt(dt *hatD) * np.random.randn(2)
    # Update dog positions and find nearest rabbit
    for j in range(m):
        if not np.all(caught):
            # Calculate distance from dog to each rabbit
            dist = np.linalg.norm(pos_r - pos_d[j], axis=1)
            # Find index of nearest rabbit
            idx = np.argmin(dist)
            while caught[idx]:
                # If rabbit is already caught, find the next nearest one
                dist[idx] = np.inf
                idx = np.argmin(dist)
            # Calculate direction vector from dog to rabbit
            dir_vec = pos_r[idx, :] - pos_d[j, :]
            # Calculate distance from dog to rabbit
            dist = np.linalg.norm(dir_vec)
            # Normalize direction vector and update dog position
            if dist > 0:
                dir_vec /= dist 
            pos_d[j, :] += v * dir_vec * dt
            # Check if dog caught a rabbit
            if dist <= 0.1:
                caught[idx] = True
    # Update the plot data
    for j in range(n):
        lines_r[j].set_data(pos_r[j, 0], pos_r[j, 1])
        if caught[j]:
            lines_r[j].set_color('C3')
    for j in range(m):
        lines_d[j].set_data(pos_d[j, 0], pos_d[j, 1])
    return lines_r + lines_d

#Create the animation object and show the plot
re = np.zeros(20)
for i in range(20):
    hatD=i*0.2
    re[i]=capture(10)
    print(hatD,re[i])
txtf="data.txt"
f=open(txtf,"w+")
np.savetxt("data.txt",re)
# Initialize position arrays for the rabbits and the dogs
pos_r = np.zeros((n, 2))
pos_d = np.zeros((m, 2))
pos_d[:, 0] = np.cos(np.linspace(0, 2*np.pi, m, endpoint=False)) 
pos_d[:, 1] = np.sin(np.linspace(0, 2*np.pi, m, endpoint=False)) 
caught = np.zeros(n, dtype=bool)  # boolean array indicating whether a rabbit is caught
ani = FuncAnimation(fig, update, frames=np.arange(1, N), interval=50, blit=True)
plt.show()