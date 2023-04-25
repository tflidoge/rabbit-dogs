import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
n = 1  # number of rabbits
m = 1 # number of dogs
dt = 0.001  # time step
N=(int)(100/dt)
pos_r = np.zeros((n, 2))
pos_d = np.zeros((m, 2))
pos_d[:, 0] = np.cos(np.linspace(0, 2*np.pi, m, endpoint=False)) 
pos_d[:, 1] = np.sin(np.linspace(0, 2*np.pi, m, endpoint=False)) 
caught = np.zeros(n, dtype=bool) 

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
                        break
    return T/Ns*dt

Ns=20
hatD_vals = np.linspace(0, 5, 50)
avg_times = []
for hatD in hatD_vals:
    times = Parallel(n_jobs=-1)(delayed(capture)(Ns, hatD) for i in range(20))
    avg_time = sum(times) / len(times)
    avg_times.append(avg_time)
    print(f'hatD: {hatD:.2f}, average time: {avg_time:.2f}s')
