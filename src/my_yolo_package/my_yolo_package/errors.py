import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('tracking_log4.csv', delimiter=',', skiprows=1)

time = data[:,0]
dt   = data[:,1]
lat  = data[:,2]
fwd  = data[:,3]

# Convert to relative time (nicer plots)
time = time - time[0]

plt.figure()
plt.plot(time, lat)
plt.xlabel("Time [s]")
plt.ylabel("Lateral error [px]")
plt.title("Lateral Tracking Error")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(time, fwd)
plt.xlabel("Time [s]")
plt.ylabel("Forward error [m]")
plt.title("Forward Tracking Error")
plt.grid(True)
plt.show()
