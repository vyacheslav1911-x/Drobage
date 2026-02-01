import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration: setpoints
# -----------------------------
lateral_setpoint = 0.0      # pixels (center of image)
forward_setpoint = 0.3      # meters (desired distance)

# -----------------------------
# Load CSV
# -----------------------------
# CSV format: time_s, dt_s, lateral_error, forward_distance
data = np.loadtxt('tracking_log4.csv', delimiter=',', skiprows=1)

time_abs = data[:,0]       # original timestamps in seconds
dt       = data[:,1]       # delta time between samples in seconds
lat_meas = data[:,2]       # lateral error (pixels)
fwd_meas = data[:,3]       # forward distance (meters)

# -----------------------------
# Use relative time for plotting
# -----------------------------
time = time_abs - time_abs[0]  # start at 0 s

# -----------------------------
# Compute errors relative to setpoints
# -----------------------------
e_lat = lat_meas - lateral_setpoint
e_fwd = fwd_meas - forward_setpoint

# -----------------------------
# Cumulative ISE
# -----------------------------
cum_ise_lat = np.cumsum(e_lat**2 * dt)
cum_ise_fwd = np.cumsum(e_fwd**2 * dt)

# -----------------------------
# Cumulative RMSE
# -----------------------------
cum_rmse_lat = np.sqrt(np.cumsum(e_lat**2) / np.arange(1, len(e_lat)+1))
cum_rmse_fwd = np.sqrt(np.cumsum(e_fwd**2) / np.arange(1, len(e_fwd)+1))

# -----------------------------
# Print final overall values
# -----------------------------
print("Final RMSE lateral:", cum_rmse_lat[-1])
print("Final RMSE forward:", cum_rmse_fwd[-1])
print("Final ISE lateral:", cum_ise_lat[-1])
print("Final ISE forward:", cum_ise_fwd[-1])

# -----------------------------
# Plot 1: RMSE Lateral
# -----------------------------
plt.figure("RMSE Lateral")
plt.plot(time, cum_rmse_lat, color='skyblue')
plt.xlabel('Time [s]')
plt.ylabel('Cumulative RMSE [pixels]')
plt.title('Cumulative RMSE - Lateral')
plt.grid(True)
plt.tight_layout()

# -----------------------------
# Plot 2: RMSE Forward
# -----------------------------
plt.figure("RMSE Forward")
plt.plot(time, cum_rmse_fwd, color='salmon')
plt.xlabel('Time [s]')
plt.ylabel('Cumulative RMSE [m]')
plt.title('Cumulative RMSE - Forward')
plt.grid(True)
plt.tight_layout()

# -----------------------------
# Plot 3: ISE Lateral
# -----------------------------
plt.figure("ISE Lateral")
plt.plot(time, cum_ise_lat, color='skyblue')
plt.xlabel('Time [s]')
plt.ylabel('Cumulative ISE [pixels^2·s]')
plt.title('Cumulative ISE - Lateral')
plt.grid(True)
plt.tight_layout()

# -----------------------------
# Plot 4: ISE Forward
# -----------------------------
plt.figure("ISE Forward")
plt.plot(time, cum_ise_fwd, color='salmon')
plt.xlabel('Time [s]')
plt.ylabel('Cumulative ISE [m^2·s]')
plt.title('Cumulative ISE - Forward')
plt.grid(True)
plt.tight_layout()

plt.show()
