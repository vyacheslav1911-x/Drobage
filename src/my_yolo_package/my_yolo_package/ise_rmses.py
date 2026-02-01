import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration: setpoints
# -----------------------------
lateral_setpoint = 0.0      # pixels
forward_setpoint = 0.3      # meters

# -----------------------------
# Load CSV
# -----------------------------
# CSV format: time_s, dt_s, lateral_error, forward_distance
data = np.loadtxt('tracking_log.csv', delimiter=',', skiprows=1)

time = data[:,0]        # seconds
dt   = data[:,1]        # Δt in seconds
lat_meas = data[:,2]    # lateral error (pixels)
fwd_meas = data[:,3]    # forward distance (meters)

# -----------------------------
# Compute errors relative to setpoint
# -----------------------------
e_lat = lat_meas - lateral_setpoint
e_fwd = fwd_meas - forward_setpoint

# -----------------------------
# Cumulative calculations
# -----------------------------
cum_ise_lat = np.cumsum(e_lat**2 * dt)
cum_ise_fwd = np.cumsum(e_fwd**2 * dt)

cum_rmse_lat = np.sqrt(np.cumsum(e_lat**2) / np.arange(1, len(e_lat)+1))
cum_rmse_fwd = np.sqrt(np.cumsum(e_fwd**2) / np.arange(1, len(e_fwd)+1))

# -----------------------------
# Print final results
# -----------------------------
print("Final RMSE lateral:", cum_rmse_lat[-1])
print("Final RMSE forward:", cum_rmse_fwd[-1])
print("Final ISE lateral:", cum_ise_lat[-1])
print("Final ISE forward:", cum_ise_fwd[-1])

# -----------------------------
# Plot cumulative curves
# -----------------------------
plt.figure(figsize=(10,5))

# RMSE
plt.subplot(1,2,1)
plt.plot(time, cum_rmse_lat, label='Lateral RMSE')
plt.plot(time, cum_rmse_fwd, label='Forward RMSE')
plt.xlabel('Time [s]')
plt.ylabel('Cumulative RMSE')
plt.title('Cumulative RMSE over time')
plt.grid(True)
plt.legend()

# ISE
plt.subplot(1,2,2)
plt.plot(time, cum_ise_lat, label='Lateral ISE')
plt.plot(time, cum_ise_fwd, label='Forward ISE')
plt.xlabel('Time [s]')
plt.ylabel('Cumulative ISE')
plt.title('Cumulative ISE over time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
