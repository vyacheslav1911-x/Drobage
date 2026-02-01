import numpy as np
import matplotlib.pyplot as plt
import sys

# Load CSV
csv_file = 'tracking_log4.csv'

try:
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
except Exception as e:
    sys.exit(f"Error loading CSV file: {e}")

if data.size == 0:
    sys.exit(f"CSV file '{csv_file}' is empty or contains no numeric data.")

# If only 1 row, reshape to 2D
if data.ndim == 1:
    data = data.reshape(1, -1)

# Extract columns
time = data[:,0]    # seconds
dt   = data[:,1]    # sampling interval (seconds)
lat  = data[:,2]    # lateral error
fwd  = data[:,3]    # forward error

# Make relative time
time_rel = time - time[0]

# Compute cumulative ISE
cum_ise_lat = np.cumsum(lat**2 * dt)
cum_ise_fwd = np.cumsum(fwd**2 * dt)

# Plot cumulative ISE
plt.figure(figsize=(8,4))
plt.plot(time_rel, cum_ise_lat, label='Cumulative ISE - Lateral')
plt.xlabel('Time [s]')
plt.ylabel('Cumulative ISE')
plt.title('Cumulative ISE – Lateral')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_rel, cum_ise_fwd, label='Cumulative ISE - Forward')
plt.xlabel('Time [s]')
plt.ylabel('Cumulative ISE')
plt.title('Cumulative ISE – Forward')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
