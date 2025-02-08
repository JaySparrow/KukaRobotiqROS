import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the waypoints
waypoints = [
    np.array([0., 0., 0.]),
    np.array([0., -1., 0.]),
    np.array([0., -1., -0.5]),
    np.array([0., 0., -0.5]),
    np.array([0., 1., -0.5]),
    np.array([0., 1., -1.]),
    np.array([0., 0., -1.]),

    np.array([0., -1., -1.]),
    np.array([0., -1., -0.5]),
    np.array([0., 0., -0.5]),
    np.array([0., 1., -0.5]),
    np.array([0., 1., 0.]),
    np.array([0., 0., 0.]),
]

# Convert waypoints to numpy array for easier slicing
waypoints = np.array(waypoints)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot waypoints
ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='r', marker='o', label='Waypoints')

# Add arrows for direction
for i in range(len(waypoints) - 1):
    ax.quiver(
        waypoints[i, 0], waypoints[i, 1], waypoints[i, 2],
        waypoints[i+1, 0] - waypoints[i, 0],
        waypoints[i+1, 1] - waypoints[i, 1],
        waypoints[i+1, 2] - waypoints[i, 2],
        color='b', arrow_length_ratio=0.1
    )

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Search Pattern')

# Show the plot
plt.show()
