import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2 * np.pi   # Length of the domain
T = 2.0         # Total time
N = 512         # Number of spatial grid points
M = 2000        # Number of time steps
nu = 0.1        # Viscosity
dx = L / N      # Spatial step size
dt = T / M      # Time step size
amplitude = 0.02  # Amplitude of the trigonometric noise

# Discretized space and time
x = np.linspace(0, L, N)
t = np.linspace(0, T, M)

# Gaussian initial condition
u0 = np.exp(-((x - np.pi) ** 2) / 0.1)

# Initialize the solution matrix
u = np.zeros((M, N))
u[0, :] = u0

# Spatial function ξ1(x)
xi_1 = 0.01 * np.sin(2 * np.pi * x / L)

# Function for periodic boundary conditions
def periodic_bc(u):
    u[0] = u[-2]
    u[-1] = u[1]
    return u

# Brownian increments (dW_t) for the noise
dW = np.sqrt(dt) * np.random.randn(M)

# List to store slopes at the inflection point
slopes_at_inflection = []

# Time-stepping loop for the stochastic Burgers equation
for n in range(0, M-1):
    # Compute the first and second derivatives
    u_x = (np.roll(u[n, :], -1) - np.roll(u[n, :], 1)) / (2 * dx)
    u_xx = (np.roll(u[n, :], -1) - 2 * u[n, :] + np.roll(u[n, :], 1)) / dx**2
    
    # Stochastic noise term ξ1(x) * u_x * dW_t
    noise = amplitude * xi_1 * u_x * dW[n]
    
    # Update rule (explicit method)
    u[n+1, :] = u[n, :] - u[n, :] * u_x * dt + nu * u_xx * dt + noise
    
    # Apply periodic boundary conditions
    u[n+1, :] = periodic_bc(u[n+1, :])
    
    # Identify the inflection point: where the second derivative changes sign
    sign_changes = np.where(np.diff(np.sign(u_xx)))[0]
    
    if len(sign_changes) > 0:
        # Take the first sign change (assuming a single inflection point)
        inflection_idx = sign_changes[0]
        
        # Linearly interpolate to get a more precise inflection point
        slope_at_inflection = u_x[inflection_idx] + \
                              (u_x[inflection_idx + 1] - u_x[inflection_idx]) * \
                              (-u_xx[inflection_idx] / (u_xx[inflection_idx + 1] - u_xx[inflection_idx]))
    else:
        # If no sign change is detected, use the previous inflection point
        slope_at_inflection = slopes_at_inflection[-1] if slopes_at_inflection else 0
    
    slopes_at_inflection.append(slope_at_inflection)

# Plot the variation of slope at the inflection point
plt.figure(figsize=(10, 6))
plt.plot(t[:-1], slopes_at_inflection, color='red')
plt.xlabel('Time')
plt.ylabel('Slope at Inflection Point')
plt.title('Variation of Slope at Inflection Point Over Time:trigonometric')
plt.grid(True)
plt.savefig('Variation of Slope at Inflection Point Over Time:trigonometric.pdf')
plt.savefig('Variation of Slope at Inflection Point Over Time:trigonometric.png', dpi=200)