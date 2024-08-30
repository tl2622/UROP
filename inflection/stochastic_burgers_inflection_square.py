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
noise_amplitude = 0.02  # Amplitude of the trigonometric noise

# Discretized space and time
x = np.linspace(0, L, N)
t = np.linspace(0, T, M)

# Gaussian initial condition
u0 = np.exp(-((x - np.pi) ** 2) / 0.1)

# Initialize the solution matrix
u = np.zeros((M, N))
u[0, :] = u0

# Function for periodic boundary conditions
def periodic_bc(u):
    u[0] = u[-2]
    u[-1] = u[1]
    return u

# Function to generate the indicator function noise
def generate_indicator_noise(x, amplitude):
    noise = np.zeros_like(x)
    indicator = (x >= np.pi/2) & (x <= 3*np.pi/2)
    noise[indicator] = amplitude * np.random.randn(np.sum(indicator))
    return noise

# List to store slopes at the inflection point
slopes_at_inflection = []

# Time-stepping loop for the stochastic Burgers equation
for n in range(0, M-1):
    # Compute the first and second derivatives
    u_x = (np.roll(u[n, :], -1) - np.roll(u[n, :], 1)) / (2 * dx)
    u_xx = (np.roll(u[n, :], -1) - 2 * u[n, :] + np.roll(u[n, :], 1)) / dx**2
    
    # Generate indicator function noise
    noise = generate_indicator_noise(x, noise_amplitude)
    
    # Update rule (explicit method)
    u[n+1, :] = u[n, :] - u[n, :] * u_x * dt + nu * u_xx * dt + noise * np.sqrt(dt)
    
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
plt.title('Variation of Slope at Inflection Point Over Time:square')
plt.grid(True)
plt.savefig('Variation of Slope at Inflection Point Over Time:square.pdf')
plt.savefig('Variation of Slope at Inflection Point Over Time:square.png', dpi=200)