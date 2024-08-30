import numpy as np
import matplotlib.pyplot as plt
import os
from moviepy.editor import ImageSequenceClip

# Parameters
L = 2 * np.pi
T = 2.0
N = 512
M = 2000
nu = 0.001
dx = L / N
dt = T / M
amplitude = 0.1

x = np.linspace(0, L, N)
t = np.linspace(0, T, M)

# Initial condition
u0 = np.exp(-((x - np.pi) ** 2) / 0.1)

u = np.zeros((M, N))
u[0, :] = u0

# Spatial function ξ1(x)
xi_1 = 0.01 * np.sin(2 * np.pi * x / L)

# Function for periodic boundary conditions
def periodic_bc(u):
    u[0] = u[-2]
    u[-1] = u[1]
    return u

frame_dir = "frames_stochastic_burgers"
os.makedirs(frame_dir, exist_ok=True)

# Brownian increments (dW_t) for the noise
dW = np.sqrt(dt) * np.random.randn(M)


for n in range(0, M-1):

    u_x = (np.roll(u[n, :], -1) - np.roll(u[n, :], 1)) / (2 * dx)
    u_xx = (np.roll(u[n, :], -1) - 2 * u[n, :] + np.roll(u[n, :], 1)) / dx**2
    
    # Stochastic noise term ξ1(x) * u_x * dW_t
    noise = amplitude * xi_1 * u_x * dW[n]
    
    # Update rule
    u[n+1, :] = u[n, :] - u[n, :] * u_x * dt + nu * u_xx * dt + noise
    
    # Apply periodic boundary conditions
    u[n+1, :] = periodic_bc(u[n+1, :])
    
    if n % 10 == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(x, u[n+1, :], label=f't = {t[n+1]:.3f}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.ylim(-1.5, 1.5)
        plt.title('Stochastic Burgers Equation with Transport Noise --- trigonometric case')
        plt.legend()
        plt.savefig(f"{frame_dir}/frame_{n:04d}.png")
        plt.close()

image_files = [f"{frame_dir}/frame_{n:04d}.png" for n in range(0, M-1, 10)]

clip = ImageSequenceClip(image_files, fps=24)
clip.write_videofile("stochastic_burgers_transport_noise_trigonometric.mp4", codec="libx264")

print("Video generated successfully.")