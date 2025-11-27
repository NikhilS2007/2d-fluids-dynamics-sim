# fluid_sim.py
#
# Simple 2D "stable fluids" Navier–Stokes simulation on CPU
# - Uses Jos Stam style steps: add source → diffuse → project → advect → project
# - Visualizes density (like smoke) with matplotlib

import numpy as np
import matplotlib.pyplot as plt

class Fluid:
    def __init__(self, N, diffusion, viscosity, dt):
        self.N = N
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity

        size = (N + 2, N + 2)  # include boundary cells

        # density
        self.dens = np.zeros(size, dtype=np.float64)
        self.dens_prev = np.zeros_like(self.dens)

        # velocity
        self.u = np.zeros(size, dtype=np.float64)
        self.v = np.zeros(size, dtype=np.float64)
        self.u_prev = np.zeros_like(self.u)
        self.v_prev = np.zeros_like(self.v)

    def add_density(self, x, y, amount):
        self.dens[x, y] += amount

    def add_velocity(self, x, y, amount_u, amount_v):
        self.u[x, y] += amount_u
        self.v[x, y] += amount_v

    def step(self):
        N = self.N
        visc = self.visc
        diff = self.diff
        dt = self.dt

        # Velocity step
        self.u_prev, self.u = self.u, self.u_prev
        self.v_prev, self.v = self.v, self.v_prev

        self.diffuse(1, self.u, self.u_prev, visc, dt)
        self.diffuse(2, self.v, self.v_prev, visc, dt)
        self.project(self.u, self.v, self.u_prev, self.v_prev)

        self.u_prev, self.u = self.u, self.u_prev
        self.v_prev, self.v = self.v, self.v_prev

        self.advect(1, self.u, self.u_prev, self.u_prev, self.v_prev, dt)
        self.advect(2, self.v, self.v_prev, self.u_prev, self.v_prev, dt)
        self.project(self.u, self.v, self.u_prev, self.v_prev)

        # Density step
        self.dens_prev, self.dens = self.dens, self.dens_prev
        self.diffuse(0, self.dens, self.dens_prev, diff, dt)
        self.dens_prev, self.dens = self.dens, self.dens_prev
        self.advect(0, self.dens, self.dens_prev, self.u, self.v, dt)

    def diffuse(self, b, x, x0, diff, dt):
        N = self.N
        a = dt * diff * N * N
        self.lin_solve(b, x, x0, a, 1 + 4 * a)

    def advect(self, b, d, d0, u, v, dt):
        N = self.N
        dt0 = dt * N

        for i in range(1, N + 1):
            for j in range(1, N + 1):
                x = i - dt0 * u[i, j]
                y = j - dt0 * v[i, j]

                if x < 0.5:
                    x = 0.5
                if x > N + 0.5:
                    x = N + 0.5
                i0 = int(x)
                i1 = i0 + 1

                if y < 0.5:
                    y = 0.5
                if y > N + 0.5:
                    y = N + 0.5
                j0 = int(y)
                j1 = j0 + 1

                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1

                d[i, j] = (
                    s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                    s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
                )

        self.set_bnd(b, d)

    def project(self, u, v, p, div):
        N = self.N
        h = 1.0 / N

        for i in range(1, N + 1):
            for j in range(1, N + 1):
                div[i, j] = -0.5 * h * (
                    u[i + 1, j] - u[i - 1, j] +
                    v[i, j + 1] - v[i, j - 1]
                )
                p[i, j] = 0.0

        self.set_bnd(0, div)
        self.set_bnd(0, p)

        self.lin_solve(0, p, div, 1, 4)

        for i in range(1, N + 1):
            for j in range(1, N + 1):
                u[i, j] -= 0.5 * (p[i + 1, j] - p[i - 1, j]) / h
                v[i, j] -= 0.5 * (p[i, j + 1] - p[i, j - 1]) / h

        self.set_bnd(1, u)
        self.set_bnd(2, v)

    def lin_solve(self, b, x, x0, a, c):
        N = self.N
        for _ in range(20):  # Gauss–Seidel iterations
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    x[i, j] = (x0[i, j] + a * (
                        x[i - 1, j] + x[i + 1, j] +
                        x[i, j - 1] + x[i, j + 1]
                    )) / c
            self.set_bnd(b, x)

    def set_bnd(self, b, x):
        N = self.N
        for i in range(1, N + 1):
            x[0, i]     = -x[1, i]     if b == 1 else x[1, i]
            x[N + 1, i] = -x[N, i]     if b == 1 else x[N, i]
            x[i, 0]     = -x[i, 1]     if b == 2 else x[i, 1]
            x[i, N + 1] = -x[i, N]     if b == 2 else x[i, N]

        x[0, 0]         = 0.5 * (x[1, 0]     + x[0, 1])
        x[0, N + 1]     = 0.5 * (x[1, N + 1] + x[0, N])
        x[N + 1, 0]     = 0.5 * (x[N, 0]     + x[N + 1, 1])
        x[N + 1, N + 1] = 0.5 * (x[N, N + 1] + x[N + 1, N])


def run_demo():
    N = 64          # grid size (N x N); keep modest for weak laptop
    diff = 0.0001   # diffusion coefficient
    visc = 0.0001   # viscosity
    dt = 0.1        # timestep

    fluid = Fluid(N, diff, visc, dt)

    plt.ion()
    fig, ax = plt.subplots()
    img = ax.imshow(fluid.dens.T, cmap='magma', origin='lower', vmin=0, vmax=100)
    plt.colorbar(img, ax=ax)
    ax.set_title("2D Stable Fluid (Density)")

    # Simple interactive-ish loop: add smoke & velocity in the center
    cx, cy = N // 2, N // 2

    for frame in range(1000):
        # Add density (like smoke source)
        fluid.add_density(cx, cy, 50)

        # Add some swirling velocity
        fluid.add_velocity(cx, cy, 5, 2)

        # Step the simulation
        fluid.step()

        # Update visualization
        img.set_data(fluid.dens.T)
        ax.set_xlabel(f"Frame: {frame}")
        plt.pause(0.001)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_demo()
