import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the functions for the equations
def eq1_rhs(x, y):
    return -2*x + y

def eq2_rhs(x, y, z):  # For second-order ODE
    return -2*z - y

def eq3_rhs(t, x, y):  # For system of ODEs
    dxdt = y
    dydt = 3*x - 4*y
    return dxdt, dydt

# Runge-Kutta 4th Order for a single equation
def rk4_single(f, x0, y0, h, n):
    x = x0
    y = y0
    results = [(x, y)]
    for i in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        results.append((x, y))
    return results

# Runge-Kutta 4th Order for a system of equations
def rk4_system(f, t0, x0, y0, h, n):
    t = t0
    x = x0
    y = y0
    results = [(t, x, y)]
    for i in range(n):
        k1x, k1y = f(t, x, y)
        k2x, k2y = f(t + h/2, x + h*k1x/2, y + h*k1y/2)
        k3x, k3y = f(t + h/2, x + h*k2x/2, y + h*k2y/2)
        k4x, k4y = f(t + h, x + h*k3x, y + h*k3y)
        x += h * (k1x + 2*k2x + 2*k3x + k4x) / 6
        y += h * (k1y + 2*k2y + 2*k3y + k4y) / 6
        t += h
        results.append((t, x, y))
    return results

# Analytical solutions
def eq1_analytical(x, C1=1):
    return (C1 + 2*(x + 1)*np.exp(-x)) * np.exp(x)

def eq2_analytical(x, C1=1, C2=1):
    return (C1 + C2*x) * np.exp(-x)

def eq3_analytical(t, C1=1, C2=1):
    sqrt_7 = np.sqrt(7)
    x = C1 * (2 - sqrt_7) * np.exp(-t * (2 + sqrt_7)) / 3 + C2 * (2 + sqrt_7) * np.exp(-t * (2 - sqrt_7)) / 3
    y = C1 * np.exp(-t * (2 + sqrt_7)) + C2 * np.exp(-t * (2 - sqrt_7))
    return x, y

# Parameters
x0, y0, t0 = 0, 1, 0
h = 0.1
n = 50  # Number of steps

# RK4 approximations
rk4_eq1 = rk4_single(eq1_rhs, x0, y0, h, n)
rk4_eq2 = rk4_single(lambda x, y: eq2_rhs(x, y, -y), x0, y0, h, n)  # Convert second-order to first-order
rk4_eq3 = rk4_system(eq3_rhs, t0, y0, y0, h, n)

# Generate data for analytical solutions
x_vals = np.arange(x0, x0 + (n+1)*h, h)
t_vals = np.arange(t0, t0 + (n+1)*h, h)

analytical_eq1 = [eq1_analytical(x) for x in x_vals]
analytical_eq2 = [eq2_analytical(x) for x in x_vals]
analytical_eq3 = [eq3_analytical(t) for t in t_vals]
analytical_eq3_x, analytical_eq3_y = zip(*analytical_eq3)

# Convert RK4 results to arrays for comparison
rk4_eq1_x, rk4_eq1_y = zip(*rk4_eq1)
rk4_eq2_x, rk4_eq2_y = zip(*rk4_eq2)
rk4_eq3_t, rk4_eq3_x, rk4_eq3_y = zip(*rk4_eq3)

# Create tables
table_eq1 = pd.DataFrame({'x': rk4_eq1_x, 'RK4': rk4_eq1_y, 'Analytical': analytical_eq1})
table_eq2 = pd.DataFrame({'x': rk4_eq2_x, 'RK4': rk4_eq2_y, 'Analytical': analytical_eq2})
table_eq3 = pd.DataFrame({'t': rk4_eq3_t, 'RK4_x': rk4_eq3_x, 'Analytical_x': analytical_eq3_x, 
                          'RK4_y': rk4_eq3_y, 'Analytical_y': analytical_eq3_y})

# Display tables
print("Equation 1 Results:")
print(table_eq1)
print("\nEquation 2 Results:")
print(table_eq2)
print("\nEquation 3 Results:")
print(table_eq3)

# Plot solutions
plt.figure(figsize=(12, 8))

# Plot Equation 1
plt.subplot(3, 1, 1)
plt.plot(rk4_eq1_x, rk4_eq1_y, 'b--', label='RK4 Approximation')
plt.plot(x_vals, analytical_eq1, 'r-', label='Analytical Solution')
plt.title('Equation 1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Plot Equation 2
plt.subplot(3, 1, 2)
plt.plot(rk4_eq2_x, rk4_eq2_y, 'b--', label='RK4 Approximation')
plt.plot(x_vals, analytical_eq2, 'r-', label='Analytical Solution')
plt.title('Equation 2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Plot Equation 3
plt.subplot(3, 1, 3)
plt.plot(rk4_eq3_t, rk4_eq3_x, 'b--', label='RK4 Approximation (x)')
plt.plot(rk4_eq3_t, rk4_eq3_y, 'g--', label='RK4 Approximation (y)')
plt.plot(t_vals, analytical_eq3_x, 'r-', label='Analytical Solution (x)')
plt.plot(t_vals, analytical_eq3_y, 'y-', label='Analytical Solution (y)')
plt.title('Equation 3')
plt.xlabel('t')
plt.ylabel('x, y')
plt.legend()

plt.tight_layout()
plt.show()
