import numpy as np
import matplotlib.pyplot as plt

# Objective Function
def f(x):
    return 100 * (x[1] - x[0])**2 + (1 - x[0])**2

def grad_f(x):
    df_dx1 = -200 * (x[1] - x[0]) - 2 * (1 - x[0])
    df_dx2 = 200 * (x[1] - x[0])
    return np.array([df_dx1, df_dx2])

def hess_f(x):
    # Convex quadratic function has a constant Hessian
    return np.array([[202, -200],
                     [-200, 200]])

def backtracking_line_search(x, d, f_val, g_val, alpha_init=1.0, rho=0.25, beta1=0.5):
    alpha = alpha_init
    while f(x + alpha * d) > f_val + alpha * beta1 * np.dot(g_val, d):
        alpha = rho * alpha
    return alpha

def gradient_descent(x0, tol=1e-5):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    while True:
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        d = -g
        alpha = backtracking_line_search(x, d, f(x), g)
        x = x + alpha * d
        history.append(x.copy())
    return np.array(history)

def newtons_method(x0, tol=1e-5):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    while True:
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess_f(x)
        d = np.linalg.solve(H, -g)
        alpha = backtracking_line_search(x, d, f(x), g)
        x = x + alpha * d
        history.append(x.copy())
    return np.array(history)

def conjugate_gradient(x0, tol=1e-5):
    """Using Fletcher-Reeves Non-Linear Conjugate Gradient"""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    g = grad_f(x)
    d = -g
    while True:
        if np.linalg.norm(g) < tol:
            break
        alpha = backtracking_line_search(x, d, f(x), g)
        x_new = x + alpha * d
        g_new = grad_f(x_new)
        
        # Fletcher-Reeves formula for beta
        beta = np.dot(g_new, g_new) / np.dot(g, g)
        d_new = -g_new + beta * d
        
        x = x_new
        g = g_new
        d = d_new
        history.append(x.copy())
    return np.array(history)

## EXECUTION & PLOTTING CONFIGURATION
if __name__ == "__main__":
    x0 = [2.0, 5.0]
    history_gd = gradient_descent(x0)
    history_nm = newtons_method(x0)
    history_cg = conjugate_gradient(x0)

    print(f"Gradient Descent iterations: {len(history_gd)-1}")
    print(f"Newton's Method iterations: {len(history_nm)-1}")
    print(f"Conjugate Gradient iterations: {len(history_cg)-1}")

    x_star = np.array([1.0, 1.0])
    x_gd = history_gd[-1]
    x_nm = history_nm[-1]
    x_cg = history_cg[-1]

    print("\nFinal approximations:")
    print("Gradient Descent x =", x_gd)
    print("Newton's Method x =", x_nm)
    print("Conjugate Gradient x =", x_cg)

    err_gd = np.linalg.norm(x_gd - x_star)
    err_nm = np.linalg.norm(x_nm - x_star)
    err_cg = np.linalg.norm(x_cg - x_star)

    print("\nErrors ||x_k - x*||:")
    print(f"Gradient Descent error: {err_gd:.2e}")
    print(f"Newton's Method error: {err_nm:.2e}")
    print(f"Conjugate Gradient error: {err_cg:.2e}")

    # Meshgrid setup for plotting objective function vs optimization method
    x_vals = np.linspace(-1, 3, 400)
    y_vals = np.linspace(-1, 6, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 100 * (Y - X)**2 + (1 - X)**2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Optimization Trajectories by Method", fontsize=16)
    plot_data = [
        (axes[0], history_gd, "Gradient Descent", 'C0', 'o-'),
        (axes[1], history_nm, "Newton's Method", 'C1', 's-'),
        (axes[2], history_cg, "Conjugate Gradient", 'C2', '.-')
    ]

    for ax, history, title, color, fmt in plot_data:
        ax.contour(X, Y, Z, levels=np.logspace(-1, 4, 30), cmap='viridis', alpha=0.5)
        ax.plot(history[:, 0], history[:, 1], fmt, color=color, label=title, markersize=3)
        ax.plot(x0[0], x0[1], 'ro', label='Start $x_0=(2,5)$', markersize=6)
        ax.plot(1, 1, 'k*', label='Minimum $x^*=(1,1)$', markersize=10)
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('assets/optimization_trajectories.png', dpi=300)
    plt.show()
