import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def signo(val):
    if val > 0:
        return 1.0
    elif val < 0:
        return -1.0
    else:
        return 0.0

def robot_2gdl_ode(t, x):
    q1, q2, q1p, q2p = x
    qp = np.array([q1p, q2p])

    # Matriz de inercia M(q)
    M = np.array([
        [3.117 + 0.2 * np.cos(q2), 0.108 + 0.1 * np.cos(q2)],
        [0.108 + 0.1 * np.cos(q2), 0.108]
    ])

    # Matriz de fuerzas centripetas y de Coriolis C(q, qp)
    C = np.array([
        [-0.2 * np.sin(q2) * q2p, -0.1 * np.sin(q2) * q2p],
        [0.1 * np.sin(q2) * q1p,   0.0]
    ])

    # Vector de pares gravitacionales g(q)
    par_grav = np.array([
        39.3 * np.sin(q1) + 1.95 * np.sin(q1 + q2),
        1.95 * np.sin(q1 + q2)
    ])

    # Vector de friccion viscosa y de Coulomb f_f(qp)
    fr = np.array([
        1.86 * q1p + 1.93 * signo(q1p),
        0.16 * q2p + 0.30 * signo(q2p)
    ])

    # Pares aplicados tau(t)
    tau1 = (1.0 - np.exp(-0.8 * t)) * 32.0 + 56.0 * np.sin(16.0 * t + 0.1) + 12.0 * np.sin(20.0 * t + 0.15)
    tau2 = (1.0 - np.exp(-1.8 * t)) * 1.2  + 8.0  * np.sin(26.0 * t + 0.08) + 2.0  * np.sin(12.0 * t + 0.34)
    tau = np.array([tau1, tau2])

    # Aceleracion articular: q¨ = M^(-1) * (tau - C*q˙ - g - f_f)
    q2p_vec = np.linalg.inv(M) @ (tau - C @ qp - par_grav - fr)

    return [q1p, q2p, q2p_vec[0], q2p_vec[1]]

# Parametros de simulacion (10 segundos, paso 0.0025 s)
ti = 0.0
h = 0.0025
tf = 10.0
t_eval = np.arange(ti, tf + h, h)
x0 = [0.0, 0.0, 0.0, 0.0]

sol = solve_ivp(
    robot_2gdl_ode,
    t_span=(ti, tf),
    y0=x0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-3,
    atol=1e-6,
    first_step=2.5e-3,
    max_step=2.5e-3
)

q1_deg = np.rad2deg(sol.y[0])
q2_deg = np.rad2deg(sol.y[1])

plt.figure(figsize=(10, 5))
plt.plot(sol.t, q1_deg, label=r"$q_1$ (Hombro)")
plt.plot(sol.t, q2_deg, label=r"$q_2$ (Codo)")
plt.title("Respuesta de posicion articular — Robot planar de 2 gdl (10 s)")
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Posicion articular (grados)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()