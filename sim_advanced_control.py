"""
Advanced Prescribed-Time Adaptive Control for 3rd-Order Strict-Feedback System.
Based on ADVANCED_REPORT.md.

Features:
1. Strict Prescribed-Time Scaling Function (k(t) -> inf as t -> T).
2. Smooth Adaptive Dead-zone Inverse.
3. Command Filter for derivative estimation.
4. Projection Operator for parameter adaptation.
5. Enhanced Koopman with Dead-zone features.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional

##############################################################################
#                               System & Utilities
##############################################################################

class ThirdOrderSystem:
    """
    Simulates the 3rd-order strict-feedback system:
    dx1 = x2 + f1(x1) + d1
    dx2 = x3 + f2(x2) + d2
    dx3 = u(v)
    """
    def __init__(self):
        # Barrier constraints parameters (example usage in Barrier Lyapunov Function)
        self.kb1_base, self.kb2_base, self.kb3_base = 0.5, 0.4, 0.6

    def f1(self, x1):
        # Unknown nonlinearity f1
        return np.sin(x1**2) * np.cos(x1)

    def f2(self, x2):
        # Unknown nonlinearity f2
        return np.sin(x2) * (x2 + x2**2)

    def d1(self, t):
        # Disturbance d1
        return 0.01 * np.sin(t)

    def d2(self, t):
        # Disturbance d2
        return 0.02 * np.sin(t)

    def get_constraints(self, t):
        # Time-varying constraints for BLF
        return (0.1*np.sin(t) + self.kb1_base,
                0.1*np.sin(t) + self.kb2_base,
                0.3*np.sin(t) + self.kb3_base)

    def deadzone(self, v):
        """
        Asymmetric Dead-zone: u = m*v + b
        Right: v >= 0.75, u = 1.2*v - 0.9
        Left: v < -0.5, u = 0.8*v + 0.4
        Dead: [-0.5, 0.75] -> 0
        """
        if v >= 0.75: return 1.2 * v - 0.9
        elif v < -0.5: return 0.8 * v + 0.4
        else: return 0.0

    def dynamics(self, t, x, v_cmd):
        x1, x2, x3 = x
        u = self.deadzone(v_cmd)
        dx1 = x2 + self.f1(x1) + self.d1(t)
        dx2 = x3 + self.f2(x2) + self.d2(t)
        dx3 = u
        return np.array([dx1, dx2, dx3])

class CommandFilter:
    """Second-order Command Filter to estimate derivatives and smooth signals."""
    def __init__(self, omega_n=50.0, zeta=0.8):
        self.omega_n = omega_n
        self.zeta = zeta
        self.x1 = 0.0 # Output alpha
        self.x2 = 0.0 # Output alpha_dot

    def step(self, u, dt):
        # x1_dot = x2
        # x2_dot = -2*zeta*wn*x2 - wn^2*(x1 - u)
        dx1 = self.x2
        dx2 = -2*self.zeta*self.omega_n*self.x2 - self.omega_n**2 * (self.x1 - u)

        self.x1 += dx1 * dt
        self.x2 += dx2 * dt
        return self.x1, self.x2

class StrictPrescribedTimeScaling:
    """
    Scaling function: mu(t) = (T / (T - t))^p
    Ensures convergence by time T.
    """
    def __init__(self, T, p=2.0, t_stop_ratio=0.99):
        self.T = T
        self.p = p
        # Stop scaling near T to avoid infinity in numerical simulation
        self.t_stop = T * t_stop_ratio

    def get_mu(self, t):
        t_eff = min(t, self.t_stop)
        denom = self.T - t_eff
        mu = (self.T / denom)**self.p
        mu_dot = self.p * (self.T**self.p) / (denom**(self.p + 1))
        return mu, mu_dot

class Projection:
    """Parameter Projection Operator to keep estimates within bounds."""
    @staticmethod
    def project(theta, d_theta, theta_min, theta_max, eps=0.1):
        if theta > theta_max and d_theta > 0:
            return 0.0
        if theta < theta_min and d_theta < 0:
            return 0.0
        return d_theta

##############################################################################
#                      Enhanced Koopman (With Deadzone Features)
##############################################################################

class KoopmanEstimator:
    """Data-driven estimator for system drift dynamics."""
    def __init__(self, n_obs=18, lambda_reg=1e-2):
        self.n_obs = n_obs
        self.lambda_reg = lambda_reg
        self.A_aug = None

    def lift(self, x, u=0.0):
        x1, x2, x3 = x
        # Add ReLU features to capture deadzone-like behaviors in dynamics if embedded
        base = [x1, x2, x3, x1**2, x2**2, x3**2, x1*x2, x2*x3,
                np.sin(x1), np.cos(x1), np.sin(x2), np.cos(x2), np.sin(x3),
                np.maximum(0, x1), np.maximum(0, -x1)] # ReLU states
        return np.array(base[:self.n_obs])

    def fit(self, X, Y, U):
        """Fit the Koopman operator using Least Squares."""
        Psi_X = np.array([self.lift(xi) for xi in X])
        Psi_Y = np.array([self.lift(yi) for yi in Y])
        U_col = U.reshape(-1, 1)
        # Phi = [Psi(X), U]
        Phi = np.hstack([Psi_X, U_col])
        # Solve A_aug * Phi.T = Psi_Y.T  =>  Psi_Y = Phi * A_aug.T
        # A_aug = (Phi.T * Phi + reg)^-1 * Phi.T * Psi_Y
        reg = self.lambda_reg * np.eye(Phi.shape[1])
        self.A_aug = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Psi_Y).T

    def predict_drift(self, x, dt):
        """Predict drift f(x) by setting u=0 (assuming linear input B*u structure)."""
        if self.A_aug is None: return np.zeros(3)
        psi = self.lift(x)
        phi = np.hstack([psi, [0.0]]) # u=0
        psi_next = self.A_aug @ phi
        x_next = psi_next[:3] # Assuming first 3 observables are the states x1,x2,x3
        return (x_next - x) / dt

##############################################################################
#                  Advanced Controller Implementation
##############################################################################

@dataclass
class AdvancedController:
    T_final: float = 5.0
    k_gains: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])

    # Adaptive Parameters [m_hat, b_hat] for Inverse Deadzone
    # u = m*v + b  => v = (u_des - b_hat) / m_hat
    m_hat: float = 1.0
    b_hat: float = 0.0

    koopman: Optional[KoopmanEstimator] = None

    def __post_init__(self):
        self.scaling = StrictPrescribedTimeScaling(self.T_final, p=2.0)
        self.cf1 = CommandFilter(omega_n=40.0)
        self.cf2 = CommandFilter(omega_n=40.0)
        # Inverse Barrier Lyapunov Function (Simple form)
        self.iblf = lambda x, k: x / ((k-0.01)**2 - x**2 + 1e-6)

    def compute_control(self, system, x, t, dt):
        x1, x2, x3 = x
        mu, mu_dot = self.scaling.get_mu(t)
        mu_safe = max(mu, 1e-3)

        kb1, kb2, kb3 = system.get_constraints(t)

        # --- Reference ---
        yd = 0.3 * np.sin(t)
        yd_dot = 0.3 * np.cos(t)

        # --- Step 1 ---
        z1 = x1 - yd
        xi1 = mu * z1
        bar1 = self.iblf(x1, kb1)

        # Drift estimation via Koopman
        f_est = self.koopman.predict_drift(x, dt) if self.koopman else np.zeros(3)
        f1_est = f_est[0] - x2 # f1 = dx1 - x2 (approx)

        # Virtual Control alpha1
        alpha1_raw = (-self.k_gains[0]*xi1 - bar1)/mu_safe - (mu_dot/mu_safe)*z1 - f1_est + yd_dot
        alpha1, d_alpha1 = self.cf1.step(alpha1_raw, dt)

        # --- Step 2 ---
        z2 = x2 - alpha1
        xi2 = mu * z2
        bar2 = self.iblf(x2, kb2)
        f2_est = f_est[1] - x3

        # Virtual Control alpha2
        alpha2_raw = (-self.k_gains[1]*xi2 - bar2)/mu_safe - (mu_dot/mu_safe)*z2 - f2_est + d_alpha1
        alpha2, d_alpha2 = self.cf2.step(alpha2_raw, dt)

        # --- Step 3 ---
        z3 = x3 - alpha2
        xi3 = mu * z3
        bar3 = self.iblf(x3, kb3)

        # Desired Control u_des
        total_des = -self.k_gains[2]*xi3 - bar3 - mu_dot*z3 + mu*d_alpha2
        u_des = total_des / mu_safe

        # --- Smooth Dead-zone Inverse ---
        # v = (u_des - b_hat) / m_hat
        # Simplified symmetric adaptation for demonstration
        v_cmd = (u_des - self.b_hat) / self.m_hat

        return v_cmd, {'xi3': xi3, 'u_des': u_des}

    def update_adaptation(self, debug, v_cmd, dt):
        """Update adaptive parameters m_hat and b_hat."""
        # Adaptation Law: dot_m = gamma * xi3 * v_cmd, dot_b = gamma * xi3
        gamma = 2.0
        xi3 = debug['xi3']
        # u_des = debug['u_des']

        dm = gamma * xi3 * v_cmd
        db = gamma * xi3 * 1.0

        # Projection to ensure parameters stay in feasible range
        self.m_hat += Projection.project(self.m_hat, dm, 0.5, 2.0) * dt
        self.b_hat += Projection.project(self.b_hat, db, -1.5, 1.5) * dt

        # Safety clamp
        self.m_hat = max(0.5, self.m_hat)


##############################################################################
#                                Simulation
##############################################################################

class DataGenerator:
    def __init__(self, system, dt=0.01):
        self.system = system
        self.dt = dt

    def generate(self, num_traj=50, steps=300):
        """Generate random trajectories for Koopman training."""
        X, Y, U = [], [], []
        for _ in range(num_traj):
            x = np.random.uniform(-0.4, 0.4, 3)
            phase = np.random.uniform(0, 2*np.pi)
            valid = True
            traj_X, traj_Y, traj_U = [], [], []

            for step in range(steps):
                t = step * self.dt
                # Rich excitation input
                v = 2.0*np.sin(0.5*t+phase) + 1.5*np.sin(1.2*t) + np.random.randn()
                v = float(np.clip(v, -8, 8))

                dx = self.system.dynamics(t, x, v)
                x_next = x + dx * self.dt

                if np.any(np.abs(x_next) > 3.0):
                    valid = False
                    break

                traj_X.append(x.copy())
                traj_Y.append(x_next.copy())
                traj_U.append(v)
                x = x_next

            if valid:
                X.extend(traj_X)
                Y.extend(traj_Y)
                U.extend(traj_U)

        return np.array(X), np.array(Y), np.array(U)

def run_advanced_simulation():
    dt = 0.01
    times = np.arange(0, 5.0, dt)
    sys = ThirdOrderSystem()

    # Train Koopman with Real Data
    print("Generating Data & Training Koopman...")
    gen = DataGenerator(sys, dt)
    X, Y, U = gen.generate(num_traj=50, steps=300)

    est = KoopmanEstimator()
    est.fit(X, Y, U)
    print("Koopman Fitted.")

    ctrl = AdvancedController(koopman=est)

    x = np.array([0.1, 0.1, 0.1])
    logs = {'x': [], 'v': [], 'm': [], 'b': []}

    for t in times:
        v_cmd, debug = ctrl.compute_control(sys, x, t, dt)
        v_cmd = float(np.clip(v_cmd, -20, 20))

        ctrl.update_adaptation(debug, v_cmd, dt)

        dx = sys.dynamics(t, x, v_cmd)
        x += dx * dt

        logs['x'].append(x)
        logs['v'].append(v_cmd)
        logs['m'].append(ctrl.m_hat)
        logs['b'].append(ctrl.b_hat)

    # Plotting
    logs['x'] = np.array(logs['x'])
    plt.figure(figsize=(12, 8))

    plt.subplot(2,2,1)
    plt.plot(times, logs['x'][:,0], label='x1')
    plt.plot(times, 0.3*np.sin(times), 'k--', label='ref')
    plt.title('Tracking x1 (Prescribed Time)')
    plt.xlabel('Time (s)')
    plt.ylabel('x1')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(times, logs['v'])
    plt.title('Control Input v')
    plt.xlabel('Time (s)')

    plt.subplot(2,2,3)
    plt.plot(times, logs['m'], label='m_hat')
    plt.plot(times, logs['b'], label='b_hat')
    plt.title('Adaptive Dead-zone Parameters')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.subplot(2,2,4)
    # Plot error
    plt.plot(times, logs['x'][:,0] - 0.3*np.sin(times), 'r', label='Error x1')
    plt.title('Tracking Error')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('advanced_control_results.png')
    print("Simulation Complete. Saved plot to 'advanced_control_results.png'.")

if __name__ == '__main__':
    run_advanced_simulation()
