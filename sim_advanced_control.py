"""
Advanced Prescribed-Time Adaptive Control for 3rd-Order Strict-Feedback System.
Refactored for Comprehensive Experimentation (v2).

Key Updates:
1. Fixed logging: Logs clipped v_app and actual u_app.
2. Fixed Scaling: mu_dot=0 when t >= t_stop.
3. Fixed Adaptive: Hard clipping for m_hat, b_hat; Split estimation (Right/Left).
4. Improved Koopman: Uses previous input v for prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional, Dict

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
    def __init__(self, disturbance_scale=1.0, extra_disturbance=False):
        self.kb1_base, self.kb2_base, self.kb3_base = 0.5, 0.4, 0.6
        self.disturbance_scale = disturbance_scale
        self.extra_disturbance = extra_disturbance

    def f1(self, x1):
        return np.sin(x1**2) * np.cos(x1)

    def f2(self, x2):
        return np.sin(x2) * (x2 + x2**2)

    def d1(self, t):
        return self.disturbance_scale * 0.01 * np.sin(t)

    def d2(self, t):
        val = self.disturbance_scale * 0.02 * np.sin(t)
        if self.extra_disturbance and t > 2.0:
            val += 0.5 # Step disturbance
        return val

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
    """Second-order Command Filter."""
    def __init__(self, omega_n=40.0, zeta=0.8, enabled=True):
        self.omega_n = omega_n
        self.zeta = zeta
        self.enabled = enabled
        self.x1 = 0.0 # Output alpha
        self.x2 = 0.0 # Output alpha_dot

    def step(self, u, dt):
        if not self.enabled:
            # Bypass filter: approximation needed outside or return u, 0 (if naive)
            # Better approach: Caller handles bypass if they want finite diff.
            # Here we just pass through if "disabled" effectively infinite bandwidth
            self.x1 = u
            self.x2 = 0.0 # No derivative info
            return u, 0.0

        dx1 = self.x2
        dx2 = -2*self.zeta*self.omega_n*self.x2 - self.omega_n**2 * (self.x1 - u)

        self.x1 += dx1 * dt
        self.x2 += dx2 * dt
        return self.x1, self.x2

class StrictPrescribedTimeScaling:
    """
    Scaling function: mu(t) = (T / (T - t))^p
    """
    def __init__(self, T, p=2.0, t_stop_ratio=0.95, enabled=True): # Changed default stop to 0.95
        self.T = T
        self.p = p
        self.t_stop = T * t_stop_ratio
        self.enabled = enabled

    def get_mu(self, t):
        if not self.enabled:
            return 1.0, 0.0

        # FIX: Ensure mu_dot is 0 when t >= t_stop
        if t >= self.t_stop:
            denom = self.T - self.t_stop
            mu = (self.T / denom)**self.p
            return mu, 0.0

        denom = self.T - t
        mu = (self.T / denom)**self.p
        mu_dot = self.p * (self.T**self.p) / (denom**(self.p + 1))
        return mu, mu_dot

class Projection:
    @staticmethod
    def project(theta, d_theta, theta_min, theta_max):
        if theta > theta_max and d_theta > 0:
            return 0.0
        if theta < theta_min and d_theta < 0:
            return 0.0
        return d_theta

##############################################################################
#                      Enhanced Koopman
##############################################################################

class KoopmanEstimator:
    def __init__(self, n_obs=18, lambda_reg=1e-2):
        self.n_obs = n_obs
        self.lambda_reg = lambda_reg
        self.A_aug = None

    def lift(self, x):
        x1, x2, x3 = x
        base = [x1, x2, x3, x1**2, x2**2, x3**2, x1*x2, x2*x3,
                np.sin(x1), np.cos(x1), np.sin(x2), np.cos(x2), np.sin(x3),
                np.maximum(0, x1), np.maximum(0, -x1)]
        return np.array(base[:self.n_obs])

    def fit(self, X, Y, U):
        Psi_X = np.array([self.lift(xi) for xi in X])
        Psi_Y = np.array([self.lift(yi) for yi in Y])
        U_col = U.reshape(-1, 1)
        Phi = np.hstack([Psi_X, U_col])
        reg = self.lambda_reg * np.eye(Phi.shape[1])
        self.A_aug = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Psi_Y).T

    def predict_drift(self, x, u_prev, dt):
        if self.A_aug is None: return np.zeros(3)
        psi = self.lift(x)
        phi = np.hstack([psi, [u_prev]]) # Use previous u for better prediction
        psi_next = self.A_aug @ phi
        x_next = psi_next[:3]
        return (x_next - x) / dt

##############################################################################
#                  Advanced Controller Implementation
##############################################################################

@dataclass
class ControllerConfig:
    T_final: float = 5.0
    p_scaling: float = 2.0
    scaling_enabled: bool = True

    cf_omega: float = 40.0
    cf_enabled: bool = True

    iblf_enabled: bool = True

    adaptive_enabled: bool = True
    deadzone_inverse_enabled: bool = True

    koopman_enabled: bool = True

    k_gains: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])

class AdvancedController:
    def __init__(self, config: ControllerConfig, koopman: Optional[KoopmanEstimator] = None):
        self.cfg = config
        self.koopman = koopman

        self.scaling = StrictPrescribedTimeScaling(
            self.cfg.T_final,
            p=self.cfg.p_scaling,
            enabled=self.cfg.scaling_enabled
        )

        self.cf1 = CommandFilter(omega_n=self.cfg.cf_omega, enabled=self.cfg.cf_enabled)
        self.cf2 = CommandFilter(omega_n=self.cfg.cf_omega, enabled=self.cfg.cf_enabled)

        # Split adaptive parameters (Right / Left)
        # Init close to true values (1.2, -0.9) and (0.8, 0.4) or neutral (1.0, 0.0)
        self.m_hat_r = 1.0
        self.b_hat_r = 0.0
        self.m_hat_l = 1.0
        self.b_hat_l = 0.0

        # Helper for finite difference fallback if CF disabled
        self.last_alpha1 = 0.0
        self.last_alpha2 = 0.0
        self.first_step = True
        self.v_prev = 0.0

    def iblf(self, x, k):
        if not self.cfg.iblf_enabled:
            return 0.0
        # Barrier function: x / ((k-eps)^2 - x^2)
        denom = (k - 0.01)**2 - x**2
        if denom < 1e-4: denom = 1e-4 # Avoid division by zero
        return x / denom

    def compute_control(self, system, x, t, dt):
        x1, x2, x3 = x
        mu, mu_dot = self.scaling.get_mu(t)
        mu_safe = max(mu, 1e-3)

        kb1, kb2, kb3 = system.get_constraints(t)

        # --- Reference (Using lower amplitude if needed for deadzone exp, handled by caller?)
        # For consistency with previous report, keep 0.3 but Exp2 runner might want lower.
        # Hardcoding 0.3 here for general case.
        yd = 0.3 * np.sin(t)
        yd_dot = 0.3 * np.cos(t)

        # --- Step 1 ---
        z1 = x1 - yd
        xi1 = mu * z1
        bar1 = self.iblf(x1, kb1)

        # Drift estimation
        f_est = np.zeros(3)
        if self.cfg.koopman_enabled and self.koopman:
            f_est = self.koopman.predict_drift(x, self.v_prev, dt)

        f1_est = f_est[0] - x2

        # Alpha1
        alpha1_raw = (-self.cfg.k_gains[0]*xi1 - bar1)/mu_safe - (mu_dot/mu_safe)*z1 - f1_est + yd_dot

        if self.cfg.cf_enabled:
            alpha1, d_alpha1 = self.cf1.step(alpha1_raw, dt)
        else:
            # Simple finite difference fallback
            alpha1 = alpha1_raw
            if self.first_step: d_alpha1 = 0.0
            else: d_alpha1 = (alpha1 - self.last_alpha1) / dt
            self.last_alpha1 = alpha1

        # --- Step 2 ---
        z2 = x2 - alpha1
        xi2 = mu * z2
        bar2 = self.iblf(x2, kb2)
        f2_est = f_est[1] - x3

        alpha2_raw = (-self.cfg.k_gains[1]*xi2 - bar2)/mu_safe - (mu_dot/mu_safe)*z2 - f2_est + d_alpha1

        if self.cfg.cf_enabled:
            alpha2, d_alpha2 = self.cf2.step(alpha2_raw, dt)
        else:
            alpha2 = alpha2_raw
            if self.first_step: d_alpha2 = 0.0
            else: d_alpha2 = (alpha2 - self.last_alpha2) / dt
            self.last_alpha2 = alpha2

        # --- Step 3 ---
        z3 = x3 - alpha2
        xi3 = mu * z3
        bar3 = self.iblf(x3, kb3)

        total_des = -self.cfg.k_gains[2]*xi3 - bar3 - mu_dot*z3 + mu*d_alpha2
        u_des = total_des / mu_safe

        # --- Inverse Dead-zone (Split) ---
        v_cmd = u_des
        if self.cfg.deadzone_inverse_enabled:
            if u_des >= 0:
                v_cmd = (u_des - self.b_hat_r) / self.m_hat_r
            else:
                v_cmd = (u_des - self.b_hat_l) / self.m_hat_l

        self.first_step = False

        # NOTE: Caller (run_simulation) will clip v_cmd and call update_adaptation
        # Debug info should reflect 'pre-clip' intent, but update_adaptation needs executed command.

        debug = {
            't': t,
            'mu': mu, 'mu_dot': mu_dot,
            'yd': yd, 'yd_dot': yd_dot,
            'z': (z1, z2, z3),
            'xi': (xi1, xi2, xi3),
            'alpha1_info': (alpha1_raw, alpha1, d_alpha1),
            'alpha2_info': (alpha2_raw, alpha2, d_alpha2),
            'bar': (bar1, bar2, bar3),
            'f_est': f_est,
            'u_des': u_des,
            'v_raw': v_cmd, # Unclipped
            'm_hat_r': self.m_hat_r,
            'b_hat_r': self.b_hat_r,
            'm_hat_l': self.m_hat_l,
            'b_hat_l': self.b_hat_l
        }

        return v_cmd, debug

    def update_adaptation(self, debug, v_app, dt):
        if not self.cfg.adaptive_enabled:
            return

        # Use applied v (v_app) to determine which side we are on
        # Update law: dot_m = gamma * xi3 * v, dot_b = gamma * xi3
        gamma = 2.0
        xi3 = debug['xi'][2]

        dm = gamma * xi3 * v_app
        db = gamma * xi3 * 1.0

        # Split Update
        if v_app >= 0: # Right side
             self.m_hat_r += Projection.project(self.m_hat_r, dm, 0.5, 2.0) * dt
             self.b_hat_r += Projection.project(self.b_hat_r, db, -1.5, 1.5) * dt
             # Hard Clip
             self.m_hat_r = np.clip(self.m_hat_r, 0.5, 2.0)
             self.b_hat_r = np.clip(self.b_hat_r, -1.5, 1.5)
        else: # Left side
             self.m_hat_l += Projection.project(self.m_hat_l, dm, 0.5, 2.0) * dt
             self.b_hat_l += Projection.project(self.b_hat_l, db, -1.5, 1.5) * dt
             # Hard Clip
             self.m_hat_l = np.clip(self.m_hat_l, 0.5, 2.0)
             self.b_hat_l = np.clip(self.b_hat_l, -1.5, 1.5)

        # Store for next step Koopman prediction
        self.v_prev = v_app


##############################################################################
#                                Simulation
##############################################################################

class DataGenerator:
    def __init__(self, system, dt=0.01):
        self.system = system
        self.dt = dt

    def generate(self, num_traj=50, steps=300):
        X, Y, U = [], [], []
        for _ in range(num_traj):
            x = np.random.uniform(-0.4, 0.4, 3)
            phase = np.random.uniform(0, 2*np.pi)
            valid = True
            traj_X, traj_Y, traj_U = [], [], []

            for step in range(steps):
                t = step * self.dt
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

def run_simulation(config: ControllerConfig, sys_args: Dict = None, x0: np.ndarray = None):
    if sys_args is None: sys_args = {}
    dt = 0.002 # Reduced dt for better stability (Euler)
    times = np.arange(0, config.T_final, dt)

    # Setup System
    sys = ThirdOrderSystem(**sys_args)

    # Train Koopman (always needed if enabled in config)
    est = None
    if config.koopman_enabled:
        # Generate generic training data
        gen_sys = ThirdOrderSystem() # Standard system for training
        gen = DataGenerator(gen_sys, dt=dt) # Train on matched dt
        X, Y, U = gen.generate(num_traj=30, steps=200) # Reduced for speed
        est = KoopmanEstimator()
        est.fit(X, Y, U)

    ctrl = AdvancedController(config, koopman=est)

    # Initial Condition
    if x0 is None:
        x = np.array([0.1, 0.1, 0.1])
    else:
        x = np.array(x0)

    # Logging
    logs = {k: [] for k in ['t', 'x', 'mu', 'xi', 'z', 'u_des', 'v_cmd', 'u_act', 'm_hat', 'b_hat']}

    for t in times:
        v_raw, debug = ctrl.compute_control(sys, x, t, dt)

        # FIX: Clip immediately and use this for dynamics AND logging
        v_app = float(np.clip(v_raw, -20, 20))
        u_app = sys.deadzone(v_app)

        ctrl.update_adaptation(debug, v_app, dt)

        dx = sys.dynamics(t, x, v_app)
        x += dx * dt

        # Store Data
        logs['t'].append(t)
        logs['x'].append(x.copy())
        logs['mu'].append(debug['mu'])
        logs['xi'].append(debug['xi'])
        logs['z'].append(debug['z'])
        logs['u_des'].append(debug['u_des'])
        logs['v_cmd'].append(v_app) # LOGGING FIX: Log applied v
        logs['u_act'].append(u_app) # LOGGING FIX: Log actual u

        # Log average m_hat for simplified plotting, or specific one?
        # Plotting usually expects one line, let's log the one active or average
        if v_app >= 0:
            logs['m_hat'].append(debug['m_hat_r'])
            logs['b_hat'].append(debug['b_hat_r'])
        else:
            logs['m_hat'].append(debug['m_hat_l'])
            logs['b_hat'].append(debug['b_hat_l'])

    # Convert to arrays
    for k in logs:
        logs[k] = np.array(logs[k])

    return logs

if __name__ == '__main__':
    # Quick test run
    cfg = ControllerConfig()
    logs = run_simulation(cfg)
    print("Test simulation complete. Data points:", len(logs['t']))
