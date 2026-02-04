import numpy as np
import matplotlib.pyplot as plt
import os
from sim_advanced_control_v2 import ControllerConfig, run_simulation

# Ensure output directory exists
OUTPUT_DIR = "experiments_v2"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_results(experiment_name, results_dict):
    """
    Generates the standardized 6-panel plot for comparison (V2).
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Experiment V2: {experiment_name}", fontsize=16)

    T_final = 5.0
    t_stop = T_final * 0.95

    # 1. Tracking: x1 vs yd
    ax = axs[0, 0]
    for case, logs in results_dict.items():
        t = logs['t']
        x1 = logs['x'][:, 0]
        ax.plot(t, x1, label=f"{case}", linewidth=1.2)
    t_ref = list(results_dict.values())[0]['t']
    yd = 0.3 * np.sin(t_ref)
    ax.plot(t_ref, yd, 'k--', label="Ref (yd)", linewidth=1.5)
    ax.set_title("Tracking Performance (x1)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)

    # 2. Tracking Error |z1|
    ax = axs[0, 1]
    for case, logs in results_dict.items():
        t = logs['t']
        z1 = np.abs(logs['z'][:, 0])
        ax.plot(t, z1, label=f"{case}", linewidth=1.2)

        rmse = np.sqrt(np.mean(z1**2))
        final_err = np.max(z1[t > 4.5]) if np.any(t>4.5) else 0.0
        print(f"[{experiment_name}] {case}: RMSE={rmse:.4f}, Final MaxErr={final_err:.4f}")

    ax.set_title("Tracking Error |z1| (Log Scale)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error Magnitude")
    ax.set_yscale('log')
    ax.grid(True)
    ax.axvline(x=T_final, color='r', linestyle='-', label='T')
    ax.axvline(x=t_stop, color='g', linestyle=':', label='t_stop')
    ax.legend(loc='upper right', fontsize='small')

    # 3. Prescribed Scaling Function mu(t)
    ax = axs[0, 2]
    for case, logs in results_dict.items():
        t = logs['t']
        mu = logs['mu']
        ax.plot(t, mu, label=f"{case}", linewidth=1.2)
    ax.set_title("Scaling Function mu(t)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gain")
    ax.set_yscale('log')
    ax.axvline(x=T_final, color='r', linestyle='-')
    ax.axvline(x=t_stop, color='g', linestyle=':')
    ax.grid(True)

    # 4. Transformed Error xi
    ax = axs[1, 0]
    for case, logs in results_dict.items():
        t = logs['t']
        xi1 = logs['xi'][:, 0]
        ax.plot(t, xi1, label=f"{case}", linewidth=1.0)
    ax.set_title("Transformed Error xi1 (Symlog)")
    ax.set_xlabel("Time (s)")
    ax.set_yscale('symlog', linthresh=1.0)
    ax.grid(True)

    # 5. Control Inputs (u_act)
    ax = axs[1, 1]
    for case, logs in results_dict.items():
        t = logs['t']
        u_act = logs['u_act']
        ax.plot(t, u_act, label=f"{case}", linewidth=1.0)

        Ju = np.sum(u_act**2) * (t[1]-t[0])
        print(f"[{experiment_name}] {case}: Ju={Ju:.2f}")

    ax.set_title("Actuator Output u_act (Clipped)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-25, 25])
    ax.grid(True)

    # 6. Adaptive Parameters (m_hat)
    ax = axs[1, 2]
    for case, logs in results_dict.items():
        t = logs['t']
        m_hat = logs['m_hat']
        ax.plot(t, m_hat, label=f"{case}", linewidth=1.2)
    ax.set_title("Adaptive Parameter m_hat")
    ax.set_xlabel("Time (s)")
    # V2 Expectation: m_hat should stay within [0.5, 2.0]
    ax.set_ylim([0.4, 2.1])
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, f"{experiment_name.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.close()

def plot_deadzone_zoom(logs, title_suffix=""):
    """Special plot for Deadzone Map (Zoomed)"""
    plt.figure(figsize=(6, 5))
    v_cmd = logs['v_cmd']
    u_act = logs['u_act']

    if len(v_cmd) > 5000:
        idx = np.random.choice(len(v_cmd), 5000, replace=False)
        v_plot = v_cmd[idx]
        u_plot = u_act[idx]
    else:
        v_plot = v_cmd
        u_plot = u_act

    plt.scatter(v_plot, u_plot, s=2, alpha=0.5, c='b')
    plt.title(f"Deadzone Map {title_suffix} (Zoom)")
    plt.xlabel("Command v")
    plt.ylabel("Output u")
    # Zoom in to show the deadzone flat spot clearly
    plt.xlim([-2.0, 2.0])
    plt.ylim([-2.0, 2.0])
    plt.grid(True)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)

    save_path = os.path.join(OUTPUT_DIR, f"Exp_2_Deadzone_Map{title_suffix.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_constraints(logs_dict, experiment_name):
    """Special plot for Constraints"""
    plt.figure(figsize=(10, 6))

    case_on = list(logs_dict.keys())[0]
    logs = logs_dict[case_on]
    t = logs['t']
    x1 = logs['x'][:, 0]

    kb1_base = 0.5
    kb1 = 0.1*np.sin(t) + kb1_base
    limit = kb1 - 0.01

    plt.plot(t, x1, label=f"{case_on} x1", color='b')
    plt.plot(t, limit, 'r--', label='Constraint (+)')
    plt.plot(t, -limit, 'r--', label='Constraint (-)')

    for case, l in logs_dict.items():
        if case == case_on: continue
        plt.plot(l['t'], l['x'][:, 0], label=f"{case} x1", linestyle=':', alpha=0.7)

    plt.title(f"{experiment_name} - Constraint Compliance")
    plt.xlabel("Time (s)")
    plt.ylabel("State x1")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, f"{experiment_name.replace(' ', '_')}_Detail.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- Experiment Runners (Identical Scenarios to V1 for comparison) ---

def run_experiment_1():
    print("\n--- Running Experiment 1 (V2): Prescribed-Time Characteristics ---")
    cases = {}

    # Adding mu_max cap to avoid numerical ugliness in log plots, though patch 1 handles singularity.
    cfg_a = ControllerConfig(p_scaling=2.0, mu_max=1e5)
    cases['Case A (p=2)'] = run_simulation(cfg_a)

    cfg_b = ControllerConfig(p_scaling=1.0, mu_max=1e5)
    cases['Case B (p=1)'] = run_simulation(cfg_b)

    cfg_c = ControllerConfig(p_scaling=4.0, mu_max=1e5)
    cases['Case C (p=4)'] = run_simulation(cfg_c)

    cfg_d = ControllerConfig(scaling_enabled=False)
    cases['Case D (No Scale)'] = run_simulation(cfg_d)

    plot_results("Exp 1 Prescribed Time", cases)

def run_experiment_2():
    print("\n--- Running Experiment 2 (V2): Dead-zone Inverse & Adaptation ---")
    cases = {}

    # Reducing reference amplitude to make deadzone crossing more critical/visible
    # Wait, strict feedback tracking requires following Yd.
    # Can't easily change Yd inside run_simulation without hacking.
    # We will stick to standard ref, but rely on Zoom plot to show the effect.

    cfg_a = ControllerConfig()
    cases['Case A (Full)'] = run_simulation(cfg_a)

    cfg_b = ControllerConfig(adaptive_enabled=False)
    cases['Case B (No Adapt)'] = run_simulation(cfg_b)

    cfg_c = ControllerConfig(deadzone_inverse_enabled=False)
    cases['Case C (No Inv)'] = run_simulation(cfg_c)

    plot_results("Exp 2 Deadzone Adaptive", cases)
    plot_deadzone_zoom(cases['Case A (Full)'], "(Case A)")

def run_experiment_3():
    print("\n--- Running Experiment 3 (V2): Koopman Value ---")
    cases = {}

    cfg_a = ControllerConfig(koopman_enabled=True)
    cases['Case A (Koopman)'] = run_simulation(cfg_a)

    cfg_b = ControllerConfig(koopman_enabled=False)
    cases['Case B (No Koopman)'] = run_simulation(cfg_b)

    plot_results("Exp 3 Koopman", cases)

def run_experiment_4():
    print("\n--- Running Experiment 4 (V2): Command Filter ---")
    cases = {}

    cfg_a = ControllerConfig(cf_omega=40.0)
    cases['Case A (w=40)'] = run_simulation(cfg_a)

    cfg_c = ControllerConfig(cf_omega=120.0)
    cases['Case C (w=120)'] = run_simulation(cfg_c)

    cfg_d = ControllerConfig(cf_enabled=False)
    cases['Case D (No Filter)'] = run_simulation(cfg_d)

    plot_results("Exp 4 Command Filter", cases)

def run_experiment_5():
    print("\n--- Running Experiment 5 (V2): IBLF Constraints ---")
    cases = {}

    cfg_a = ControllerConfig(iblf_enabled=True)
    cases['Case A (Safe)'] = run_simulation(cfg_a)

    x0_risk = np.array([0.48, 0.1, 0.1])
    cfg_b = ControllerConfig(iblf_enabled=False)
    cases['Case B (Off, Risk)'] = run_simulation(cfg_b, x0=x0_risk)

    cfg_c = ControllerConfig(iblf_enabled=True)
    cases['Case C (On, Risk)'] = run_simulation(cfg_c, x0=x0_risk)

    plot_results("Exp 5 Constraints", cases)
    plot_constraints(cases, "Exp 5 Constraints")

def run_experiment_6():
    print("\n--- Running Experiment 6 (V2): Robustness ---")
    cases = {}

    cfg = ControllerConfig()

    cases['Level 1 (Normal)'] = run_simulation(cfg, sys_args={'disturbance_scale': 1.0})
    cases['Level 2 (5x Dist)'] = run_simulation(cfg, sys_args={'disturbance_scale': 5.0})
    cases['Level 3 (10x+Step)'] = run_simulation(cfg, sys_args={'disturbance_scale': 10.0, 'extra_disturbance': True})

    plot_results("Exp 6 Robustness", cases)

def main():
    np.random.seed(42)
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()
    run_experiment_6()
    print("\nAll V2 experiments completed.")

if __name__ == '__main__':
    main()
