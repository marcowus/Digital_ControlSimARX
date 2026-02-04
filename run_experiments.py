import numpy as np
import matplotlib.pyplot as plt
import os
from sim_advanced_control import ControllerConfig, run_simulation

# Ensure output directory exists
OUTPUT_DIR = "experiments"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_results(experiment_name, results_dict):
    """
    Generates the standardized 6-panel plot for comparison.
    results_dict: { 'Case Name': logs }
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Experiment: {experiment_name}", fontsize=16)

    # 1. Tracking: x1 vs yd
    ax = axs[0, 0]
    for case, logs in results_dict.items():
        t = logs['t']
        x1 = logs['x'][:, 0]
        ax.plot(t, x1, label=f"{case}")
    # Plot reference once
    t_ref = list(results_dict.values())[0]['t']
    yd = 0.3 * np.sin(t_ref)
    ax.plot(t_ref, yd, 'k--', label="Ref (yd)", linewidth=1.5)
    ax.set_title("Tracking Performance (x1)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
    ax.legend()
    ax.grid(True)

    # 2. Tracking Error |z1| (Log Scale option implies checking magnitude)
    ax = axs[0, 1]
    for case, logs in results_dict.items():
        t = logs['t']
        z1 = np.abs(logs['z'][:, 0])
        ax.plot(t, z1, label=f"{case}")
    ax.set_title("Tracking Error |z1|")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error Magnitude")
    ax.set_yscale('log')
    ax.grid(True)
    # Mark T
    T = 5.0
    ax.axvline(x=T, color='r', linestyle=':', label='T')

    # 3. Prescribed Scaling Function mu(t)
    ax = axs[0, 2]
    for case, logs in results_dict.items():
        t = logs['t']
        mu = logs['mu']
        ax.plot(t, mu, label=f"{case}")
    ax.set_title("Scaling Function mu(t)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gain")
    ax.set_yscale('log')
    ax.axvline(x=T, color='r', linestyle=':')
    ax.grid(True)

    # 4. Transformed Error xi (Stability check)
    ax = axs[1, 0]
    for case, logs in results_dict.items():
        t = logs['t']
        # plotting norm of xi vector or just xi1
        xi1 = logs['xi'][:, 0]
        ax.plot(t, xi1, label=f"{case}")
    ax.set_title("Transformed Error xi1")
    ax.set_xlabel("Time (s)")
    ax.grid(True)

    # 5. Control Inputs (u_act vs v_cmd)
    ax = axs[1, 1]
    # To avoid clutter, only plot the first case or specific ones if needed.
    # Here we plot u_act for all to see saturation/chattering
    for case, logs in results_dict.items():
        t = logs['t']
        u_act = logs['u_act']
        ax.plot(t, u_act, label=f"{case} u_act")
    ax.set_title("Actuator Output u_act")
    ax.set_xlabel("Time (s)")
    ax.grid(True)

    # 6. Adaptive Parameters (m_hat)
    ax = axs[1, 2]
    for case, logs in results_dict.items():
        t = logs['t']
        m_hat = logs['m_hat']
        ax.plot(t, m_hat, label=f"{case} m_hat")
    ax.set_title("Adaptive Parameter m_hat")
    ax.set_xlabel("Time (s)")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, f"{experiment_name.replace(' ', '_')}.png")
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

def run_experiment_1():
    print("Running Experiment 1: Prescribed-Time Characteristics...")
    cases = {}

    # Case A: Default (p=2)
    cfg_a = ControllerConfig(p_scaling=2.0)
    cases['Case A (p=2)'] = run_simulation(cfg_a)

    # Case B: Mild (p=1)
    cfg_b = ControllerConfig(p_scaling=1.0)
    cases['Case B (p=1)'] = run_simulation(cfg_b)

    # Case C: Aggressive (p=4)
    cfg_c = ControllerConfig(p_scaling=4.0)
    cases['Case C (p=4)'] = run_simulation(cfg_c)

    # Case D: No Scaling
    cfg_d = ControllerConfig(scaling_enabled=False)
    cases['Case D (No Scale)'] = run_simulation(cfg_d)

    plot_results("Exp 1 Prescribed Time", cases)

def run_experiment_2():
    print("Running Experiment 2: Dead-zone Inverse & Adaptation...")
    cases = {}

    # Case A: Default (Inverse + Adaptive)
    cfg_a = ControllerConfig()
    cases['Case A (Full)'] = run_simulation(cfg_a)

    # Case B: No Adaptation (Fixed parameters)
    cfg_b = ControllerConfig(adaptive_enabled=False)
    cases['Case B (No Adapt)'] = run_simulation(cfg_b)

    # Case C: No Inverse (v = u_des)
    cfg_c = ControllerConfig(deadzone_inverse_enabled=False)
    cases['Case C (No Inv)'] = run_simulation(cfg_c)

    # Case D: Oracle (Not easily implemented without changing sim structure deeply, skipping for now or approximating)
    # We will focus on A/B/C comparisons which are most telling.

    plot_results("Exp 2 Deadzone Adaptive", cases)

    # Extra Plot for Deadzone: v_cmd vs u_act
    plt.figure(figsize=(6, 5))
    logs = cases['Case A (Full)']
    plt.scatter(logs['v_cmd'], logs['u_act'], s=1, alpha=0.5, label='Case A Data')
    plt.title("Deadzone Map (Case A)")
    plt.xlabel("Command v")
    plt.ylabel("Output u")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "Exp_2_Deadzone_Map.png"))
    plt.close()

def run_experiment_3():
    print("Running Experiment 3: Koopman Value...")
    cases = {}

    # Case A: Koopman On
    cfg_a = ControllerConfig(koopman_enabled=True)
    cases['Case A (Koopman)'] = run_simulation(cfg_a)

    # Case B: Koopman Off
    cfg_b = ControllerConfig(koopman_enabled=False)
    cases['Case B (No Koopman)'] = run_simulation(cfg_b)

    plot_results("Exp 3 Koopman", cases)

def run_experiment_4():
    print("Running Experiment 4: Command Filter...")
    cases = {}

    # Case A: Default omega=40
    cfg_a = ControllerConfig(cf_omega=40.0)
    cases['Case A (w=40)'] = run_simulation(cfg_a)

    # Case B: Slow omega=10
    cfg_b = ControllerConfig(cf_omega=10.0)
    cases['Case B (w=10)'] = run_simulation(cfg_b)

    # Case C: Fast omega=120
    cfg_c = ControllerConfig(cf_omega=120.0)
    cases['Case C (w=120)'] = run_simulation(cfg_c)

    # Case D: No Filter
    cfg_d = ControllerConfig(cf_enabled=False)
    cases['Case D (No Filter)'] = run_simulation(cfg_d)

    plot_results("Exp 4 Command Filter", cases)

def run_experiment_5():
    print("Running Experiment 5: IBLF Constraints...")
    cases = {}

    # Case A: IBLF On
    cfg_a = ControllerConfig(iblf_enabled=True)
    cases['Case A (IBLF On)'] = run_simulation(cfg_a)

    # Case B: IBLF Off
    cfg_b = ControllerConfig(iblf_enabled=False)
    cases['Case B (IBLF Off)'] = run_simulation(cfg_b)

    plot_results("Exp 5 Constraints", cases)

def run_experiment_6():
    print("Running Experiment 6: Robustness...")
    cases = {}

    cfg = ControllerConfig()

    # Level 1: Default
    cases['Level 1 (Normal)'] = run_simulation(cfg, sys_args={'disturbance_scale': 1.0})

    # Level 2: High Noise
    cases['Level 2 (5x Dist)'] = run_simulation(cfg, sys_args={'disturbance_scale': 5.0})

    # Level 3: Extreme + Step
    cases['Level 3 (10x+Step)'] = run_simulation(cfg, sys_args={'disturbance_scale': 10.0, 'extra_disturbance': True})

    plot_results("Exp 6 Robustness", cases)

def main():
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()
    run_experiment_6()
    print("All experiments completed.")

if __name__ == '__main__':
    main()
