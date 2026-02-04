# Digital Control Simulation — ARX Model + Saturation (Python)
# Reference: PROJECT_REPORT.md for mathematical details.

import numpy as np
import matplotlib.pyplot as plt

def run_simulation():
    # -----------------------
    # Sampling period
    # -----------------------
    Ts = 0.004  # 4 ms

    # -----------------------
    # Reference signal
    # -----------------------
    # 2 seconds simulation
    # Corresponds to MATLAB: t = 0:Ts:2
    t = np.arange(0, 2 + Ts, Ts)
    Ref = 5 * np.ones_like(t)                 # Step reference

    # -----------------------
    # ARX model coefficients
    # System Model: A(z)y(k) = B(z)u(k)
    # A(z) = 1 - 1.9366 z^-1 + 1.1523 z^-2 - 0.2144 z^-3
    # B(z) = -0.001961 z^-3
    # -----------------------
    # Note: Signs are stored such that y(k) = -sum(a[i]*y[k-i]) + ...
    # a[0] corresponds to y(k), a[1] to y(k-1), etc.
    a = np.array([1, -1.9366, 1.1523, -0.2144], dtype=float)
    b = np.array([0, 0, 0, -0.001961], dtype=float)           # B(z) with zeros for delays

    # -----------------------
    # Initialize variables
    # -----------------------
    y = np.zeros_like(t, dtype=float)

    # Past outputs: y(k-1), y(k-2), y(k-3)
    y1 = 0.0
    y2 = 0.0
    y3 = 0.0

    # Past inputs: u(k-1), u(k-2), u(k-3), u(k-4)
    u1 = 0.0
    u2 = 0.0
    u3 = 0.0
    u4 = 0.0

    # Past error: e(k-1)
    error1 = 0.0

    # Store control signal for plotting
    Usim = np.zeros_like(t, dtype=float)

    # -----------------------
    # PI controller parameters
    # -----------------------
    Kp = -11.6083  # proportional gain
    Ti = 0.1014    # integral time

    # Discretization (Tustin / Trapezoidal Method)
    # K0 = Kp * (1 + Ts / (2*Ti))
    # K1 = Kp * (-1 + Ts / (2*Ti))
    K0 = Kp + Kp * Ts / (2 * Ti)
    K1 = -Kp + Kp * Ts / (2 * Ti)

    # -----------------------
    # Saturation limits
    # -----------------------
    Umax = 100.0
    Umin = -100.0

    # -----------------------
    # Simulation loop
    # -----------------------
    for k in range(len(t)):
        # ARX model implementation
        # y(k) = 1.9366*y(k-1) - 1.1523*y(k-2) + 0.2144*y(k-3) - 0.001961*u(k-4)
        # Note: In the code below, -a[1] becomes +1.9366, matching the equation.
        y[k] = (-a[1] * y1
                -a[2] * y2
                -a[3] * y3
                + b[0] * u1
                + b[1] * u2
                + b[2] * u3
                + b[3] * u4)

        # Control error
        error = Ref[k] - y[k]

        # PI control (incremental form)
        # u(k) = u(k-1) + K0*e(k) + K1*e(k-1)
        u = u1 + K0 * error + K1 * error1

        # Saturation
        if u > Umax:
            u = Umax
        elif u < Umin:
            u = Umin

        # Update past values for next iteration
        # Shift history: y(k-2) -> y(k-3), y(k-1) -> y(k-2), y(k) -> y(k-1)
        y3, y2, y1 = y2, y1, y[k]
        # Shift history: u(k-3) -> u(k-4), ...
        u4, u3, u2, u1 = u3, u2, u1, u

        error1 = error
        Usim[k] = u

    return t, y, Usim, Ref

def main():
    t, y, Usim, Ref = run_simulation()

    # -----------------------
    # Plot results
    # -----------------------
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, y, 'b+', markersize=4, label='System Output')
    plt.plot(t, Ref, 'r--', label='Reference')
    plt.xlabel('Time (s)')
    plt.ylabel('Output y(k)')
    plt.legend()
    plt.title('ARX System Response')

    plt.subplot(2, 1, 2)
    plt.plot(t, Usim, 'k+', markersize=4)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Signal u(k)')
    plt.title('PI Control Signal with Saturation')

    plt.tight_layout()
    # Save the figure instead of just showing it, useful for headless verification
    plt.savefig('simulation_result.png')
    print("Simulation complete. Results saved to 'simulation_result.png'.")
    # plt.show() # Optional: Uncomment to see interactive plot

if __name__ == "__main__":
    main()
