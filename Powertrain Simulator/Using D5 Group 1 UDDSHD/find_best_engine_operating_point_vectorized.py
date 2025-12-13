import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt

def find_best_engine_operating_point_vectorized(ENG, GEN, Peng_req, plot_flag=False):
    Peng_req = np.atleast_1d(Peng_req).flatten()
    N = Peng_req.size

    Torq_rng  = np.array(ENG.Torq_range).flatten()
    speed_rng = np.array(ENG.speed_range).flatten()
    
    we_grid = ENG.speed_range
    Te_grid = ENG.Torq_range

    # Torque limits
    Tmax_engine_interp = interp1d(ENG.maxTrq_w, ENG.max_torque_v, kind='linear', fill_value='extrapolate')
    Tmax_gen_interp = interp1d(GEN.maxTrqCrv_w, GEN.maxTrqCrv_trq_peak, kind='linear', fill_value='extrapolate')

    Tmax_engine = Tmax_engine_interp(we_grid)
    Tmax_gen = Tmax_gen_interp(we_grid)
    Tmax_both = np.minimum(Tmax_engine, Tmax_gen)

    # Preallocate outputs
    best_we     = np.zeros(N)
    best_Te     = np.zeros(N)
    best_mf     = np.zeros(N)
    best_Peng   = np.zeros(N)
    Pgen_output = np.zeros(N)
    gen_eff     = np.zeros(N)

    # Fuel map interpolator
    mf_interp = RegularGridInterpolator((we_grid, Te_grid), ENG.mf, bounds_error=False, fill_value=np.inf)
    # mf_interp = RegularGridInterpolator((we_grid, Te_grid), ENG.mf.T, bounds_error=False, fill_value=np.inf)

    # Precompute max map point
    Pmax_map = we_grid * Tmax_both
    idx_map_max = np.argmax(Pmax_map)
    we_map_max = we_grid[idx_map_max]
    Te_map_max = Tmax_both[idx_map_max]
    mf_map_max = mf_interp((we_map_max, Te_map_max))

    for k in range(N):
        P_req = Peng_req[k]

        mf_cand = np.full_like(we_grid, np.inf)
        Te_cand = np.full_like(we_grid, np.nan)

        for i, w in enumerate(we_grid):
            if w == 0:
                continue  # Avoid division by zero
            T_req = P_req / w
            if 0 <= T_req <= Tmax_both[i]:
                mf_cand[i] = mf_interp((w, T_req))
                Te_cand[i] = T_req

        if np.all(np.isinf(mf_cand)):
            # Above max power — use max map point
            best_we[k]   = we_map_max
            best_Te[k]   = Te_map_max
            best_mf[k]   = mf_map_max
            best_Peng[k] = Pmax_map[idx_map_max]
        else:
            idx_min = np.argmin(mf_cand)
            best_we[k]   = we_grid[idx_min]
            best_Te[k]   = Te_cand[idx_min]
            best_mf[k]   = mf_cand[idx_min]
            best_Peng[k] = best_we[k] * best_Te[k]

        # Generator efficiency
        gen_eff_interp = RegularGridInterpolator(
            (GEN.effMapPos_w, GEN.effMapPos_trq),
            GEN.effMapPos_eff.T,
            bounds_error=False, fill_value=np.nan
        )
        eff = gen_eff_interp((best_we[k], best_Te[k]))
        if np.isnan(eff) or eff <= 0 or eff > 1.5:
            gen_eff[k] = 0
            Pgen_output[k] = 0
        else:
            gen_eff[k] = eff
            Pgen_output[k] = eff * best_Peng[k]

    # Optional plotting
    if plot_flag:
        for k in range(N):
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))

            # Engine fuel map
            cs1 = axs[0].contourf(we_grid, Te_grid, ENG.mf.T, levels=20)
            fig.colorbar(cs1, ax=axs[0])
            axs[0].set_title(f"Engine Fuel Map (P_req = {Peng_req[k]:.1f} W)")
            axs[0].set_xlabel('ω_eng (rad/s)')
            axs[0].set_ylabel('T_eng (Nm)')
            if best_Peng[k] > 0:
                axs[0].plot(best_we[k], best_Te[k], 'ro', markersize=8, linewidth=2)
            else:
                axs[0].text(np.mean(we_grid), np.mean(Te_grid), 'Engine Off', color='red', fontsize=12, ha='center')

            # Generator efficiency map
            Wg, Tg = np.meshgrid(GEN.effMapPos_w, GEN.effMapPos_trq)
            cs2 = axs[1].contourf(Wg, Tg, GEN.effMapPos_eff, levels=20)
            fig.colorbar(cs2, ax=axs[1])
            axs[1].set_title(f"Gen Efficiency Map (P_req = {Peng_req[k]:.1f} W)")
            axs[1].set_xlabel('ω_gen (rad/s)')
            axs[1].set_ylabel('T_gen (Nm)')
            if best_Peng[k] > 0:
                axs[1].plot(best_we[k], best_Te[k], 'ro', markersize=8, linewidth=2)
            else:
                axs[1].text(np.mean(GEN.effMapPos_w), np.mean(GEN.effMapPos_trq),
                            'Engine Off', color='red', fontsize=12, ha='center')

            plt.tight_layout()
            plt.show()

    # best_Peng = best_Peng*41666.6666667*10
    # Pgen_output = Pgen_output*41666.6666667*10*1e3*10

    return best_we, best_Te, best_mf, best_Peng, Pgen_output, gen_eff
