import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams.update({'font.size': 16})

linestyles = ['-', '--', '-.', ':', (0, (1, 1))]
markers = ['.', '+', 'x', '*', 'o']
colors = ['green', 'red', 'blue', 'orange', 'black']

plt.style.use('science')

names = ["TE-CDE", "TESAR-CDE (Two-step)", "TESAR-CDE (Multitask)"]

FIGS_PATH = os.path.join(os.path.dirname(__file__), "figures")


def generate_plot_gamma(x, y, y_std, names, xlim, filename):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=1000)
    for i in range(len(y)):
        # ax.errorbar(x, y[i], yerr=y_std[i], label=names[i],
        #              linestyle=linestyles[i], color=colors[i], marker=markers[i])
        ax.plot(x[0:xlim + 1], y[i, 0:xlim + 1], label=names[i], linestyle=linestyles[i], color=colors[i],
                marker=markers[i])
        plt.fill_between(x[0:xlim + 1], y[i, 0:xlim + 1] - y_std[i, 0:xlim + 1],
                         y[i, 0:xlim + 1] + y_std[i, 0:xlim + 1], alpha=0.1, color=colors[i], edgecolor=None)
    ax.set_ylim([0, None])
    leg = ax.legend(fontsize=12, frameon=False, facecolor='gray', framealpha=0.05, edgecolor='black')
    ax.set_xlabel('Sampling informativeness $\gamma$')
    ax.set_ylabel('RMSE')
    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_PATH, str(filename) + '.pdf'))
    plt.show()

gammas = np.array([0, 2, 4, 6, 8, 10, 12])

rmse_avg_baseline = [0.2556, 0.2632, 0.3346, 0.3463, 0.3464, 0.345]
rmse_avg_two_step = [0.2559, 0.2556, 0.2661, 0.3107, 0.3523, 0.379]
rmse_avg_multitask = [0.253, 0.2563, 0.257, 0.2802, 0.3153, 0.3375]
rmses_avg = np.array([rmse_avg_baseline, rmse_avg_two_step, rmse_avg_multitask])

rmse_avg_baseline_std = [0.01619, 0.01489, 0.01262, 0.01393, 0.01406, 0.01324]
rmse_avg_two_step_std = [0.01644, 0.01531, 0.01464, 0.01489, 0.01625, 0.01661]
rmse_avg_multitask_std = [0.01596, 0.0158, 0.01543, 0.01451, 0.01535, 0.0153]
rmses_avg_std = np.array([rmse_avg_baseline_std, rmse_avg_two_step_std, rmse_avg_multitask_std])


rmse_tau1_baseline = [0.1984, 0.2092, 0.2907, 0.3033, 0.3036, 0.3022]
rmse_tau1_two_step = [0.1981, 0.1996, 0.2139, 0.2655, 0.3119, 0.3405]
rmse_tau1_multitask = [0.1972, 0.1998, 0.2006, 0.2283, 0.2684, 0.293]
rmses_tau1 = np.array([rmse_tau1_baseline, rmse_tau1_two_step, rmse_tau1_multitask])

rmse_tau1_baseline_std = [0.01247, 0.01125, 0.01025, 0.01175, 0.01188, 0.01092]
rmse_tau1_two_step_std = [0.01236, 0.01151, 0.01091, 0.01211, 0.01454, 0.01609]
rmse_tau1_multitask_std = [0.01225, 0.01209, 0.01164, 0.01104, 0.0132, 0.01367]
rmses_tau1_std = np.array([rmse_tau1_baseline_std, rmse_tau1_two_step_std, rmse_tau1_multitask_std])


def generate_plot_tau(x, y, y_std, names, filename):
    fig, ax = plt.subplots(figsize=(7, 3), dpi=1000)
    for i in range(len(y)):
        ax.plot(x, y[i, :], label=names[i], linestyle=linestyles[i], color=colors[i],
                marker=markers[i])
        plt.fill_between(x, y[i, :] - y_std[i, :],
                         y[i, :] + y_std[i, :], alpha=0.1, color=colors[i], edgecolor=None)
    ax.set_ylim([0, None])
    leg = ax.legend(fontsize=12, frameon=False, facecolor='gray', framealpha=0.02, edgecolor='black', loc='lower right')
    ax.set_xlabel('Forecasting horizon $\\tau$')
    ax.set_ylabel('RMSE')
    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_PATH, str(filename) + '.pdf'))
    plt.show()

taus = np.array([1, 2, 3, 4, 5])

rmse_gamma6_baseline = [0.3033, 0.3311, 0.349, 0.3621, 0.3805]
rmse_gamma6_two_step = [0.2655, 0.2926, 0.3127, 0.3288, 0.3473]
rmse_gamma6_multitask = [0.2283, 0.2588, 0.2823, 0.3012, 0.3211]
rmses_gamma6 = np.array([rmse_gamma6_baseline, rmse_gamma6_two_step, rmse_gamma6_multitask])

rmse_gamma6_baseline_std = [0.01175, 0.01276, 0.014, 0.01513, 0.01615]
rmse_gamma6_two_step_std = [0.01211, 0.01369, 0.015, 0.01623, 0.01732]
rmse_gamma6_multitask_std = [0.01104, 0.01285, 0.01456, 0.01612, 0.0173]
rmses_gamma6_std = np.array([rmse_gamma6_baseline_std, rmse_gamma6_two_step_std, rmse_gamma6_multitask_std])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2), dpi=1000)
for i in range(len(rmses_tau1)):
    ax1.plot(gammas[0:-1], rmses_tau1[i, :], label=names[i], linestyle=linestyles[i], color=colors[i],
            marker=markers[i])
    ax1.fill_between(gammas[0:-1], rmses_tau1[i, :] - rmses_tau1_std[i, :],
                     rmses_tau1[i, :] + rmses_tau1_std[i, :], alpha=0.1, color=colors[i], edgecolor=None)
# ax1.grid(True, linewidth=1, alpha=0.6, linestyle='dotted', axis='y')
ax1.patch.set_facecolor("gray")
ax1.patch.set_alpha(0.025)
ax1.set_xticks(gammas[0:-1])
ax1.set_yticks([0.2, 0.25, 0.3, 0.25])
ax1.set_xlabel('Sampling informativeness $\gamma$')
ax1.set_ylabel('RMSE')
for i in range(len(rmses_gamma6)):
    ax2.plot(taus, rmses_gamma6[i, :], label=names[i], linestyle=linestyles[i], color=colors[i],
            marker=markers[i])
    ax2.fill_between(taus, rmses_gamma6[i, :] - rmses_gamma6_std[i, :],
                     rmses_gamma6[i, :] + rmses_gamma6_std[i, :], alpha=0.1, color=colors[i], edgecolor=None)
# ax2.set_ylim([0, None])
leg = ax2.legend(fontsize=14, frameon=False, facecolor='gray', framealpha=0.02, edgecolor='black',
                 bbox_to_anchor=(1.05, 0.5), loc='center left', labelspacing=1)
# ax2.grid(True, linewidth=1, alpha=0.6, linestyle='dotted', axis='y')
# ax2.patch.set_facecolor("gray")
# ax2.patch.set_alpha(0.025)
ax2.set_xticks(taus)
ax2.set_yticks([0.25, 0.3, 0.35])
ax2.set_xlabel('Forecasting horizon $\\tau$')
ax2.set_ylabel('RMSE')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_PATH, "ResultsMain.pdf"))
# fig.subplots_adjust(wspace=0.4)
plt.show()

##########################################
## Experiment with intensity covariates ##
##########################################

rmse_intcovAvg_baseline = [0.2635, 0.2635, 0.2635, 0.2634, 0.2631, 0.263]
rmse_intcovAvg_two_step = [0.2629, 0.2642, 0.2632, 0.2624, 0.2623, 0.2622]
rmse_intcovAvg_multitask = [0.2633, 0.2727, 0.2698, 0.2669, 0.266, 0.2653]
rmses_intcovAvg = np.array([rmse_intcovAvg_baseline, rmse_intcovAvg_two_step, rmse_intcovAvg_multitask])

rmse_intcovAvg_baseline_std = [0.009455, 0.009453, 0.009484, 0.009493, 0.009458, 0.009464]
rmse_intcovAvg_two_step_std = [0.009651, 0.009381, 0.00921, 0.0094, 0.009512, 0.00928]
rmse_intcovAvg_multitask_std = [0.009599, 0.009103, 0.008575, 0.008847, 0.009147, 0.009222]
rmses_intcovAvg_std = np.array([rmse_intcovAvg_baseline_std, rmse_intcovAvg_two_step_std, rmse_intcovAvg_multitask_std])


fig, ax = plt.subplots(figsize=(6, 2), dpi=1000)
for i in range(len(rmses_intcovAvg)):
    # ax.errorbar(x, y[i], yerr=y_std[i], label=names[i],
    #              linestyle=linestyles[i], color=colors[i], marker=markers[i])
    ax.plot(gammas[0:-1], rmses_intcovAvg[i, :], label=names[i], linestyle=linestyles[i], color=colors[i],
            marker=markers[i])
    plt.fill_between(gammas[0:-1], rmses_intcovAvg[i, :] - rmses_intcovAvg_std[i, :],
                     rmses_intcovAvg[i, :] + rmses_intcovAvg_std[i, :], alpha=0.1, color=colors[i], edgecolor=None)
ax.set_ylim([0.2, 0.3])
leg = ax.legend(fontsize=12, frameon=False, facecolor='gray', framealpha=0.02, edgecolor='black',
                bbox_to_anchor=(1.05, 0.5), loc='center left', labelspacing=1)
# ax.grid(True, linewidth=1, alpha=0.6, linestyle='dotted', axis='y')
# ax.patch.set_facecolor("gray")
# ax.patch.set_alpha(0.025)
ax.set_xticks(gammas[0:-1])
# ax.set_yticks([0.2, 0.25, 0.3])
ax.set_xlabel('Sampling informativeness $\gamma$')
ax.set_ylabel('RMSE')
fig.tight_layout()
plt.savefig(os.path.join(FIGS_PATH, "ResultsIntCovAverage.pdf"))
plt.show()

##########################################
#### Experiment with max intensities #####
##########################################

scaling_factors = [1, 2, 3, 4]

rmse_maxInt1_baseline_2 = [0.2907, 0.2655, 0.3928, 0.5874]
rmse_maxInt1_two_step_2 = [0.2139, 0.2465, 0.5093, 0.7937]
rmse_maxInt1_multitask_2 = [0.2006, 0.2104, 0.2967, 0.4736]

rmses_maxInt1_2 = np.array([rmse_maxInt1_baseline_2, rmse_maxInt1_two_step_2, rmse_maxInt1_multitask_2])

rmse_maxInt1_baseline_std_2 = [0.01025, 0.01524, 0.01788, 0.01257]
rmse_maxInt1_two_step_std_2 = [0.01091, 0.01448, 0.02084, 0.02229]
rmse_maxInt1_multitask_std_2 = [0.01164, 0.01253, 0.01142, 0.01313]
rmses_maxInt1_std_2 = np.array([rmse_maxInt1_baseline_std_2, rmse_maxInt1_two_step_std_2, rmse_maxInt1_multitask_std_2])

fig, ax = plt.subplots(figsize=(6, 2), dpi=1000)
for i in range(len(rmses_maxInt1_2)):
    # ax.errorbar(x, y[i], yerr=y_std[i], label=names[i],
    #              linestyle=linestyles[i], color=colors[i], marker=markers[i])
    ax.plot(scaling_factors, rmses_maxInt1_2[i, :], label=names[i], linestyle=linestyles[i], color=colors[i],
            marker=markers[i])
    plt.fill_between(scaling_factors, rmses_maxInt1_2[i, :] - rmses_maxInt1_std_2[i, :],
                     rmses_maxInt1_2[i, :] + rmses_maxInt1_std_2[i, :], alpha=0.1, color=colors[i], edgecolor=None)
# ax.set_ylim([0.2, 0.3])
leg = ax.legend(fontsize=12, frameon=False, facecolor='gray', framealpha=0.02, edgecolor='black',
                bbox_to_anchor=(1.05, 0.5), loc='center left', labelspacing=1)
# ax.grid(True, linewidth=1, alpha=0.6, linestyle='dotted', axis='y')
# ax.patch.set_facecolor("gray")
# ax.patch.set_alpha(0.015)
ax.set_xticks(scaling_factors)
# ax.set_yticks([0.2, 0.25, 0.3])
ax.set_xlabel('Sampling scarcity $S_{\lambda}$')
ax.set_ylabel('RMSE')
fig.tight_layout()
plt.savefig(os.path.join(FIGS_PATH, "ResultsScarcity.pdf"))
plt.show()

############################
### Intensity prediction ###
############################

bs_avg_two_step = [0.0005302, 0.0067, 0.008288, 0.007536, 0.009081, 0.01055]
bs_avg_multitask = [0.0005225, 0.006714, 0.008686, 0.008421, 0.01022, 0.01173]
bs_avg = np.array([bs_avg_two_step, bs_avg_multitask])

bs_avg_two_step_std = [0.00001356, 0.000162, 0.0005929, 0.0006896, 0.0007743, 0.0008791]
bs_avg_multitask_std = [0.0000135, 0.0001631, 0.0006059, 0.0007202, 0.0008344, 0.0009374]
bs_avg_std = np.array([bs_avg_two_step_std, bs_avg_multitask_std])


fig, ax = plt.subplots(figsize=(6, 2), dpi=1000)
for i in range(len(bs_avg)):
    ax.plot(gammas[0:-1], bs_avg[i, :], label=names[i + 1], linestyle=linestyles[i + 1], color=colors[i + 1],
            marker=markers[i + 1])
    ax.fill_between(gammas[0:-1], bs_avg[i, :] - bs_avg_std[i, :],
                     bs_avg[i, :] + bs_avg_std[i, :], alpha=0.1, color=colors[i + 1], edgecolor=None)
# ax1.grid(True, linewidth=1, alpha=0.6, linestyle='dotted', axis='y')
ax.patch.set_facecolor("gray")
ax.patch.set_alpha(0.025)
ax.set_xticks(gammas[0:-1])
ax.set_yticks([0., 0.005, 0.010, 0.015])
ax.set_xlabel('Sampling informativeness $\gamma$')
ax.set_ylabel('Brier Score')
ax.legend(fontsize=12, frameon=False, facecolor='gray', framealpha=0.02, edgecolor='black',
                bbox_to_anchor=(1.05, 0.5), loc='center left', labelspacing=1)
fig.tight_layout()
plt.savefig(os.path.join(FIGS_PATH, "ResultsIntensitiesBrierScore.pdf"))
plt.show()


#########################
### Alpha sensitivity ###
#########################

alphas = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9]

rmse_1_mt = [0.2318, 0.2299, 0.2299, 0.2291, 0.2283, 0.2284]
rmses_1_mt = np.array([rmse_1_mt])

rmse_1_mt_std = [0.01111, 0.01127, 0.01064, 0.01046, 0.01104, 0.01166]
rmses_1_mt_std = np.array([rmse_1_mt_std])

fig, ax = plt.subplots(figsize=(4, 2), dpi=1000)
for i in range(len(rmses_1_mt)):
    ax.plot(alphas, rmses_1_mt[i, :], label=names[i + 2], linestyle=linestyles[i + 2], color=colors[i + 2],
            marker=markers[i + 1])
    ax.fill_between(alphas, rmses_1_mt[i, :] - rmses_1_mt_std[i, :],
                     rmses_1_mt[i, :] + rmses_1_mt_std[i, :], alpha=0.1, color=colors[i + 2], edgecolor=None)
# ax1.grid(True, linewidth=1, alpha=0.6, linestyle='dotted', axis='y')
ax.patch.set_facecolor("gray")
ax.patch.set_alpha(0.025)
ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax.set_yticks([0.2, 0.22, 0.24])
ax.set_ylim([0.2, 0.25])
ax.set_xlabel('Hyperparameter ' + r'$\alpha$')
ax.set_ylabel('RMSE')
# ax.legend(fontsize=12, frameon=False, facecolor='gray', framealpha=0.02, edgecolor='black',
#                 bbox_to_anchor=(1.05, 0.5), loc='center left', labelspacing=1)
fig.tight_layout()
plt.savefig(os.path.join(FIGS_PATH, "ResultsAlphaSensitivity.pdf"))
plt.show()
