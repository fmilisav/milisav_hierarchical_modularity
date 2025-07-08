from score_plotting_utils import plot_score_distributions, plot_multitasking_score_distributions, plot_empirical_score_distributions
from dynamics_plotting_utils import plot_timescale, plot_LE_curve

def save_main_results(path):
    plot_score_distributions(path, save_results=True)
    plot_score_distributions(path, levels=['3', 'MS'])
    plot_timescale(path)
    plot_LE_curve(path)

def save_multitasking_results(paths):
    for path in paths:
        plot_multitasking_score_distributions(path)

if __name__ == '__main__':

    multitasking_paths = ['results/SBM/SBM_MT_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse',
                          'results/SBM/SBM_MT_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse_1',
                          'results/SBM/SBM_MT_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse_2',
                          'results/SBM/SBM_MT_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse_3',
                          'results/SBM/SBM_MT_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse_4']
    sensitivity_paths = ['results/SBM/SBM_MC_tanh_nnodes25_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse',
                         'results/SBM/SBM_MC_tanh_nnodes50_p10.5_delta_p0.5_min_weight-1.0_bin_directedTrue_wei_directedFalse',
                         'results/SBM/SBM_MC_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedFalse_wei_directedFalse',
                         'results/SBM/SBM_MC_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse_1',
                         'results/SBM/SBM_MC_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse_2',
                         'results/SBM/SBM_MC_tanh_nnodes50_p10.5_delta_p0.25_min_weight0_bin_directedTrue_wei_directedFalse',
                         'results/SBM/SBM_MC_tanh_nnodes50_p10.5_delta_p0.75_min_weight0_bin_directedTrue_wei_directedFalse',
                         'results/SBM/SBM_MC_tanh_nnodes75_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse']

    save_main_results('results/SBM/SBM_MC_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse')
    save_multitasking_results(multitasking_paths)
    plot_multitasking_score_distributions('results/SBM/SBM_MT_tanh_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse', levels=['3', 'MS'])

    for path in sensitivity_paths:
        plot_score_distributions(path)
    plot_score_distributions('results/SBM/SBM_MC_tanh_nnodes50_p10.5_delta_p0.5_min_weight-1.0_bin_directedFalse_wei_directedFalse', alphas=[0.6, 0.8, 1.0])
    
    plot_empirical_score_distributions('results/empirical/empirical_MC_gamma1')

    plot_LE_curve('results/SBM/SBM_MC_linear_nnodes50_p10.5_delta_p0.5_min_weight0_bin_directedTrue_wei_directedFalse')

        