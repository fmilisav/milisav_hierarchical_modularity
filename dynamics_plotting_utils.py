import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from score_plotting_utils import mannwhitneyu_print, create_fig_path

import pickle
from joblib import Parallel, delayed

from statsmodels.tsa.stattools import acf

import json

sns.set_style("ticks")
sns.set(context=None, style=None, palette=None, font_scale=5, color_codes=None)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 15})
plt.rcParams['legend.fontsize'] = 15

from matplotlib import cm
PuBu = cm.get_cmap('PuBu')
GnBu = cm.get_cmap('GnBu')
PuRd = cm.get_cmap('PuRd')
palette = [PuBu(0.6), GnBu(0.6), PuRd(0.6)]

#function to linearly fit the exponential decay of the ACF
#adapted from https://stackoverflow.com/a/3938548
def fit_exp_linear(t, y, C=0):
    #cut off starting from the negative values
    negatives = np.where(y < 0)[0]
    if len(negatives) > 0:
        first_negative = negatives[0]
        t = t[:first_negative]
        y = y[:first_negative]
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K

#function to calculate a timeseries' timescale
def timescale_analysis(timeseries):
    acf_arr = acf(timeseries)
    t = np.arange(len(acf_arr))
    A, K = fit_exp_linear(t, acf_arr)
    timescale = -1/K
    return timescale
    
def plot_timescale(experiment_path, aggregate='std', input=False, nmodules=8, 
                   save_timescales=True):
    """
    Plots the timescale distributions as KDE plots across seeds
    for different network levels at criticality
    Generates statistics files for pairwise comparisons
    
    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder
    aggregate : str
        Aggregation method ('average', 'std', or 'cumulative')
    input : bool
        Whether to include input nodes in the analysis
    nmodules : int
        Number of modules in the network
    save_timescales : bool
        Whether to save the timescales DataFrame as a pickle file
    """
    rs_path = os.path.join(experiment_path, 'reservoir_states')
    fig_path = create_fig_path(experiment_path)

    #check if the dataframe already exists
    df_path = os.path.join(experiment_path,
                           f'timescales_df_input{input}.pickle')
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)    
    else:          
        data = []
        seeds = []
        labels = []
        for rs_file in os.listdir(rs_path):
            if 'critical' in rs_file:
                
                if not input:
                    filename = rs_file.split('/')[-1]
                    #Get the input module
                    module = filename.split('_')[-2]
                    module = int(module[-1])
                    #Get the number of nodes in the module
                    #from the filename
                    nnodes = int(experiment_path.split('_')[3][6:])
                    nodes = (list(range(module*nnodes)) + 
                            list(range((module+1)*nnodes, nmodules*nnodes)))

                task = experiment_path.split('_')[1]
                rs = np.load(os.path.join(rs_path, rs_file), allow_pickle=True)
                if task == 'NG':
                    rs = np.concatenate(rs)
                rs = rs.T
                if not input:
                    rs = rs[nodes]

                timescales = Parallel(n_jobs=25, verbose=0)(delayed(timescale_analysis)(timeseries) for timeseries in rs)

                data.extend(timescales)
                seeds.extend([rs_file.split('_')[-1].replace('.npy', '')] * len(timescales))
                labels.extend([rs_file.split('_')[3][5:]] * len(timescales))


        df = pd.DataFrame({'timescale': data, 'seed': seeds, 'level': labels})
        df = df[df['level'] != 'MS']
    
    if aggregate == 'average':
        df = df.groupby(['seed', 'level']).mean().reset_index()
    elif aggregate == 'std':
        df = df.groupby(['seed', 'level']).std().reset_index()
    
    if save_timescales:
            df.to_pickle(os.path.join(experiment_path, 
                         f'timescales_df_input{input}.pickle'))

    with open(os.path.join(fig_path, f'timescale_criticality_{aggregate}_input{input}_stats.txt'), 'w') as f:
        timescales_1 = df[df['level'] == '1']['timescale']
        timescales_2 = df[df['level'] == '2']['timescale']
        timescales_3 = df[df['level'] == '3']['timescale']
        x_stats, y_stats, MWU_stats, cles_stats = mannwhitneyu_print(timescales_1, timescales_2, '1', '2')
        f.write(f'{x_stats}\n{y_stats}\n{MWU_stats}\n{cles_stats}\n\n')
        x_stats, y_stats, MWU_stats, cles_stats = mannwhitneyu_print(timescales_1, timescales_3, '1', '3')
        f.write(f'{x_stats}\n{y_stats}\n{MWU_stats}\n{cles_stats}\n\n')
        x_stats, y_stats, MWU_stats, cles_stats = mannwhitneyu_print(timescales_2, timescales_3, '2', '3')
        f.write(f'{x_stats}\n{y_stats}\n{MWU_stats}\n{cles_stats}\n\n')

    ax = sns.kdeplot(data=df, x='timescale', 
                     hue='level', palette=palette,
                     fill=True, cut=0, alpha=0.8)

    ax.set_xlabel('timescale')
    ax.set_box_aspect(1)

    fig = ax.get_figure()
    fig.savefig(os.path.join(fig_path, 
                f'criticality_{aggregate}_input{input}.png'), dpi=300)
    fig.savefig(os.path.join(fig_path, 
                f'criticality_{aggregate}_input{input}.svg'), dpi=300)
    plt.close(fig)

def plot_LE_curve(experiment_path):
    """
    Plots the maximum Lyapunov exponent distributions as boxplots across seeds
    for different network levels and alpha values
    Generates statistics files for pairwise comparisons

    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder
    """
    LE_path = os.path.join(experiment_path, 'LE')

    data = []
    labels = []
    alphas = []
    #Get nnetworks in config.txt
    with open(os.path.join(experiment_path, 'config.txt'), 'r') as f:
        config = json.load(f)
        nnetworks = config['nnetworks']

    levels = ['1', '2', '3']
    for seed in range(nnetworks):
        for level in levels:
            for filename in os.listdir(LE_path):
                if ('LEs_alpha' in filename and f'level{level}' in filename and
                    f'{seed}.npy' in filename):
                        LEs = np.load(os.path.join(LE_path, filename), 
                                      allow_pickle=True)

                        data.append(LEs[0])
                        labels.append(level)

                        #Get the alpha value from the filename
                        alpha = filename.split('alpha')[1].split('_')[0]
                        alphas.append(float(alpha))

    fig_path = create_fig_path(experiment_path)

    df = pd.DataFrame({'maximum Lyapunov exponent': data, 'level': labels, 'alpha': alphas})
    ax = sns.boxplot(x='alpha', y='maximum Lyapunov exponent', hue='level',
                     data=df, palette=palette,
                     linewidth=0.75, showfliers=False)

    for _, s in ax.spines.items():
        s.set_linewidth(0.5)
    ax.set_box_aspect(1)

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('maximum Lyapunov exponent')

    fig = ax.get_figure()
    fig.savefig(os.path.join(fig_path, f'LE_curve.png'), dpi=300)
    fig.savefig(os.path.join(fig_path, f'LE_curve.svg'), dpi=300)
    plt.close(fig)