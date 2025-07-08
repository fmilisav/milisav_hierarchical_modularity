import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pingouin as pg
from scipy.stats import mannwhitneyu, iqr

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

def create_fig_path(experiment_path):
    fig_path = os.path.join(experiment_path, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    return fig_path

#function to perform Mann-Whitney U test
def mannwhitneyu_print(x, y, xlabel, ylabel):
    u, p_val = mannwhitneyu(x, y)
    x_stats = '{} median={}, IQR={}'.format(xlabel, np.median(x), iqr(x))
    y_stats = '{} median={}, IQR={}'.format(ylabel, np.median(y), iqr(y))
    MWU_stats = 'Mann-Whitney U rank test {}: u={}, p={}'.format(xlabel + '-' +
                                                          ylabel, u, p_val)
    cles_stats = 'cles = {}'.format(pg.compute_effsize(x, y, eftype = 'CLES'))

    return x_stats, y_stats, MWU_stats, cles_stats

#function to perform pairwise comparisons within alpha
#and store stats in a results text file
def pairwise_comparisons(fig_path, results_df, labels, 
                         task='MC', aggregate='average'):
    filename = f'{task}_{aggregate}_stats_MS.txt' if 'MS' in results_df['level'].unique() else f'{task}_{aggregate}_stats.txt'
    with open(os.path.join(fig_path, filename), 
              'w') as f:
        for alpha in results_df['alpha'].unique():
            f.write(f'Alpha = {alpha}\n')
            scores1 = results_df[(results_df['alpha'] == alpha) & 
                                 (results_df['level'] == labels[0])]['score']
            scores2 = results_df[(results_df['alpha'] == alpha) & 
                                 (results_df['level'] == labels[1])]['score']
            if len(labels) >= 3:
                scores3 = results_df[(results_df['alpha'] == alpha) & 
                                    (results_df['level'] == labels[2])]['score']

                for x, y, xlabel, ylabel in zip([scores1, scores1, scores2], 
                                                [scores2, scores3, scores3],
                                                [labels[0], labels[0], labels[1]],
                                                [labels[1], labels[2], labels[2]]):
                    x_stats, y_stats, MWU_stats, cles_stats = mannwhitneyu_print(x, y, xlabel, ylabel)
                    f.write(f'{x_stats}\n{y_stats}\n{MWU_stats}\n{cles_stats}\n\n')
            else:
                x_stats, y_stats, MWU_stats, cles_stats = mannwhitneyu_print(scores1, scores2, labels[0], labels[1])
                f.write(f'{x_stats}\n{y_stats}\n{MWU_stats}\n{cles_stats}\n\n')

#function to generate performance boxplots across alphas
def generate_boxplots(results_df, fig_path, task='MC', aggregate='average'):
    
    boxplot_palette = []
    for level in results_df['level'].unique():
        if level == '1' or level == 'modular':
            boxplot_palette.append(palette[0])
        elif level == '2' or level == 'hierarchical modular':
            boxplot_palette.append(palette[1])
        elif level == '3' or level == 'empirical':
            boxplot_palette.append(palette[2])
        else:
            boxplot_palette.append('lightgrey')

    ax = sns.boxplot(x='alpha', y='score', hue='level',
                     data=results_df, palette=boxplot_palette,
                     linewidth=0.75, showfliers=False)

    for _,s in ax.spines.items():
        s.set_linewidth(0.5)
    ax.set_box_aspect(1)

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Score')

    fig = ax.get_figure()
    filename = f'{task}_{aggregate}'
    if 'MS' in results_df['level'].unique():
        filename += '_MS'
    fig.savefig(os.path.join(fig_path, filename + '.png'), dpi=300)
    fig.savefig(os.path.join(fig_path, filename + '.svg'), dpi=300)
    plt.close(fig)

#function to aggregate scores across modules
def aggregate_scores(df, aggregate='average'):
    if aggregate == 'average':
        df_agg = df.groupby('alpha').agg({'score': 'mean'}).reset_index()
    elif aggregate == 'max':
        df_agg = df.groupby('alpha').agg({'score': 'max'}).reset_index()
    else:
        raise ValueError("Invalid aggregation method. " \
                         "Choose from 'average' or 'max'.")
    
    return df_agg

def plot_score_distributions(experiment_path, aggregate='average', 
                             save_results=False, levels=['1', '2', '3'],
                             alphas=None):
    """
    Plots the score distributions as boxplots across seeds 
    for different network levels and alpha values
    Produces statistics file for pairwise comparisons

    Parameters
    ----------
    experiment_path : str 
        Path to the experiment folder
    aggregate : str 
        Aggregation method for the scores across modules ('average' or 'max')
    save_results : bool
        Whether to save the results DataFrame as a pickle file
    levels : list
        List of network levels to consider for the plot
    alphas : list
        List of alpha values to consider for the plot
    """
    performance_path = os.path.join(experiment_path, 'performance')

    data = []
    for seed_file in os.listdir(performance_path):
        seed_results = np.load(os.path.join(performance_path, seed_file), 
                               allow_pickle=True)
        #Get seed number from seed_file
        seed = seed_file.split('_')[-1][4:].replace('.npy', '')

        for level in levels:
            df = seed_results[level]
            df_agg = aggregate_scores(df, aggregate=aggregate)
            df_agg['seed'] = seed
            df_agg['level'] = level
            data.append(df_agg)

    results_df = pd.concat(data, ignore_index=True)
    if alphas is not None:
        results_df = results_df[results_df['alpha'].isin(alphas)]
    if save_results:
        results_df.to_pickle(os.path.join(experiment_path,
                                          f'{aggregate}_results_df.pickle'))

    fig_path = create_fig_path(experiment_path)
    pairwise_comparisons(fig_path, results_df, levels, 
                         aggregate=aggregate)
    generate_boxplots(results_df, fig_path, aggregate=aggregate)

def plot_empirical_score_distributions(experiment_path, aggregate='average'):
    """
    Plots the score distributions as boxplots across seeds 
    for different network types and alpha values
    Produces statistics file for pairwise comparisons

    Parameters
    ----------
    experiment_path : str 
        Path to the experiment folder
    aggregate : str 
        Aggregation method for the scores across modules ('average' or 'max')
    """
    performance_path = os.path.join(experiment_path, 'performance')

    data = []
    for seed_file in os.listdir(performance_path):
        seed_results = np.load(os.path.join(performance_path, seed_file), 
                               allow_pickle=True)
        #Get seed number from seed_file
        seed = seed_file.split('_')[-1][4:].replace('.npy', '')

        for net_type in ['modular', 'hierarchical modular', 'empirical']:
            MCs = seed_results[net_type]
            #Loop over results from different input modules
            input_module_MCs = []
            for input_module in range(len(MCs)):
                input_module_MC = MCs[input_module]
                #Aggregate scores across output modules
                df_agg = aggregate_scores(input_module_MC, aggregate=aggregate)
                df_agg['input_module'] = input_module
                input_module_MCs.append(df_agg)
            agg_input_module_MCs = pd.concat(input_module_MCs, 
                                             ignore_index=True)

            #Aggregate scores across input modules
            if aggregate == 'average':
                agg_output_module_MCs = aggregate_scores(agg_input_module_MCs,
                                                         aggregate=aggregate)
            else:
                idx = agg_input_module_MCs.groupby('alpha')['score'].idxmax()
                agg_output_module_MCs = agg_input_module_MCs.loc[idx].reset_index()
            agg_output_module_MCs['seed'] = seed
            agg_output_module_MCs['level'] = net_type
            data.append(agg_output_module_MCs)

    results_df = pd.concat(data, ignore_index=True)

    fig_path = create_fig_path(experiment_path)
    pairwise_comparisons(fig_path, results_df, 
                         ['modular', 'hierarchical modular', 'empirical'], 
                         aggregate=aggregate)
    generate_boxplots(results_df, fig_path, aggregate=aggregate)

def plot_multitasking_score_distributions(experiment_path,
                                          levels=['1', '2', '3']):
    """
    Plots the score distributions as boxplots across seeds 
    for different network levels and alpha values
    Produces statistics file for pairwise comparisons

    Parameters
    ----------
    experiment_path : str 
        Path to the experiment folder
    nmodules : int
        Number of modules in the network
    levels : list
        List of network levels to consider for the plot
    """
    performance_path = os.path.join(experiment_path, 'performance')

    data = []
    for seed_file in os.listdir(performance_path):
        #scores averaged over tasks/output modules
        seed_results = np.load(os.path.join(performance_path, seed_file), 
                                allow_pickle=True)[-1]

        for level in levels:
            df_agg = seed_results[level].to_frame()
            #Make alpha index a column
            df_agg.reset_index(inplace=True)
            df_agg['level'] = level
            data.append(df_agg)

    results_df = pd.concat(data, ignore_index=True)

    fig_path = create_fig_path(experiment_path)
    pairwise_comparisons(fig_path, results_df, levels, task='MT')
    generate_boxplots(results_df, fig_path, task='MT')