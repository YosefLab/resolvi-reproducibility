import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import torch
from pomegranate.distributions import Poisson
from pomegranate.gmm import GeneralMixtureModel
from scipy.spatial.distance import cosine
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
sns.reset_orig()
sc.settings._vector_friendly = True
sc.settings.n_jobs = -1
# p9.theme_set(p9.theme_classic)
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["savefig.transparent"] = True
plt.rcParams["figure.figsize"] = (4, 4)

plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.titleweight"] = 500
plt.rcParams["axes.titlepad"] = 8.0
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.labelweight"] = 500
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.labelpad"] = 6.0
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

plt.rcParams["font.size"] = 11
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', "Computer Modern Sans Serif", "DejaVU Sans"]
plt.rcParams['font.weight'] = 500

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['xtick.minor.size'] = 1.375
plt.rcParams['xtick.major.size'] = 2.75
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['xtick.minor.pad'] = 2

plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['ytick.minor.size'] = 1.375
plt.rcParams['ytick.major.size'] = 2.75
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['ytick.minor.pad'] = 2

plt.rcParams["legend.fontsize"] = 12
plt.rcParams['legend.handlelength'] = 1.4
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.scatterpoints'] = 3

plt.rcParams['lines.linewidth'] = 1.7
DPI = 300

def prettify_axis(ax, all_=False):
    if not all_:
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)

def compute_umap_embedding(adata, representation_key, n_comps=20, n_neighbors=20, min_dist_umap=0.3, show=False, key=None, extra_save='', batch_key='patient'):
    if n_comps is not None:
        adata.X = adata.layers[representation_key].copy()
        sc.pp.normalize_total(adata)
        sc.pp.sqrt(adata)
        sc.tl.pca(adata)
        if 'harmony' in key:
            from harmony import harmonize
            adata.obsm['X_pca_harmony'] = harmonize(adata.obsm['X_pca'], adata.obs, batch_key=batch_key)
            neighbor_key='X_pca_harmony'
            extra_save +='_harmony'
        else:
            neighbor_key='X_pca'
    else:
        neighbor_key = representation_key
    sc.pp.neighbors(adata, use_rep=neighbor_key, n_neighbors=n_neighbors, method='rapids')
    sc.tl.umap(adata, min_dist=min_dist_umap, method='rapids')
    sc.tl.louvain(adata, resolution=0.3, flavor='rapids')
    sc.pl.umap(
        adata, color=["louvain", "cluster", "diffusion_proportion", "total_counts"],
        frameon=False, ncols=2, save=f'_{representation_key}{extra_save}.pdf', show=show)
    if key:
        adata.obsm[f'X_umap_{key}'] = adata.obsm['X_umap'].copy()
        if n_comps:
            adata.obsm[f'X_pca_{key}'] = adata.obsm['X_pca'].copy()
    

def cosine_distance(adata, marker_list, layer_key='counts', output_dir=''):
    adata.uns['cosine_distances_' + layer_key] = pd.DataFrame(0, index=marker_list, columns=marker_list)
    for ind_x, gene_x in enumerate(marker_list):
        for ind_y, gene_y in enumerate(marker_list):
            if ind_x<ind_y:
                adata.uns['cosine_distances_' + layer_key].loc[gene_x, gene_y] = 1 - cosine(
                    u=adata.obsm['counts'][gene_x].values, v=adata.obsm['counts'][gene_y].values)
            else:
                adata.uns['cosine_distances_' + layer_key].loc[gene_x, gene_y] = 1 - cosine(
                    u=adata.obsm[layer_key][gene_x].values, v=adata.obsm[layer_key][gene_y].values)
                 
    plt.figure(figsize=(40, 40), dpi=DPI)
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(adata.uns['cosine_distances_' + layer_key], annot=True, annot_kws={"size": 8},
               square=True, mask=np.identity(len(marker_list)), linecolor='black', fmt=".2f",
                cmap='viridis', linewidths=1.2, vmax=0.2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.savefig(f'{output_dir}/cosine_distances_{layer_key}.pdf')
    
    
def cosine_distance_celltype(adata, marker_dict, layer_key='counts', output_dir="", vmax=0.5, extra_save=''):
    adata.uns['cosine_distances_ct_' + layer_key] = pd.DataFrame(
        0, index=marker_dict.keys(), columns=marker_dict.keys())
    for ind_x, ct_x in enumerate(marker_dict.keys()):
        for ind_y, ct_y in enumerate(marker_dict.keys()):
            if ind_x<ind_y:
                adata.uns['cosine_distances_ct_' + layer_key].loc[ct_x, ct_y] = 1 - cosine(
                    u=np.sum(adata.obsm['counts'][marker_dict[ct_x]], axis=1),
                    v=np.sum(adata.obsm['counts'][marker_dict[ct_y]], axis=1),
                )
            else:
                adata.uns['cosine_distances_ct_' + layer_key].loc[ct_x, ct_y] = 1 - cosine(
                    u=np.sum(adata.obsm[layer_key][marker_dict[ct_x]], axis=1),
                    v=np.sum(adata.obsm[layer_key][marker_dict[ct_y]], axis=1),
                )
                
    plt.figure(figsize=(10, 10), dpi=DPI)
    sns.set(font_scale=1.4) # for label size
    ax = sns.heatmap(adata.uns['cosine_distances_ct_' + layer_key], annot=True, annot_kws={"size": 8},
                square=True, linecolor='black', fmt=".2f", mask=np.identity(len(marker_dict.keys())),
                cmap='viridis', linewidths=1.2, vmax=vmax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.savefig(f'{output_dir}/cosine_distances_ct_{extra_save}_{layer_key}.pdf')
    

def double_positive_pmm(adata, marker_list, marker_dict=None, layer_key='counts', output_dir="", file_save='', vmax=0.3):
    if True: # 'positive_pmm_counts' not in adata.obsm:
        adata.obsm['positive_pmm_counts'] = pd.DataFrame(
            0, index=adata.obs_names, columns=marker_list)
        for gene_x in marker_list:
            model = GeneralMixtureModel(
                [Poisson([0.001], inertia=0.99), Poisson([0.5], inertia=0.999)], verbose=False).cuda()
            model_fit = model.fit(
                torch.tensor(adata.obsm['counts'][gene_x].values[..., np.newaxis]).cuda())
            adata.obsm['positive_pmm_counts'][gene_x] = model_fit.predict(torch.tensor(adata.obsm['counts'][gene_x].values[..., np.newaxis]).cuda()).cpu()
    
    adata.obsm[f'positive_pmm_{layer_key}'] = pd.DataFrame(
            0, index=adata.obs_names, columns=marker_list)
    for gene_x in marker_list:
        model = GeneralMixtureModel(
            [Poisson([0.001], inertia=0.99), Poisson([0.5], inertia=0.999)], verbose=False).cuda()
        model_fit = model.fit(
                torch.tensor(adata.obsm[layer_key][gene_x].values[..., np.newaxis]).cuda())
        adata.obsm[f'positive_pmm_{layer_key}'][gene_x] = model_fit.predict(torch.tensor(adata.obsm[layer_key][gene_x].values[..., np.newaxis]).cuda()).cpu()
        
    adata.uns[f'double_positive_{layer_key}'] = pd.DataFrame(
        0, index=marker_list, columns=marker_list)
        
    for ind_x, gene_x in enumerate(marker_list):  
        for ind_y, gene_y in enumerate(marker_list):
            if ind_x<ind_y:
                positives = adata.obsm['positive_pmm_counts'][[gene_x, gene_y]].sum(1)
                adata.uns[f'double_positive_{layer_key}'].loc[gene_x, gene_y] = np.sum(positives==2) / np.sum(positives>0) if np.sum(positives)>0 else None
            else:
                positives = adata.obsm[f'positive_pmm_{layer_key}'][[gene_x, gene_y]].sum(1)
                adata.uns[f'double_positive_{layer_key}'].loc[gene_x, gene_y] = np.sum(positives==2) / np.sum(positives>0) if np.sum(positives)>0 else None
                
    fig = plt.figure(figsize=(40, 40), dpi=DPI)
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    sns.set(font_scale=1.) # for label size
    sns.heatmap(adata.uns[f'double_positive_{layer_key}'], annot=True, annot_kws={"size": 6},
                square=True, linecolor='black', fmt=".1%", mask=np.identity(len(marker_list)),
                cmap='viridis', linewidths=1.2, vmax=vmax, xticklabels=True, yticklabels=True, ax=ax)
    
    if marker_dict:
        current_pos = 0
        for group, genes in marker_dict.items():
            gene_count = len(genes)
            if gene_count > 0:
                position = (current_pos, current_pos + gene_count - 1)
                current_pos += gene_count
                ax.text((position[0] + position[1]) / 2, - 1.3, group, ha='center', va='bottom', color='navy', fontsize=12)
                ax.plot([position[0] + 0.5, position[1] + 0.5], [- 0.5, - 0.5], lw=4, c='navy')

        ax.set_ylim([len(marker_list), -2])
        plt.subplots_adjust(top=0.9, bottom=0.5, left=0.07, right=0.93, hspace=0.2, wspace=0.2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.savefig(f'{output_dir}/double_positives_{layer_key}{file_save}.pdf')
    
    
def gene_gene_scatter_plots(adata, marker_dict, layer_key='counts', output_dir=""):
    ct_marker_dict = {}
    for ct in marker_dict.keys():
        ct_marker_dict[ct] = marker_dict[ct][
            np.argmax(np.sum(adata.obsm['generated_expression'].loc[:, marker_dict[ct]], axis=0))]

    fig, axs = plt.subplots(len(marker_dict.keys()), len(marker_dict.keys()), figsize=(20,20))
    for ind_x, ct_x in enumerate(marker_dict):
        for ind_y, ct_y in enumerate(marker_dict):
            if ind_x > ind_y:
                adata.obs['double_positive'] = adata.obsm['positive_pmm_' + layer_key][
                    [ct_marker_dict[ct_x], ct_marker_dict[ct_y]]].sum(1)
                adata.obs['double_positive'] = adata.obs['double_positive'].astype(str)
                #palette = {'0': '#0000EE', '1': '#3D9140', '2': '#DC143C', '3': '#DC143C'}
                adata.uns['double_positive_colors'] = ['#1f77b4', '#ff7f0e', '#2ca02c']
                sc.pl.scatter(adata, x=ct_marker_dict[ct_x], y=ct_marker_dict[ct_y], use_raw=False, layers=layer_key, title='',
                              show=False, ax=axs[ind_x, ind_y], color='double_positive')
                prettify_axis(axs[ind_x, ind_y])
                axs[ind_x, ind_y].get_legend().remove()
            else:
                axs[ind_x, ind_y].set_visible(False)
                
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gene_gene_scatter_{layer_key}.pdf')
    

def double_positive_boxplot(adata, gene_pairs, save_key='', show=False):
    ranges = [0] + [i/10 for i in np.arange(5, 11)]
    index = pd.MultiIndex.from_tuples(gene_pairs)
    dp_ct_counts = pd.DataFrame(index=index, columns=ranges[1:])
    dp_ct_generated = pd.DataFrame(index=index, columns=ranges[1:])

    for index, i in enumerate(ranges[1:]):
        for gene_x, gene_y in gene_pairs:
            subset = adata[np.logical_and(adata.obs['true_proportion']>ranges[index], adata.obs['true_proportion']<ranges[index+1])] 
            positives_counts = subset.obsm['positive_pmm_counts'][[gene_x, gene_y]].sum(1)
            positives_generated = subset.obsm[f'positive_pmm_generated_expression'][[gene_x, gene_y]].sum(1)
            dp_ct_counts.loc[(gene_x, gene_y), i] = (np.sum(positives_counts==2) / np.sum(positives_counts>0) if np.sum(positives_counts)>0 else -0.01)
            dp_ct_generated.loc[(gene_x, gene_y), i] = (np.sum(positives_generated==2) / np.sum(positives_generated>0) if np.sum(positives_generated)>0 else -0.01)

    dp_ct_counts_df = pd.DataFrame(dp_ct_counts).melt()
    dp_ct_generated_df = pd.DataFrame(dp_ct_generated).melt()

    dp_ct_counts_df['source'] = 'Measured'
    dp_ct_generated_df['source'] = 'Generated'

    # Concatenate the dataframes
    df = pd.concat([dp_ct_counts_df, dp_ct_generated_df])

    # Create a color palette
    palette = {'Measured': (1, 0, 0, 0.2), 'Generated': (0, 0, 1, 0.2)}  # red and blue with alpha=0.2
    palette2 = {'Measured': (0.5, 0.5, 0.5, 0.2), 'Generated': (0.5, 0.5, 0.5, 0.2)}  # red and blue with alpha=0.2

    # Create the dotplot
    plt.figure(figsize=(12, 8))
    sns.set(style='white')
    violin_parts = sns.violinplot(df, y='value', x='variable', hue='source', palette=palette, split=True, inner=None)
    for pc in violin_parts.collections:
        pc.set_alpha(0.8)

    # Create the boxplot with a third of the width and black color
    sns.boxplot(df, y='value', x='variable', hue='source', width=0.6, palette=palette2, fliersize=1.5, gap=0.5)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'overlapping_{save_key}.pdf')

    if show:
        plt.show()