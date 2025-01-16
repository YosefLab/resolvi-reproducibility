import sys
sys.path.insert(0,'/home/cane/Documents/yoseflab/can/resolVI')
from scvi.external import RESOLVI

import scanpy as sc
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import click
import numpy as np
import pandas as pd
import os
import json
import logging
import scipy.sparse as scp
import _utils
torch.manual_seed(0)

sc.settings.n_jobs=20

@click.command()
@click.option('--adata_file', type=click.STRING, help='input dataset')
@click.option("--output_dir", type=click.STRING, help="folder to store results of evaluation")
@click.option("--model_save_path", type=click.STRING, help="path to store trained resolVI model")
@click.option('--celltype_dict', type=click.STRING, help='Path to dictionary with cell-type marker genes')
@click.option('--spatial_rep', type=click.STRING, default='spatial', help='Obsm field with spatial coordinated')
@click.option('--n_neighbors', type=click.INT, default=20, help='Number of nearest neighbors used for neighborhood')
@click.option('--batch_key', type=click.STRING, default=None, help='Obs column with batch information')
@click.option('--covariate_key', type=click.STRING, default=None, help='Obs column with fov information')
@click.option('--celltype_key', type=click.STRING, default='celltypes_coarse_hand', help='Obs column with cell-type information')
@click.option('--n_latent', type=click.INT, default=10, help='Number of latent dimensions in model')
@click.option('--max_epochs', type=click.INT, default=None, help='Number of epochs model is trained')
@click.option('--sample_quantile', type=click.STRING, default='post_sample_q50', help='Function for the computation of posterior sampling')
@click.option('--semisupervised', type=click.BOOL, default=False, help='Function for the computation of posterior sampling')
@click.option('--retrain', type=click.BOOL, default=True, help='Retrain model from scratch or load from file.')
def main(
    adata_file, output_dir, model_save_path, celltype_dict, spatial_rep, n_neighbors,
    batch_key, covariate_key, celltype_key, n_latent, max_epochs, sample_quantile, retrain, semisupervised):
    print(f'{adata_file} {output_dir} {model_save_path} {sample_quantile}')
    # Directory management
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.isdir(output_dir):
        logging.info(F"Directory {output_dir} doesn't exist, creating it")
        os.makedirs(output_dir)
    else:
        logging.info(F"Found directory at: {output_dir}")
    sc.settings.figdir = output_dir

    adata = sc.read(adata_file)
    if "sub" in output_dir:
        print("Subsetting", adata)
        adata = adata[adata.obs['fov_mean']<10].copy()
        print("Subsetted", adata)
    adata.obs['cluster'] = adata.obs[celltype_key]
    adata = adata[~adata.obs['cluster'].isna()] # Filter out nan celltypes.
    adata.layers['counts'] = scp.csr_matrix(adata.layers['counts'])
    adata.layers['raw_counts'] = adata.layers['counts'].copy()
    adata.obs_names_make_unique()
    if covariate_key is not None:
        covariate_key = covariate_key.split(',')
    if retrain is True:
        RESOLVI.setup_anndata(
            adata, layer='raw_counts', batch_key=batch_key, labels_key="cluster",
            categorical_covariate_keys=covariate_key,
            prepare_data_kwargs={
                'n_neighbors': n_neighbors, 'spatial_rep': spatial_rep}
            )
        resolvae = RESOLVI(adata, n_latent=10, semisupervised=semisupervised, mixture_k=100, deeply_inject_covariates=True, encode_covariates=False, conditional_norm='none')

        resolvae.train(max_epochs=100, lr=1e-3, lr_extra=5e-2, enable_progress_bar=False)
        resolvae.save(model_save_path + '/resolvae/', save_anndata=False, overwrite=True)
    else:
        resolvae = RESOLVI.load(model_save_path + '/resolvae/', adata=adata)
    print(f'Finished training {output_dir}')
    
    
    samples_corr = resolvae.sample_posterior_predictive(
        model=resolvae.module.model_corrected,
        return_sites=['px_rate', 'obs'],
        num_samples=30, return_samples=False, batch_size=1000, macro_batch_size=50000)
    samples_corr = pd.DataFrame(samples_corr).T

    samples = resolvae.sample_posterior_predictive(
        model=resolvae.module.model_residuals,
        return_sites=[
            'mixture_proportions', 'mean_poisson', 'per_gene_background', 
            'diffusion_mixture_proportion', 'per_neighbor_diffusion', 'px_r_inv'
            ],
        num_samples=30, return_samples=False, batch_size=1000, macro_batch_size=50000)
    samples = pd.DataFrame(samples).T
    
    adata.obs['true_proportion'] = samples.loc['post_sample_means', 'mixture_proportions'][:, 0]
    adata.obs['diffusion_proportion'] = samples.loc['post_sample_means', 'mixture_proportions'][:, 1]
    adata.obs['background_proportion'] = samples.loc['post_sample_means', 'mixture_proportions'][:, 2]
    adata.varm['background'] = samples.loc['post_sample_means', 'per_gene_background'].squeeze().T
    adata.var['px_r'] = 1/(1e-6 + samples.loc['post_sample_means', 'px_r_inv'][0, :])
    
    _ = plt.hist(adata.obs['true_proportion'], bins=30, range=(0,1))
    _ = plt.hist(adata.obs['diffusion_proportion'], bins=30, range=(0,1))
    _ = plt.hist(adata.obs['background_proportion'], bins=30, range=(0,1))
    plt.legend(['True_proportions', 'Diffusion_Proportion', 'Background_Proportion'])
    plt.savefig(f'{output_dir}/histogram_proportions.pdf')
    
    adata.obsm["X_resolVI"] = resolvae.get_latent_representation()
    adata.layers["generated_expression"] = scp.csr_matrix(samples_corr.loc[sample_quantile, 'obs'])
    adata.layers["corrected_counts"] = adata.layers['counts'].multiply((samples_corr.loc[sample_quantile, 'px_rate'] / (
        1.0 + samples_corr.loc[sample_quantile, 'px_rate'] + samples.loc['post_sample_means', 'mean_poisson']))).tocsr()
    
    _utils.compute_umap_embedding(adata, representation_key="X_resolVI", n_comps=None)
    _utils.compute_umap_embedding(adata, representation_key="counts")
    _utils.compute_umap_embedding(adata, representation_key="generated_expression")
    _utils.compute_umap_embedding(adata, representation_key="corrected_counts")
    
    adata.X = adata.layers['counts'].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sc.tl.rank_genes_groups(adata, groupby=celltype_key, use_raw=False)
    sc.pl.rank_genes_groups_dotplot(adata, layer='counts', save='markers_counts', n_genes=5, use_raw=False, show=False, standard_scale='var')
    sc.pl.rank_genes_groups_dotplot(adata, layer='generated_expression', save='markers_generated', n_genes=5, use_raw=False, show=False, standard_scale='var')
    sc.pl.rank_genes_groups_dotplot(adata, layer='corrected_counts', save='markers_corrected', n_genes=5, use_raw=False, show=False, standard_scale='var')
    
    with open(celltype_dict, "r") as j:
        marker_dict = json.load(j)
    marker_list_ = sum(marker_dict.values(), [])
    # REmove duplicates
    marker_list = []
    _ = [marker_list.append(x) for x in marker_list_ if x not in marker_list]
    
    adata.obsm['counts'] = pd.DataFrame(adata[:, marker_list].layers['counts'].A, columns=marker_list, index=adata.obs_names)
    adata.obsm['generated_expression'] = pd.DataFrame(adata[:, marker_list].layers['generated_expression'].A, columns=marker_list, index=adata.obs_names)
    adata.obsm['corrected_counts'] = pd.DataFrame(adata[:, marker_list].layers['corrected_counts'].A, columns=marker_list, index=adata.obs_names)
    
    adata.write(f'{output_dir}/complete_adata.h5ad')
    
    _utils.cosine_distance(adata, marker_list, layer_key="generated_expression", output_dir=output_dir)
    _utils.cosine_distance_celltype(adata, marker_dict, layer_key="generated_expression", output_dir=output_dir)
    _utils.double_positive_pmm(adata, marker_list, layer_key="generated_expression", output_dir=output_dir)
    _utils.gene_gene_scatter_plots(adata, marker_dict, layer_key='generated_expression', output_dir=output_dir)
        
    _utils.cosine_distance(adata, marker_list, layer_key="corrected_counts", output_dir=output_dir)
    _utils.cosine_distance_celltype(adata, marker_dict, layer_key="corrected_counts", output_dir=output_dir)
    _utils.double_positive_pmm(adata, marker_list, layer_key="corrected_counts", output_dir=output_dir)
    _utils.gene_gene_scatter_plots(adata, marker_dict, layer_key='corrected_counts', output_dir=output_dir)
    
    _utils.gene_gene_scatter_plots(adata, marker_dict, layer_key='counts', output_dir=output_dir)
    
    adata.write(f'{output_dir}/complete_adata.h5ad')
    print(f'Finished {output_dir}')
    
if __name__ == '__main__':
    main()