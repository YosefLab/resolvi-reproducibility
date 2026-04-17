# ResolVI - addressing noise and bias in spatial transcriptomics

This repository contains notebooks and scripts to reproduce analyses benchmarking the use of resolVI for spatial transcriptomics. Raw data is available at the vendor webpage for all commercial assays and on dryad for the MERFISH data. Processed data will be made available on dryad and can be shared upon reasonable request from AWS.

We recommend reviewing the [scvi-tools](https://scvi-tools.org/) documentation to get started.

## Repository structure
We provide here all notebooks to reproduce the results (notebooks with all outputs are managed by lfs).

- analysis notebooks and scripts:
  - `figure1.ipynb` - notebook to recreate molecule displays in figure 1
  - `figure2_final_with_outputs.ipynb` - notebook to recreate benchmarking in Figure 2 on Xenium brain data.
  - `figure3_vizgen_with_outputs.ipynb` - notebook to recreate figure 3 with benchmarking different segmentation on the liver cancer Vizgen MERSCOPE data.
  - `figure4_nanostring_with_outputs.ipynb` - notebook to recreate figure 4 highlighting spatial niches in human liver cancer using Nanostring CosMx data.
  - `figure_extended_stereoseq.ipynb` - notebook to recreate analysis of Stereo-SEQ brain data.
  - `figure_extended_visiumhd.ipynb` - notebook to recreate analysis of VisiumHD brain data.
  - `figure5_ibd.ipynb` - notebook to recreate figure 5 analyzing DSS colitis using MERFISH.
  - `figure5_ibd_abundance_with_outputs.ipynb` - notebook to recreate most of the differential abundance analysis in DSS colitis (rest in `figure5_ibd.ipynb`).

- revision scripts
  - `revision/figure2_ablation.ipynb` - notebook to recreate model benchmarks with ablation of components
  - `revision/figure2_de_area_count_scaling.ipynb` - notebook to recreate benchmark of count vs. area scaling DE results.
  - `revision/figure2_ovrlpy_processing.ipynb` - notebook to process Xenium brain data generating Ovrlpy results.
  - `revision/figure2_split_processing.ipynb` - notebook to process Xenium brain data generating SPLIT results.
  - `revision/figure3_benchmark_area_scaling.ipynb` - notebook to recreate benchmark of count vs. area scaling on Vizgen data
  - `revision/figure3_ovrlpy_processing.ipynb` - notebook to process Vizgen data generating Ovrlpy results.
  - `revision/figure3_split_processing.ipynb` - notebook to process Vizgen data generating SPLIT results.

- segmentation scripts
We store all scripts necessary to segment and annotate the raw data in the folder `notebooks_segmentation`, which contains files to segment and annotate Nanostring CosMx, Vizgen MERSCOPE and 10X Xenium data. For Nanostring CosMx, we show here diverse segmentation strategies (including ProSeg), while in the manuscript we only discuss the results after Baysor segmentation.

- training scripts
We store all scripts necessary to retrain all models in `training_scripts`, which contains files with commands to initialize training, marker genes used for initial double-positive calculcation and `training_scripts/run_model.py` which takes shell inputs, trains the model and generates a diverse set of evaluation metrics.

For any questions, please post an [issue](https://github.com/YosefLab/resolvi-reproducibility/issues) or reach out on [discourse](https://discourse.scverse.org).