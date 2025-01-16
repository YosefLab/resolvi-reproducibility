export PATH_EXPERIMENT=../resolvi_eval_final2/mouse_colitis/original_q25_semisupervised_redo3
python scripts/run_model.py --adata_file ../resolvi_eval_final2/mouse_colitis/cleaned_up_labels_colitis_d0-d21.h5ad --output_dir $PATH_EXPERIMENT --model_save_path $PATH_EXPERIMENT --celltype_dict ibd_moffitt/marker_celltype.json --batch_key Slice_ID --celltype_key redo_celltyping --n_neighbors 20 --max_epochs 100 --retrain True --sample_quantile post_sample_q25 --semisupervised True

export PATH_EXPERIMENT=../resolvi_eval_final2/mouse_colitis/original_q25_unsupervised_d0-d21
python scripts/run_model.py --adata_file /external_data/other/resolvi_final_other_files/ibd_moffitt/adata_day1_21_processed_new.h5ad --output_dir $PATH_EXPERIMENT --model_save_path $PATH_EXPERIMENT --celltype_dict ibd_moffitt/marker_celltype.json --batch_key Slice_ID --celltype_key Tier3 --n_neighbors 20 --max_epochs 100 --retrain True --sample_quantile post_sample_q25 --semisupervised False
