from virus_interactome import run_ipsae_for_all_parallel, merge_ipsae_results

# 1. Run ipSAE for all AF3 models (parallel)
# run_ipsae_for_all_parallel("/home/daniel/ppi_data_remote/adeno/2_AF/output", ipsae_script="~/ppi_data_remote/scripts/ipsae.py", n_cores=4)
run_ipsae_for_all_parallel("/media/DATA/ppi_data/adeno/2_AF/output", ipsae_script="/media/DATA/ppi_data/scripts/ipsae.py", n_cores=4)
# 2. Merge all results into pandas DataFrames
#df_global, df_byres = merge_ipsae_results("/home/daniel/ppi_data_remote/adeno/2_AF/output")
df_global, df_byres = merge_ipsae_results("/media/DATA/ppi_data/adeno/2_AF/output")