from virus_interactome import proteome_json, load_json

proteome_dictionary = load_json("/home/daniel/ppi_data_remote/adeno/0_proteomes/curated/HAdV5_AC_000008_1_modified.fa")
# AdV5_proteome_json(proteome_dictionary, "/tmp/AdV5_2")
proteome_json(proteome_dictionary, "/home/daniel/ppi_data_remote/adeno/2_AF/input/")
