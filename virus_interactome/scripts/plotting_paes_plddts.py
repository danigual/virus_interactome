from virus_interactome import batch_plotting 
from pathlib import Path

base_path = Path("/home/daniel/ppi_data_remote/adeno/2_AF/output")
#base_path = Path("/media/DATA/ppi_data/adeno/2_AF/output")
for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            batch_plotting(subfolder)
