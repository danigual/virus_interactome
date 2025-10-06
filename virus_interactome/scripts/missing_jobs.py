from glob import glob
import json

## Find executed jobs
executed_jobs = glob("/home/daniel/ppi_data_remote/adeno/2_AF/output/*/")
executed_jobs = [i.split("/")[-2].lower() for i in executed_jobs]

## Find all jobs that I wanted to complete
files_that_i_sended = [f"/home/daniel/ppi_data_remote/adeno/2_AF/input/{i}.json" for i in range(0, 21)]

all_ids_that_i_sended = []
all_original_names = []
for my_file in files_that_i_sended:
    with open(my_file) as f:
        tmp_data = json.load(f)
        parsed_names = [i["name"].lower().replace("-","_").strip() for i in tmp_data]
        all_ids_that_i_sended.extend(parsed_names)
        original_names = [i["name"] for i in tmp_data]
        all_original_names.extend(original_names)

print(len(all_ids_that_i_sended), len(executed_jobs))

executed_jobs_cp = executed_jobs.copy()
## Get a list of missing jobs, that did not execute
with open("missing_jobs_original.txt", "w") as f:
    for idx, i in enumerate(all_ids_that_i_sended):
        if i not in executed_jobs:
            f.write(all_original_names[idx] + "\n")
           
