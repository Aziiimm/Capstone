tiwa25@ubuntu1804:~/Capstone$ conda activate rapids-25.10
(rapids-25.10) tiwa25@ubuntu1804:~/Capstone$ python3 -m data.run_pipeline \
>   --source ../data/Electronics.jsonl \
>   --meta ../data/meta_Electronics.jsonl \
>   --output ../data/electronics_filtered_gpu.parquet \
>   --gpu \
>   --timings-file electronics_gpu_benchmarks.json
Loading (Backend: gpu)...
  Load: 64.06s
Filtering (user >= 6 reviews, item >= 11 reviews)...
  Filter: 0.45s
Writing Parquet...
/data/anaconda3/envs/rapids-25.10/lib/python3.12/site-packages/cudf/core/groupby/groupby.py:571: UserWarning: GroupBy.groups() performance scales poorly with number of groups. Got 353 groups.
  warnings.warn(
/data/anaconda3/envs/rapids-25.10/lib/python3.12/site-packages/cudf/core/groupby/groupby.py:571: UserWarning: GroupBy.groups() performance scales poorly with number of groups. Got 353 groups.
  warnings.warn(
/data/anaconda3/envs/rapids-25.10/lib/python3.12/site-packages/cudf/core/groupby/groupby.py:571: UserWarning: GroupBy.groups() performance scales poorly with number of groups. Got 353 groups.
  warnings.warn(
/data/anaconda3/envs/rapids-25.10/lib/python3.12/site-packages/cudf/core/groupby/groupby.py:571: UserWarning: GroupBy.groups() performance scales poorly with number of groups. Got 353 groups.
  warnings.warn(
/data/anaconda3/envs/rapids-25.10/lib/python3.12
