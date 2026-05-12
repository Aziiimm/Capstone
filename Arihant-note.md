# how to activate rapids env 

cd ~/Capstone
source /data/anaconda3/etc/profile.d/conda.sh
conda activate rapids-25.10


ssh -J tiwa2147@134.74.112.50 tiwa25@134.74.112.18

# installed verions 

cudf: 25.10.00
cuml: 25.10.00
cupy: 13.6.0
fastapi: 0.136.0
uvicorn: 0.45.0


python -c "import cupy; print('CUDA devices:', cupy.cuda.runtime.getDeviceCount())"

CUDA devices: 1

# HOW TO TRAIN 

# Run the training script using the recursive path
python models/new_model.py \
  --path "output/SD/**/*.parquet" \
  --neighbors 10 \
  --output models/full_recommender.pkl`


# VERIFICATION of asin_to_idx mappings 

(rapids-25.10) tiwa25@ubuntu1804:~/Capstone$ python -c "
> import pickle
> from recommender import AmazonRecommenderGPU
> with open('models/full_recommender.pkl', 'rb') as f:
>     r = pickle.load(f)
> print('--- Pickle Contents ---')
> print(f'Total Items in Map: {len(r.title_map):,}')
> print(f'Total ASINs in Map: {len(r.asin_to_idx):,}')
> "
--- Pickle Contents ---
Total Items in Map: 2,168,296
Total ASINs in Map: 2,168,296
(rapids-25.10) tiwa25@ubuntu1804:~/Capstone$ 


# RUNNING THE BACKEND SERVER USING nohup, does logging too

MODEL_PATH=models/full_recommender.pkl \
nohup uvicorn models.server:app \
--host 0.0.0.0 \
--port 8000 \
--workers 1 \
> logs/server.log 2>&1 &

# CURL testing FASTAPI backend results. 

(rapids-25.10) tiwa25@ubuntu1804:~/Capstone$ python -c "import urllib.request print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"

{"status":"ok","model_loaded":true,"catalogue_size":2168296}

(rapids-25.10) tiwa25@ubuntu1804:~/Capstone$ 

# CHECK THE 8000 port 

fuser 8000/tcp

# KILL process running on port 8000 

kill -9 514351 

replace with process ID 

then restart the API server

# SAMPLE CODE TO SEND API REQUEST

# QUERY RESPONSE FORMAT `

```
{
    "asin": "B07N69T6TM",
    "query_title": "Toys for 4-5 Year Old Boys, Mom&myaboys 8 X 21 Kids Binoculars for Children,Compact Telescope Boys Gifts 4-8 Years Old to Bird Watching &Scenery(Yellow)",
    "recommendations": [
        {
            "asin": "B06ZZRVVL8",
            "title": "Rust-Oleum 318223 Rocksolid Deck Resurfacer 20x Roller Cover, 9 inch"
        },
        {
            "asin": "B09V7RMKW7",
            "title": "JOWAVE Bluetoth Headset"
        },
        {
            "asin": "B07K2YH2BZ",
            "title": "Hotouch Ladies Night Shirt Nighties for Women Sleepwear Vintage Nightgown Grey M"
        },
        {
            "asin": "B07568JSR3",
            "title": "Skechers Men's Relaxed Fit-elent-mosen Boat Shoe"
        },
        {
            "asin": "B08FD2J5CK",
            "title": "NILOUFO Womens Casual Summer Shirts Notch V Neck Blouses 3/4 Roll Sleeve Tops Tunics"
        },
        {
            "asin": "B07PXV2BJM",
            "title": "Nicky Bigs Novelties Police S.W.A.T. Team Helmet with Folding Visor Costume Accessory, Black, One Size"
        },
        {
            "asin": "B07PFK9L28",
            "title": "Bowin Amplified HD TV Antenna 50-130 Mile- 2019 Newest 4K 1080P HD Indoor Digital TV Antenna with 13.2 Feet Coax Cable for All TVs"
        },
        {
            "asin": "B07BD1ZGDZ",
            "title": "Women Fashion Large Tote Shoulder Handbag Waterproof Tote Bag Multi-function Nylon Bag for Gym Travel Work Shopping"
        },
        {
            "asin": "B07198V59S",
            "title": "ZEEMOO Men's Crazy Horse Leather Business Bag Work Tote Laptop Briefcase Messenger Bag Shoulder Bag Fit 15\" Laptop (Brown)"
        },
        {
            "asin": "B08MBDRKQK",
            "title": "Reebok Women's Socks - 12 Pack Athletic Quarter Crew Socks"
        }
    ],
    "inference_ms": 1072.23
}
(rapids-25.10) tiwa25@ubuntu1804:~/Capstone$ 
```


# Restrat the server to bring response time down in a new session 

(rapids-25.10) tiwa25@ubuntu1804:~/Capstone$ fuser -k 8000/tcp
8000/tcp:            514466
(rapids-25.10) tiwa25@ubuntu1804:~/Capstone$ MODEL_PATH=models/full_recommender.pkl nohup uvicorn models.server:app --host 0.0.0.0 --port 8000 --workers 1 > logs/server.log 2>&1 &
[2] 515990
[1]   Killed                  MODEL_PATH=models/full_recommender.pkl nohup uvicorn models.server:app --host 0.0.0.0 --port 8000 --workers 1 > logs/server.log 2>&1
(rapids-25.10) tiwa25@ubuntu1804:~/Capstone$ python -c "
import urllib.request, json, time
asin_val = 'B07N69T6TM'
latencies = []
print('--- Starting Benchmark (20 Requests) ---')
for i in range(20):
    start = time.time()
    data = json.dumps({'asin': asin_val, 'n': 5}).encode()
    req = urllib.request.Request('http://localhost:8000/recommend', data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as f:
        res = json.loads(f.read().decode())
        latencies.append(res['inference_ms'])
    print(f'Req {i+1}: {res[\"inference_ms\"]:.2f}ms')

avg = sum(latencies) / len(latencies)
print(f'\n--- RESULTS ---')
print(f'Cold Start (1st Req): {latencies[0]:.2f}ms')
print(f'Warm Average (Reqs 2-20): {sum(latencies[1:])/(len(latencies)-1):.2f}ms')
print(f'Best Time: {min(latencies):.2f}ms')
"

# 20 mock requests benchmark results

--- Starting Benchmark (20 Requests) ---
Req 1: 353.62ms
Req 2: 91.76ms
Req 3: 91.83ms
Req 4: 91.75ms
Req 5: 118.10ms
Req 6: 90.39ms
Req 7: 91.44ms
Req 8: 90.45ms
Req 9: 90.98ms
Req 10: 90.68ms
Req 11: 110.96ms
Req 12: 90.92ms
Req 13: 91.38ms
Req 14: 91.10ms
Req 15: 91.74ms
Req 16: 91.88ms
Req 17: 91.23ms
Req 18: 92.02ms
Req 19: 91.74ms
Req 20: 91.52ms

--- RESULTS ---
Cold Start (1st Req): 353.62ms
Warm Average (Reqs 2-20): 93.78ms
Best Time: 90.39ms



MODEL_PATH=models/full_recommender.pkl \
nohup uvicorn models.server:app \
--host 0.0.0.0 \
--port 8000 \
--workers 1 \
> logs/server.log 2>&1 &

PYTHONPATH=$PYTHONPATH:$(pwd)/models MODEL_PATH=models/full_recommender.pkl nohup uvicorn models.server:app --host 0.0.0.0 --port 8000

PYTHONPATH=$PYTHONPATH:$(pwd)/models MODEL_PATH=models/full_recommender.pkl uvicorn models.server:app --host 127.0.0.1 --port 8000

MODEL_PATH=models/full_recommender.pkl uvicorn models.server:app --host 127.0.0.1 --port 8000

PYTHONPATH=$PYTHONPATH:$(pwd)/models MODEL_PATH=models/full_recommender.pkl uvicorn models.server:app --host 127.0.0.1 --port 8000

curl -s http://localhost:8000/health


watch live logs tail -f ~/Capstone/uvicorn.log 


# The SSH tunnel — must run during the demo, but can be backgrounded:

ssh -fN -L 8000:localhost:8000 \
  -J tiwa2147@134.74.112.50 tiwa25@134.74.112.18
-f — fork to background after auth
-N — no remote shell, just the tunnel
Add keepalives so it doesn't drop mid-presentation:


ssh -fN -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 8000:localhost:8000 \
  -J tiwa2147@134.74.112.50 tiwa25@134.74.112.18
To find/kill it later: pgrep -f "ssh.*8000:localhost:8000" then kill <pid>.



# Results from testing using claude evaluate.py 

Loading recommender from models/full_recommender.pkl ...
  loaded in 4.25s (2,168,296 items)
Building user → item map (CSR → CSC) ...
  done in 0.33s (n_users=3,854,104, n_items=2,168,296)
Eligible users (≥5 ratings): 2,720,193
Sampling 1,000 users for evaluation.

Evaluating with K ∈ [5, 10, 20], largest_k=20 ...
  100/1000 users  (14.0s elapsed)
  200/1000 users  (29.0s elapsed)
  300/1000 users  (45.2s elapsed)
  400/1000 users  (61.5s elapsed)
  500/1000 users  (77.1s elapsed)
  600/1000 users  (93.3s elapsed)
  700/1000 users  (108.4s elapsed)
  800/1000 users  (124.1s elapsed)
  900/1000 users  (139.4s elapsed)
  1000/1000 users  (156.1s elapsed)

============================================================
Evaluated: 1,000 users in 156.1s
Skipped — inference errors: 0, unknown-asin recs: 0
------------------------------------------------------------
   K       Hit@K      NDCG@K
   5      0.0170      0.0119
  10      0.0290      0.0158
  20      0.0440      0.0194

MRR@20: 0.0127

Latency (ms) over 1000 queries:
  mean   =   156.03
  median =   126.19
  p95    =   315.24
  p99    =   367.78
  min    =    95.53
  max    =   398.16
============================================================

"We started with item-KNN (GPU-only) and SVD (CPU-only, per-category). Both showed weaknesses: KNN gave incoherent recommendations, and SVD's 0.97 NDCG turned out to be evaluation contamination — proper held-out eval gave ~0.06. We switched to implicit ALS, which has matched CPU and GPU implementations in the same library, trained on all categories combined. Same algorithm, same hyperparameters, both backends. Held-out Hit@10 went from ~3% to ~12%, and the GPU trained ~20× faster than CPU on identical data."

