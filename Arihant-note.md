# how to activate rapids env 

cd ~/Capstone
source /data/anaconda3/etc/profile.d/conda.sh
conda activate rapids-25.10


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
  --output models/full_recommender.pkl


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

# QUERY RESPONSE FORMAT 

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