import dask.bag as db
import dask.dataframe as dd
import pandas as pd
import json
import os
from dask.distributed import Client, LocalCluster

def extract_fields(line):
    try:
        d = json.loads(line)
        return {
            "parent_asin": str(d.get("parent_asin", d.get("asin", ""))),
            "product_title": str(d.get("title", "")),
            "main_category": str(d.get("main_category", ""))
        }
    except:
        return {"parent_asin": "", "product_title": "", "main_category": ""}

if __name__ == "__main__":
    # Spin up a CPU cluster to maximize text parsing speed
    cluster = LocalCluster()
    client = Client(cluster)
    
    print(f"CPU Cluster Active! Dashboard: {client.dashboard_link}")
    
    input_path = "/data/tiwa25/data/meta_Electronics.jsonl"
    output_path = "/data/tiwa25/data/meta_Electronics_clean.parquet"
    
    print(f"1. Parsing messy JSON from: {input_path}")
    
    # Read text lazily and extract only what we need
    bag = db.read_text(input_path, blocksize="128MB").map(extract_fields)
    
    # Enforce strict schema
    meta_schema = pd.DataFrame({
        "parent_asin": pd.Series(dtype="object"),
        "product_title": pd.Series(dtype="object"),
        "main_category": pd.Series(dtype="object")
    })
    
    ddf = bag.to_dataframe(meta=meta_schema)
    
    # Clean empty rows and drop duplicates
    ddf = ddf[ddf["parent_asin"] != ""]
    ddf = ddf.drop_duplicates(subset=["parent_asin"])
    
    print(f"2. Writing clean, typed Parquet file to: {output_path}")
    print("This may take 10-15 minutes. Grab a coffee...")
    
    # Execute and save
    ddf.to_parquet(output_path, engine="pyarrow", write_index=False)
    
    print(f"Done! Clean metadata cached at: {output_path}")
