import lm_dataformat as lmd
import multiprocessing as mp
import argparse
import os
import datasets
import hashlib
import re
from tqdm import tqdm
import logging


PATTERN = re.compile(r"\s+")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def get_hash(example):
    """Get hash of content field."""
    return hashlib.md5(re.sub(PATTERN, "", example["doc"]).encode("utf-8")).hexdigest()

def get_hash_batch(examples):
    for example in examples:
        example["hash"] = get_hash(example)
    return examples


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="deduped_dataset/",required=True)
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count())
    parser.add_argument('--log_every',type=int,default=1_000_000) #Log the stats every 1_000_000 datapoints.
    
    args = parser.parse_args()

    input_files = os.listdir(args.input_dir)

    dataset = datasets.load_dataset("json",data_files=input_files)
    dataset = dataset.map(get_hash_batch, batched=True, batch_size=1000, num_proc=args.num_workers)
    logger.info(f"Got the dataset with {len(dataset)} examples.")

    uniques = set(dataset["train"]["hash"])

    dataset_filter = dataset.filter(filter, fn_kwargs={"uniques": uniques, "args": args},batched=True,num_proc=mp.cpu_count())
    logger.info(f"Filtered the dataset with {len(dataset_filter)} examples...")

    dedupped_archive = lmd.Archive(args.output_dir)

    # for dataset in tqdm(dataset_filter):
    count = 0
    for dataset in tqdm(dataset_filter):
        dedupped_archive.add_data(
            dataset["doc"],
            meta = {
                k:v for k,v in dataset.items() if k != "doc"
            }
        )
        
    if count % args.log_every == 0:
        logger.info(f"Unique hashes remaining: {len(uniques)}")
        dedupped_archive.commit()