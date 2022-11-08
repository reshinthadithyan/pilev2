import logging
import lm_dataformat as lmd
import multiprocessing as mp
from transformers import GPTNeoXTokenizerFast
import argparse
import json
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, default="stats_dump/",required=True)
parser.add_argument('--num_workers', type=int, default=mp.cpu_count())
parser.add_argument('--log_every',type=int,default=1000) #Log the stats every 1000 datapoints.
parser.add_argument('--file_chunk_size',type=int,default=10) #Log the stats every 1000 datapoints.

args = parser.parse_args()

logger = logging.getLogger(__name__)

#PREPARE
if not os.path.exists(args.output_dir):
    logger.info(f"Creating output directory {args.output_dir} since it doesn't exist")
    os.makedirs(args.output_dir)



if __name__ == "__main__":
    pool = mp.Pool(args.num_workers) #Init multiprocessing pool
    ckpt_path = os.path.join(args.output_dir,"checkpoint.json")
    if os.path.exists(ckpt_path):
        logger.info(f"Found checkpoint file {ckpt_path}, loading it.")
        with open(ckpt_path,"r") as f:
            ckpt = json.load(f)
    else:
        ckpt = {
            "index" : 0
        }

    def make_chunks(l,n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    logger.info(f"Starting from index {ckpt['index']}")
    input_file_path = os.listdir(args.input_dir)[ckpt['index']:] #start from the checkpoint index

    file_chunks = make_chunks(input_file_path,args.log_every)

    for file_chunk in tqdm(file_chunks):
        pass