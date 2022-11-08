import logging
import lm_dataformat as lmd
import multiprocessing as mp
from transformers import GPTNeoXTokenizerFast
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, default="stats_dump/",required=True)
parser.add_argument('--num_workers', type=int, default=mp.cpu_count())
parser.add_argument('--log_every',type=int,default=1000) #Log the stats every 1000 datapoints.
args = parser.parse_args()

logger = logging.getLogger(__name__)

#PREPARE
if not os.path.exists(args.output_dir):
    logger.info(f"Creating output directory {args.output_dir} since it doesn't exist")
    os.makedirs(args.output_dir)

#load the tokenizer here
tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20B')

#All the stats related function goes here

def get_document_length(doc:str)-> int:
    return len(tokenizer.encode(doc))


stats_to_be_applied = [
    get_document_length
]

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

    logger.info(f"Starting from index {ckpt['index']}")

    rdr = lmd.Reader(args.input_dir) #Read the input file 
