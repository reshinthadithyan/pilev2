import datasets
import logging
import rich 
from rich.logging import RichHandler
from transformers import AutoTokenizer
import re
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import json

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", type=str, help="Path to the data files")
parser.add_argument("--output_dir", type=str, help="Path to the output files")

args = parser.parse_args()

data_dir = Path(args.input_dir)

# create the output directory if it does not exist
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
logger.info(f"Sucessfully loaded tokenizer...")
dataset_cats = [x.name for x in data_dir.iterdir() if x.is_dir()]

def tokenize_batch(examples,tokenizer):
    return {"length": tokenizer(examples['text'], return_length=True)['length'][0]}


def tokenize(example,tokenizer):
    return {"length": tokenizer(example['text'], return_length=True)['length'][0]}


token_dict = {}
for subset in tqdm(dataset_cats):
    logger.info(f"Starting to count with {subset}.")
    ds = datasets.load_from_disk(subset)
    ds = ds.map(tokenize,fn_kwargs={"tokenizer":tokenizer}, batched=True, batch_size=1000)
    tot_len =  sum(ds["length"])
    token_dict[subset] = tot_len
    logger.info(f"Finished counting with {subset}, it has {tot_len} tokens .")
    with open(output_dir + f"/token_count.json", "w") as f:
        json.dump(token_dict, f)




