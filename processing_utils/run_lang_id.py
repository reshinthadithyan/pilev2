import datasets
from stat_utils import LangDetection
import argparse
import logging
import os
from rich.logging import RichHandler
from tqdm import tqdm 
import json
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--input_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--output_stats_path", type=str)
parser.add_argument("--lang_id_model_path", type=str)
parser.add_argument("--num_proc", type=int, default=1)



args = parser.parse_args()
subset = os.listdir(args.input_path)
            
logger.info(subsets)
subset_path = [os.path.join(args.input_path,i) for i in subsets]
logger.info(f"Found {len(subset_path)} files in {args.input_path}")
logger.info(f"{subset_path}")

lang_detection = LangDetection(args.lang_id_model_path)


def add_lang_id_batch(examples):
    output = {"lang_id" : [], "lang_score" : []}
    for example in examples:
        lang_id,score = lang_detection.detect(example["text"])
        output["lang_id"].append(lang_id)
        output["lang_score"].append(float(score))
    return examples_out

def add_lang_id(example):
    lang_id,score = lang_detection.detect(example["text"])
    return {"lang_id" : lang_id, "lang_score" : float(score) }


output_stat_dict = {}


def apply_subset_path(subset_path,args):
    subset = subset_path
    dataset = datasets.load_from_disk(subset_path)
    stats_dataset = dataset.map(add_lang_id,batched=False,num_proc=128,remove_columns=["text"])
    output_path = os.path.join(args.output_path,subset.split("/")[-1])
    stats_dataset.save_to_disk(output_path)
    calc_en_ratio = stats_dataset.filter(lambda x: x['lang_id'] == "en",num_proc=128)
    stats =  {
        "subset" : subset,
        "en_count" : len(calc_en_ratio),
        "total_count" : len(dataset),
        "en_ratio" : len(calc_en_ratio)/len(dataset)
    } 
    logger.info(stats)
    return stats
    


for subset in tqdm(subset_path):

    subset_name = subset.split("/")[-1]
    logger.info(f"Running {subset_name}")
    output_stat_dict[subset_name] = apply_subset_path(subset,args)

    with open(f"stats_dump/stats_{subset_name}.json","a") as f:
        json.dump(output_stat_dict,f,indent=2)
