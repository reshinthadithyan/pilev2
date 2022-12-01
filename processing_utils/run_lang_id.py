import datasets
from stat_utils import LangDetection
import argparse
import logging
import os
from rich import RichHandler
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

parser.add_argument("--input_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--lang_id_model_path", type=str, default="lid.176.bin")
parser.add_argument("--num_proc", type=int, default=1)



args = parser.parse_args()

subset_path = [os.path.join(args.input_path,i) for i in os.listdir(args.input_path)]
logger.info(f"Found {len(subset_path)} files in {args.input_path}")

lang_detection = LangDetection(args.lang_id_model_path)


def add_lang_id(example,lang_detection):
    output_dict = {}
    lang_id,score = lang_detection.detect(example["text"])
    output_dict["lang_id"] = lang_id
    output_dict["lang_score"] = score
    return output_dict

output_dict = {}


def apply_subset_path(subset_path,args):
    subset = subset_path
    dataset = datasets.load_from_disk(subset_path)
    stats_dataset = dataset.map(add_lang_id,fn_kwargs={"lang_detection":lang_detection},batched=True,batched_remove_columns=["text"])
    output_path = os.path.join(args.output_path,subset.split("/")[-1])
    stats_dataset.save_to_disk(output_path)
    calc_en_ratio = dataset.filter(lambda x: x['lang_id'] == "en")
    return {
        "en_count" : len(calc_en_ratio),
        "total_count" : len(dataset),
        "en_ratio" : len(calc_en_ratio)/len(dataset)
    } 
    


for subset in tqdm(subset_path):
    output_dict[subset] = apply_subset_path(subset)

with open(os.path.join(args.output_path,"stats.json"),"w") as f:
    json.dump(output_dict,f,indent=2)



