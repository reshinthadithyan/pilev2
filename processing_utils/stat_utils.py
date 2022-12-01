import datasets
import logging
import rich 
from rich.logging import RichHandler
from transformers import AutoTokenizer
import re
import fasttext
import os
logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)




PATH_FASTTEXT = ""


def get_words(text: str) -> list:
    """custom regex to extract all the words in a string"""
    return re.findall(r'\w+', text.lower())

def load_fasttext_model(path_fasttext_model:str=PATH_FASTTEXT):
    return fasttext.load_model(path_fasttext_model)


def get_fasttext_info(line, model_lang_id):
    """The line should be in lower case and without \n in it."""
    pred = model_lang_id.predict(line)
    lang_pred_fasttext_id = pred[0][0].replace("__label__", "")
    score_pred = pred[1][0]
    return lang_pred_fasttext_id, score_pred


def get_all_fasttext_info(document, model_lang_id):
    document = document.lower()
    lang_pred_fasttext_id, score_pred = get_fasttext_info(
        document.replace("\n", " "), model_lang_id
    )
    info = {
        "lang_pred_fasttext_id": lang_pred_fasttext_id,
        "score_pred": score_pred,
        "on_lines": [
            {
                "id_line": id_line,
                "number_caracters_line": len(line),
                "lang_pred_fasttext_id_line": result_fasttext_line[0],
                "score_pred_line": result_fasttext_line[1],
            }
            for id_line, line in enumerate(document.split("\n"))
            for result_fasttext_line in [get_fasttext_info(line, model_lang_id)]
        ],
    }
    return info


class LangDetection:
    #adapted from https://github.com/bigcode-project/bigcode-analysis/blob/main/data_analysis/python_data_analysis/nl_language_identification/language_identifier.py
    def __init__(self,lang_model_path:str) -> None:
        if os.path.file_exists(lang_model_path):
            self.lang_model_path = lang_model_path
            self.model = load_fasttext_model(self.lang_model_path)#"src/lid.176.bin")
            logger.info(f"Loaded model from {self.lang_model_path}")


    def detect(self, text: str) -> str:
        """
        Detects the language of the text
        args:
            text (str) : Text to detect the language

        returns:
            language (str) : Predicted Language of the text
            score_pred (str) : confidence of the prediction

        """
        text = text.lower()

        fasttext_pred = get_all_fasttext_info(
            text, self.model
        )
        return fasttext_pred["lang_pred_fasttext_id"], fasttext_pred["score_pred"]

    def detect_batch(self, text_list: list[str]) -> list[str]:
        return [self.detect(text) for text in text_list]





#TODO(reshinth) : Add stats functions wrt length

def calc_document_length_tokenized(
    document:str, #document to be analyzed
    tokenizer:AutoTokenizer, #tokenizer to tokenize the documents
) -> dict: #returns True if doument length is above threshold else False
    """
    Returns True if it's above the threshold, else returns False
    """
    tokenized = tokenizer(document).input_ids
    return len(tokenized)    
def calc_document_length_words(
    document:str, #document to be analyzed
) -> int: #returns True if doument length is above threshold else False
    """
    Returns True if it's above the threshold, else returns False
    """
    pattern  = re.compile("([\w][\w]*'?\w?)")
    tokenized : list = pattern.findall(document)
    return len(tokenized)

def calc_mean_word_length(
    document:str, #document to be analyzed
) -> int: #returns the mean word length of words in the document from Gopher
    """
    Returns True if it's above the threshold, else returns False
    """
    pattern  = re.compile("([\w][\w]*'?\w?)")
    tokenized : list = pattern.findall(document)
    return sum([len(word) for word in tokenized])/len(tokenized)

def get_tok_len(
    document:str,
    tokenizer:AutoTokenizer
) -> int:
    """
    Returns the length of the tokenized document
    """
    return len(tokenizer(document).input_ids)

