import datasets
import logging
import rich 
from rich.logging import RichHandler
from transformers import AutoTokenizer
import re
from squeakily.helpers import english_flagged_words #list[str] with english flagged words.

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)




#TODO(reshinth) : Add filtering functions wrt length

def check_document_length_tokenized(
    document:str, #document to be analyzed
    tokenizer:AutoTokenizer, #tokenizer to tokenize the documents
    document_lower_len_threshold:int = 10, #document length threshold to filter
    document_upper_len_threshold:int = 200_000, #document length threshold to filter
) -> bool: #returns True if doument length is above threshold else False
    """
    Returns True if it's above the threshold, else returns False
    """
    tokenized = tokenizer(document).input_ids
    if len(tokenized) > document_lower_len_threshold and len(tokenized) < document_upper_len_threshold:
            return True
    else:
            return False
    
def check_document_length_words(
    document:str, #document to be analyzed
    document_lower_len_threshold:int = 10, #document length threshold to filter
    document_upper_len_threshold:int = 200_000, #document length threshold to filter
) -> bool: #returns True if doument length is above threshold else False
    """
    Returns True if it's above the threshold, else returns False
    """
    pattern  = re.compile("([\w][\w]*'?\w?)")
    tokenized : list = pattern.findall(document)
    if len(tokenized) > document_lower_len_threshold and len(tokenized) < document_upper_len_threshold:
            return True
    else:
            return False 


def check_mean_word_length(
    document:str, #document to be analyzed
    mean_word_len_lower_threshold:float = 3.0, #mean word length lower threshold to filter
    mean_word_len_upper_threshold:float = 10.0, #mean word length upper threshold to filter
) -> bool: 
    """
    Returns True if it's above the lower threshold and below the upper threshold, else returns False
    """
    pattern  = re.compile("([\w][\w]*'?\w?)")
    tokenized : list = pattern.findall(document)
    mean_word_length = sum([len(word) for word in tokenized])/len(tokenized)
    if mean_word_length > mean_word_len_lower_threshold and mean_word_length < mean_word_len_upper_threshold:
            return True
    else:
            return False
