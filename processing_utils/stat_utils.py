import datasets
import logging
import rich 
from rich.logging import RichHandler
from transformers import AutoTokenizer
import re
logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)




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

