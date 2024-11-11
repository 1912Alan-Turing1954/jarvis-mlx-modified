import warnings
import os
import logging
from transformers import logging as hf_logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# Suppress all warnings related to transformers
warnings.filterwarnings('ignore', module='transformers')

# Suppress PyTorch-specific warnings (UserWarnings and DeprecationWarnings)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")

# Disable tokenizer parallelism warnings in transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set Huggingface transformer logging to error only (no warnings or info logs)
hf_logging.set_verbosity_error()

# Set general logging to only show errors (useful for other libraries as well)
logging.basicConfig(level=logging.ERROR)

# Now, let's load the tokenizer and model
model_id = "bert-base-uncased"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(model_id)


def get_bert_feature(text, word2ph, device=None):
    global model
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
