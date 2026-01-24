# %%
!uv pip install -q datasets requests torch peft bitsandbytes transformers trl accelerate sentencepiece wandb matplotlib


# %%
import datetime

import torch

print("=== COLAB RUNTIME ENVIRONMENT ===")
print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Hardware
if torch.cuda.is_available():
    print(f"Machine Type: {torch.cuda.get_device_name()}")
else:
    print("Machine Type: CPU only")

print(f"\n=== SOFTWARE VERSIONS ===")
!uv --version
!uv pip list


# %%
# imports
# With much thanks to Islam S. for identifying that there was a missing import!

import os
from datetime import datetime

import torch
import wandb
from datasets import load_dataset
from google.colab import userdata
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# %%
# Constants

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "antonawinkler"

# Data

DATASET_NAME = f"{HF_USER}/pricer-data"
MAX_SEQUENCE_LENGTH = 182

# Run name for saving the model in the hub

RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Hyperparameters for QLoRA

LORA_R = 32
LORA_ALPHA = 2 * LORA_R
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # attention layers
# TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # attention + mlp layers
LORA_DROPOUT = 0.1
QUANT_4_BIT = True

# Hyperparameters for Training

EPOCHS = 3
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
OPTIMIZER = "paged_adamw_32bit"

# Admin config - note that SAVE_STEPS is how often it will upload to the hub

STEPS = 50
SAVE_STEPS = 2000
LOG_TO_WANDB = True

# 400_000 will be the length of the train set
steps_per_epoch = 400_000 // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
EVAL_STEPS = steps_per_epoch // 5
SAVE_STEPS = steps_per_epoch

%matplotlib inline


# %%
# changes by antonawinkler
# We add more target modules ("gate_proj", "up_proj", "down_proj")

TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
]  # , "gate_proj", "up_proj", "down_proj"]

# 400_000 is the length of the train set
steps_per_epoch = 400_000 // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
EVAL_STEPS = steps_per_epoch // 5
SAVE_STEPS = steps_per_epoch


# %%
HUB_MODEL_NAME


# %% [markdown]
# ### Log in to HuggingFace and Weights & Biases
#
# If you don't already have a HuggingFace account, visit https://huggingface.co to sign up and create a token.
#
# Then select the Secrets for this Notebook by clicking on the key icon in the left, and add a new secret called `HF_TOKEN` with the value as your token.
#
# Repeat this for weightsandbiases at https://wandb.ai and add a secret called `WANDB_API_KEY`


# %%
# Log in to HuggingFace

hf_token = userdata.get("HF_TOKEN")
login(hf_token, add_to_git_credential=True)


# %%
# Log in to Weights & Biases
wandb_api_key = userdata.get("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key
wandb.login()

# Configure Weights & Biases to record against our project
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"] = "checkpoint" if LOG_TO_WANDB else "end"
os.environ["WANDB_WATCH"] = "gradients"


# %% [markdown]
#


# %%
# This conversion is done to replace the data collator in Ed Donner's original version.
# The model is only trained on the completion.
def convert_to_prompt_completion(example):
    beforePrice, priceIsDollar, price = example["text"].partition("Price is $")
    return {"prompt": beforePrice + priceIsDollar, "completion": price}


train = load_dataset(DATASET_NAME, split="train[:100%]").map(
    convert_to_prompt_completion, remove_columns=["text", "price"]
)
validation = load_dataset(DATASET_NAME, split="validation[:100%]").map(
    convert_to_prompt_completion, remove_columns=["text", "price"]
)


# %%
if LOG_TO_WANDB:
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)


# %% [markdown]
# ## Now load the Tokenizer and Model
#
# The model is "quantized" - we are reducing the precision to 4 bits.


# %%
# pick the right quantization

if QUANT_4_BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
    )


# %% [markdown]
# # If you want to restart from a checkpoint, look for "Restart" below.


# %%
# Load the Tokenizer and the Model

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")


# %% [markdown]
# # AND NOW
#
# ## We set up the configuration for Training
#
# We need to create 2 objects:
#
# A LoraConfig object with our hyperparameters for LoRA
#
# An SFTConfig with our overall Training parameters


# %%
# First, specify the configuration parameters for LoRA

lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

# Next, specify the general configuration parameters for training

train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    max_length=MAX_SEQUENCE_LENGTH,
    completion_only_loss=True,
    assistant_only_loss=False,
    packing=False,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,
)

fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    eval_dataset=validation,
    peft_config=lora_parameters,
    args=train_parameters,
)


# %% [markdown]
# # Mask the extra EOS
#
# The trainer adds an EOS to the "completion" field.
# Instead of three tokens (for LLaMA) we get for, e.g.
# 42.00 with tokens "42", ".", and "00" we get
# 42.00\EOS with tokens "42", ".", "00", and "\EOS".
#
# My model results deteriorated slightly with the extra "\EOS" token
# (i.e. the predicted prices were somewhat worse on average.)
# In the following it is masked so that it is not used for training. Thus,
# I managed to reproduce the results I got with the original code.
# I did not find a way to configure this in the trainer.


# %%
base_collator = fine_tuning.data_collator
eos_id = fine_tuning.model.config.eos_token_id


def mask_trailing_eos(batch_examples):
    batch = base_collator(batch_examples)  # uses TRL's internal collator
    if eos_id is None:
        return batch

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # For each row, find last position that contributes to loss and mask it if it's EOS
    for i in range(labels.size(0)):
        contributing = (labels[i] != -100).nonzero(as_tuple=False)
        if len(contributing) == 0:
            continue
        last_idx = contributing[-1].item()
        if input_ids[i, last_idx].item() == eos_id:
            labels[i, last_idx] = -100
    return batch


# %%
fine_tuning.data_collator = mask_trailing_eos


# %% [markdown]
# ## In the next cell, we kick off fine-tuning!
#
# This will run for some time, uploading to the hub every SAVE_STEPS steps.
#
# After some time, Google might stop your colab. For people on free plans, it can happen whenever Google is low on resources. For anyone on paid plans, they can give you up to 24 hours, but there's no guarantee.
#
# If your server is stopped, you resume from your last save in Weights & Biases as described below.


# %%
# Fine-tune!
fine_tuning.train()

# Push our fine-tuned model to Hugging Face
fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")


# %%
if LOG_TO_WANDB:
    wandb.finish()


# %% [markdown]
# # Resume
#
# As opposed to the approach in the lecture, see https://colab.research.google.com/drive/1qGTDVIas_Vwoby4UVi2vwsU0tHXy8OMO#scrollTo=R_O04fKxMMT-, we resume by loading the checkpoint from Weights & Biases (W&B).
#
# By configuring LOG_TO_WANDB = True we log more than just the model which is also stored on Huggingface. In particular, we also store the
#
# * optimizer.pt – State of the optimizer (Adam, AdamW, etc.) at the checkpoint, useful for resuming training.
#
# * rng_state.pth – Captures the state of random number generators (CPU/GPU), ensuring reproducibility when resuming training.
#
# By making use of these files we can start exactly where we left off. I find that this can have a significant impact on the results.


# %%
# Login to W&B and download artifact (same as your code)
wandb.login()

local_path = "./downloaded_model_v2/"  # edit
entity = "antonawinkler"  # edit
project = "pricer"  # edit
artifact_name = "model-2025-09-16_18.51.56"  # edit
artifact_type = "model"  # edit
artifact_version = "v2"  # edit

os.makedirs(local_path, exist_ok=True)


api = wandb.Api()
artifact = api.artifact(f"{entity}/{project}/{artifact_name}:{artifact_version}")
artifact_dir = artifact.download(root=local_path)
print(f"Downloaded artifact to {artifact_dir}")

# Load base model and tokenizer (but DON'T load PEFT model manually)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

# Configure LoRA parameters
lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

# Configure training parameters
train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    max_length=MAX_SEQUENCE_LENGTH,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,
)

# Create trainer with base model (let it handle PEFT loading)
fine_tuning = SFTTrainer(
    model=base_model,  # Pass base model, not pre-loaded PEFT model
    train_dataset=train,
    eval_dataset=validation,
    peft_config=lora_parameters,
    args=train_parameters,
)

# Resume training from checkpoint - this will load both model and trainer state


# %%
# make sure to run cell with mask_trailing_eos function above
fine_tuning.data_collator = mask_trailing_eos


# %%
fine_tuning.train(resume_from_checkpoint=artifact_dir)

# Push the fine-tuned model
fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")
