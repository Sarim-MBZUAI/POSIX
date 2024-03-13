#######################################################################################################################################
# PURPOSE: This script is to be used to generate responses of an autoregressive model like Llama-2, Mistral on a given set of prompts.
# The dataset should be in the 🤗 datasets format.
# Specifically, the dataset should be a list of dictionaries with four keys - "id", "task", "prompt" and "target".
# [
#     {
#         "id": 1,
#         "task": task_1,
#         "prompt": prompt_1,
#         "target": target_1
#     },
#     ...
# ]
# The output will be a json file containing list of dictionaries with five keys - "id", "task", "prompt", "target" and "response".
#######################################################################################################################################

import os
import json
import logging
import argparse
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import datasets
from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

logger=get_logger(__name__)

@dataclass
class DataCollatorForInference:
    """Collate examples for LLM inference"""
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask=tuple([torch.tensor(feature[key][::-1]) for feature in features] for key in ["input_ids", "attention_mask"])
        input_ids=rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
        attention_mask=rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0).flip(dims=[1])
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

def parse_args():
    parser=argparse.ArgumentParser(description="Generate responses from an autoregressive model on given set of prompts")

    parser.add_argument(
        "--load_data_from_disk",
        action="store_true",
        help="If provided, the dataset will be loaded from disk"
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The path to the dataset in 🤗 huggingface format"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        required=True,
        help="The model name or path to the model"
    )
    parser.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        help="If provided, the model will use the flash_attention_2 implementation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1,
        required=True,
        help="The maximum number of tokens to generate"
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        required=True,
        help="The batch size to use for inference"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="The seed to use for reproducibility"
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="llm_responses",
        required=True,
        help="The name of the output dataset"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()

    accelerator=Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now
    if args.seed is not None:
        logger.info(f"Setting the seed to {args.seed}")
        set_seed(args.seed)
    
    logger.info("Loading Dataset")
    if args.load_data_from_disk:
        logger.info("Loading dataset from disk")
        raw_dataset=load_from_disk(args.dataset_name_or_path)
    else:
        logger.info("Loading dataset from huggingface hub")
        raw_dataset=load_dataset(args.dataset_name_or_path)

    logger.info("Loading Tokenizer")
    tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path)
    logger.info("Setting padding token")
    if tokenizer.pad_token:
        logger.info(f"Padding token already set to {tokenizer.pad_token}")
        pass
    elif tokenizer.unk_token:
        logger.info(f"Setting pad token to {tokenizer.unk_token}")
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.eos_token:
        logger.info(f"Setting pad token to {tokenizer.eos_token}")
        tokenizer.pad_token=tokenizer.eos_token
    else:
        logger.info(f"Adding special token <|pad|> as pad token")
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    logger.info(f"The final pad token details: {tokenizer.pad_token_id}, {tokenizer.pad_token}")
    logger.info("Setting padding side to left")
    tokenizer.padding_side="left"

    logger.info("Loading Model")
    if args.use_flash_attention_2:
        logger.info("Using flash_attention_2")
        model=AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2"
        )
    else:
        logger.info("Not using flash_attention_2")
        model=AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype="auto"
        )

    def tokenize_prompts(examples):
        return tokenizer(examples["prompt"])

    logger.info("Tokenizing the dataset")
    column_names=raw_dataset.column_names
    with accelerator.main_process_first():
        tokenized_dataset=raw_dataset.map(
            tokenize_prompts,
            batched=True,
            num_proc=8,
            remove_columns=column_names,
            desc="Tokenizing all prompts"
        )
    
    logger.info("Instantiating Data Collator")
    data_collator=DataCollatorForInference(tokenizer)
    inference_dataloader=DataLoader(
        tokenized_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_batch_size,
        pin_memory=True,
        num_workers=8
    )

    model, inference_dataloader=accelerator.prepare(model, inference_dataloader)
    
    total_batch_size = args.per_device_batch_size * accelerator.num_processes

    logger.info("***** Running Inference *****")
    logger.info(f"Number of prompt = {len(tokenized_dataset)}")
    logger.info(f"Instantaneous batch size per device = {args.per_device_batch_size}")
    logger.info(f"Total inference batch size = {total_batch_size}")

    progress_bar=tqdm(range(len(inference_dataloader)), disable=not accelerator.is_local_main_process)
    
    responses=[]
    model.eval()
    for batch in inference_dataloader:
        N_prompt=batch["input_ids"].shape[1]
        with torch.no_grad():
            generated_tokens=accelerator.unwrap_model(model).generate(**batch, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
            generated_tokens=generated_tokens[:, N_prompt:]
            generated_tokens=accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            generated_tokens=accelerator.gather_for_metrics(generated_tokens)
            generated_tokens=generated_tokens.cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            responses.extend(decoded_preds)
            progress_bar.update(1)

    logger.info("Creating the final dataset")
    final_dataset=[
        {
            "id": raw_dataset[i]["id"],
            "task": raw_dataset[i]["task"],
            "prompt": raw_dataset[i]["prompt"],
            "target": raw_dataset[i]["target"],
            "response": responses[i]
        }
        for i in range(len(tokenized_dataset))
    ]
    with open(args.save_as, "w") as f:
        json.dump(final_dataset, f, indent=4)

if __name__=="__main__":
    main()