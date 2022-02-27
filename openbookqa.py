from datasets import load_dataset, load_metric, ClassLabel
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import transformers
import random
import pandas as pd

from pprint import pprint

import torch
import math
import time
import sys
import json
import numpy as np

ending_names = ['A', 'B', 'C', 'D']
model_chkpt = "bert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_chkpt, use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(model_chkpt)

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [ending_names.index(feature.pop(label_name)) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def choices(example):
    for dic in example['question.choices']:
        example[dic['label']] = dic['text']
    example.pop('question.choices', None)
#    example.pop('question.stem', None)
    return example

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    pprint(df.to_html())
    
def show_one(example):
    print(f"Context: {example['fact1']}")
    print(f"  A - {example['question.stem']} {example['A']}")
    print(f"  B - {example['question.stem']} {example['B']}")
    print(f"  C - {example['question.stem']} {example['C']}")
    print(f"  D - {example['question.stem']} {example['D']}")
    print(f"\nGround truth: option {example['label']}")    
    
def preprocess_function(examples):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["fact1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["question.stem"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
    
    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    
    
def main():
    
    facts = 1

    input_files = ['train_complete.jsonl','test_complete.jsonl','dev_complete.jsonl']
    if facts == 0:
        output_files = ['train_complete_d.jsonl','test_complete_d.jsonl','dev_complete_d.jsonl']
    else:
        output_files = ['train_complete_e.jsonl','test_complete_e.jsonl','dev_complete_e.jsonl']
    
    for io in range(3):
        file_name = input_files[io]
        with open(file_name) as json_file:
            json_list = list(json_file)
        for i in range(len(json_list)):
            json_str = json_list[i]
            result = json.loads(json_str)       
            print(result['fact1'])
            if facts == 0:
                result['fact1'] = ''
            json_list[i] = json.dumps(result)
        file_name = output_files[io]
        fout = open(file_name,'wt')
        for i in range(len(json_list)):
            fout.write('%s\n' % json_list[i])
        fout.close()

    batch_size = 16
    if facts == 0:
        openbookQA = load_dataset('json', data_files={'train': 'train_complete_d.jsonl', 
                                                      'validation': 'dev_complete_d.jsonl', 
                                                      'test': 'test_complete_d.jsonl'})
    else:
        openbookQA = load_dataset('json', data_files={'train': 'train_complete_e.jsonl', 
                                                      'validation': 'dev_complete_e.jsonl', 
                                                      'test': 'test_complete_e.jsonl'})
    pprint(openbookQA['train'][0])
    
    flatten = openbookQA.flatten()
    
    updated = flatten.map(choices)
    updated = updated.rename_column('answerKey', 'label')
    pprint(updated['train'][0])
    
    show_one(updated['train'][0])
    
    examples = updated['train'][:5]
    features = preprocess_function(examples)
    print(len(features["input_ids"]), len(features["input_ids"][0]), [len(x) for x in features["input_ids"][0]])   
    
    idx = 3
    [tokenizer.decode(features["input_ids"][idx][i]) for i in range(4)]    
    show_one(updated['train'][idx])
    
    encoded_datasets = updated.map(preprocess_function, batched=True)
    
    model_name = model_chkpt.split("/")[-1]
    args = TrainingArguments(f"{model_name}-finetuned-swag",
                             evaluation_strategy = "epoch",
                             learning_rate=5e-5,
                             per_device_train_batch_size=batch_size,
                             num_train_epochs=3,
                             weight_decay=0.01)
    
    accepted_keys = ["input_ids", "attention_mask", "label"]
    features = [{k: v for k, v in encoded_datasets["train"][i].items() if k in accepted_keys} for i in range(10)]
    batch = DataCollatorForMultipleChoice(tokenizer)(features)
    
    [tokenizer.decode(batch["input_ids"][8][i].tolist()) for i in range(4)]
    show_one(updated["train"][8])
    
    trainer = Trainer(model,
                      args,
                      train_dataset=encoded_datasets["train"],
                      eval_dataset=encoded_datasets["validation"],
                      tokenizer=tokenizer,
                      data_collator=DataCollatorForMultipleChoice(tokenizer),
                      compute_metrics=compute_metrics)
    
    trainer.train()
    print('\n\n\n\n')
    print('test set:')
    print('\n\n\n\n')
    final_eval = trainer.evaluate(eval_dataset=encoded_datasets['test'])
    print(final_eval)
         
if __name__ == "__main__":
    main()
