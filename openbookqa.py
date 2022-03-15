from datasets import load_dataset, load_metric, ClassLabel
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import transformers
import random
import pandas as pd
import pdb
from pprint import pprint

import torch
import math
import time
import sys
import json
import numpy as np
import spacy 
#spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")


DIM_SIZES = [50, 100, 200, 300]
DIM_SIZE = ''
ending_names = ['A', 'B', 'C', 'D']
model_chkpt = "bert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_chkpt, use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(model_chkpt)
glove_embeddings = {}
noun_dict_train = {}
noun_dict_test = {}
noun_dict_valid = {}

def cos_sim(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    cosine_sim = dot_product / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return cosine_sim

def load_glove_embeddings(dim_size):
    global glove_embeddings
    global DIM_SIZE
    if int(dim_size) not in DIM_SIZES:
        print("Please select a dimension size that is 50, 100, 200, or 300")
    else:
        DIM_SIZE = dim_size
        filename = 'glove/glove.6B.' + str(dim_size) + 'd.txt'
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.readlines()
            for line in content:
                content_line_split = line.split()
                token = content_line_split[0]
                embedding = content_line_split[1:]
                embedding = np.array(embedding, dtype=np.float64)
                glove_embeddings[token] = embedding
            file.close()

def get_token_embeddings(token_seq):
    # Create np array of size len(token_seq) * embedding_dims
    try:
        embedding = np.zeros((len(token_seq), DIM_SIZE))
    except Exception as e:
        print(Exception)
        pdb.set_trace()
    # Loop through tokens in seq
    for idx, token in enumerate(token_seq):
        # Try to get embedding
        try:
            embedding[idx] = glove_embeddings[token]
        except:
            # Otherwise embed as unk token
            embedding[idx] = glove_embeddings['unk']

    return embedding

def get_spacy_info(token_seq, info):
    nouns = []
    # Run spaCy over string
    doc = nlp(token_seq)
    # lemmatized text, pos for nouns in string
    for token in doc:
        if info == None:
            nouns.append(token.lemma_)
        else:
            if token.pos_ in info:
                nouns.append(token.lemma_)
    if len(nouns) == 0:
        pdb.set_trace()
    return nouns

def fill_noun_dicts(dataset:dict):
    # Globals
    global noun_dict_train
    global noun_dict_test
    global nound_dict_valid
    # Get list of answer keys
    answer_list = ['A', 'B', 'C', 'D']
    # Get Dataset Dicts
    train_dict = dataset['train']
    test_dict = dataset['test']
    valid_dict = dataset['validation']
    # loop count
    loop_count = 0
    for ex_idx in range(len(train_dict)):
        # test
        if loop_count < len(test_dict):
            # Get question stem && add to nouns to dict
            question_stem = test_dict[ex_idx]['question.stem']
            noun_dict_test[question_stem] = get_spacy_info(question_stem)
            if len(noun_dict_test[question_stem]) == 0:
                pdb.set_trace()
            # Loop through all 4 possible answers and add nouns to dict
            for answer_letter in answer_list:
                answer = test_dict[ex_idx][answer_letter]
                key = (question_stem, answer)
                noun_dict_test[key] = get_spacy_info(answer)

        # Valid
        if loop_count < len(valid_dict):
                # Get question stem && add to nouns to dict
                question_stem = valid_dict[ex_idx]['question.stem']
                noun_dict_valid[question_stem] = get_spacy_info(question_stem)
                if len(noun_dict_valid[question_stem]) == 0:
                    pdb.set_trace()
                # Loop through all 4 possible answers and add nouns to dict
                for answer_letter in answer_list:
                    answer = valid_dict[ex_idx][answer_letter]
                    key = (question_stem, answer)
                    noun_dict_valid[key] = get_spacy_info(answer)

        # Test
        question_stem = train_dict[ex_idx]['question.stem']
        noun_dict_train[question_stem] = get_spacy_info(question_stem)
        if len(noun_dict_train[question_stem]) == 0:
                pdb.set_trace()
        for answer_letter in answer_list:
            answer = train_dict[ex_idx][answer_letter]
            key = (question_stem, answer)
            noun_dict_train[key] = get_spacy_info(answer)

        # Incr Loop_Count 
        loop_count += 1

def trunc_and_calc_sim(stem, ans):
    # Check if stem is larger than ans
    if len(stem) > len(ans):
        # Get ans embeds
        ans_embeddings = np.ravel(get_token_embeddings(ans))
        # Truncate stem until same length as ans
        while len(stem) != len(ans):
            # Random Pop
            stem.pop(random.randint(0, len(stem)-1))
        # Get Stem Embeddings
        stem_embeddings = np.ravel(get_token_embeddings(stem))
        # Get Cosine Similarity
        similarity = cos_sim(stem_embeddings, ans_embeddings)
        # Return
        return similarity

    # Check if ans is larger than stem
    elif len(ans) > len(stem):
        # Get stem embeds
        stem_embeddings = np.ravel(get_token_embeddings(stem))
        # Trunc answer until same length as stem
        while len(stem) != len(ans):
            try:
                ans.pop(random.randint(0, len(stem)-1))
            except:
                pdb.set_trace()
        # Get Ans Embeds
        ans_embeddings = np.ravel(get_token_embeddings(ans))
        # Get sim
        similarity = cos_sim(stem_embeddings, ans_embeddings)
        # Return 
        return similarity
    else:
        # Get Embeds
        stem_embeddings = np.ravel(get_token_embeddings(stem))
        ans_embeddings = np.ravel(get_token_embeddings(ans))
        # Get Similarity
        similarity = cos_sim(stem_embeddings, ans_embeddings)
        return similarity

def truncate_and_eval(data_set, stem, a, b, c, d):
    global noun_dict_train
    global noun_dict_test
    global noun_dict_valid 
    # Get Correct Data Dict
    if data_set == 'train':
        data_dict = noun_dict_train
    if data_set == 'test':
        data_dict = noun_dict_test
    if data_set == 'valid':
        data_dict = noun_dict_valid
    # Get Noun List 
    stem_nouns = data_dict[stem]
    ans_noun_list = [data_dict[(stem, a)], data_dict[(stem, b)], data_dict[(stem, c)], data_dict[(stem, d)]]
    similarities = []
    ans_letters = ['A', 'B', 'C', 'D']
    # Truncate
    for i in range(len(ans_noun_list)):
        # Add tuple to similarities (answer_letter, similaritiy_score)
        similarities.append((ans_letters[i], trunc_and_calc_sim(stem_nouns, ans_noun_list[i])))
    # Return similarities
    return similarities
    
def eliminate_choice(dataset: dict):
    # Globals
    global noun_dict_train
    global noun_dict_test
    global noun_dict_valid
    # Get Dataset Dicts
    train_dict = dataset['train']
    test_dict = dataset['test']
    valid_dict = dataset['validation']

    # Establish loop count
    loop_count = 0
    # loop through train dict
    for ex_idx in range(len(train_dict)):
        # Check if within len test
        if loop_count < len(test_dict):
            # Get Stem and Question Choices
            stem = test_dict[ex_idx]['question.stem']
            choice_a = test_dict[ex_idx]['A']
            choice_b = test_dict[ex_idx]['B']
            choice_c = test_dict[ex_idx]['C']
            choice_d = test_dict[ex_idx]['D']
            # Truncate and eval
            scores = truncate_and_eval('test', stem, choice_a, choice_b, choice_c, choice_d) 
            # Find max similarity (stem, answer)
            scores.sort(key = lambda x:x[1]) 
            kept_scores = scores[1:]
            # Remove least similar answer
            answer_idx_to_remove = scores[0][0]
            test_dict[ex_idx].pop(answer_idx_to_remove)
            
        # update valid
        if loop_count < len(valid_dict):
            # Get Stem and Question Choices
            stem = valid_dict[ex_idx]['question.stem']
            choice_a = valid_dict[ex_idx]['A']
            choice_b = valid_dict[ex_idx]['B']
            choice_c = valid_dict[ex_idx]['C']
            choice_d = valid_dict[ex_idx]['D']
            # Truncate and eval
            scores = truncate_and_eval('valid', stem, choice_a, choice_b, choice_c, choice_d) 
            # Find max similarity (stem, answer)
            scores.sort(key = lambda x:x[1]) 
            kept_scores = scores[1:]
            # Remove least similar answer
            answer_idx_to_remove = scores[0][0]
            valid_dict[ex_idx].pop(answer_idx_to_remove)
            

        # Update Train 
        # Dict to hold similarity scores
        similarity_dict = {}
        # Get Stem and Question Choices
        stem = train_dict[ex_idx]['question.stem']
        choice_a = train_dict[ex_idx]['A']
        choice_b = train_dict[ex_idx]['B']
        choice_c = train_dict[ex_idx]['C']
        choice_d = train_dict[ex_idx]['D']
        # Truncate and eval
        scores = truncate_and_eval('train', stem, choice_a, choice_b, choice_c, choice_d) 
        # Find max similarity (stem, answer)
        scores.sort(key = lambda x:x[1]) 
        kept_scores = scores[1:]
        # Remove least similar answer
        answer_idx_to_remove = scores[0][0]
        train_dict[ex_idx].pop(answer_idx_to_remove)
        # Incr Loop Count
        loop_count += 1
        
    return dataset

def eliminate_choice_new(json_data):
    # Load Data
    data = json.loads(json_data)
    question = data['question']
    stem = question['stem']
    choices = question['choices']
    similarities = []
    for i in range(4):
        stem_info = get_spacy_info(stem)
        choice_info = get_spacy_info(choices[i]['text'])
        similarities.append((choices[i]['label'], trunc_and_calc_sim(stem_info, choice_info), i))
    similarities.sort(key = lambda x:x[1])
    # Check if label to be deleted is same as ansKey
    if similarities[0][0] == data['answerKey']:
        # If so, delete 2nd lowest sim
        len_del_str = len(choices[similarities[1][2]]['text'].split())
        deleted = {}
        deleted['text'] = 'deleted' 
        deleted['label'] = similarities[1][0]
        choices[similarities[1][2]] = deleted
        # Update w/ new Choices
        question.update({'choices':choices})
        # Return Data
        return json.dumps(data), 'swap_delete'
    # If not
    else:
        # Delete 
        len_del_str = len(choices[similarities[0][2]]['text'].split())
        deleted = {}
        deleted['text'] = 'deleted' #' '.join(['deleted' for i in range(len_del_str)])
        deleted['label'] = similarities[0][0]
        choices[similarities[0][2]] = deleted
        # Update w/ new Choices
        question.update({'choices':choices})
        # Return Data
        return json.dumps(data), 'reg_delete'

def process_jsonl_facts(jsonl_files):
    # List of Outfile names
    output_files = ['train_complete_e_edited.jsonl', 'test_complete_e_edited.jsonl', 'dev_complete_e_edited.jsonl']
    # Load additional facts and keywords for additional facts
    facts_keywords, facts = load_cs_facts(None)
    # Loop through json files
    for jsonl_idx, jsonl_file in enumerate(jsonl_files):
        # Read the JSON into a DF
        json_data = pd.read_json(jsonl_file, lines = True)
        # Split Keywords into it's own column
        json_data['keywords'] = json_data['question'].to_frame().apply(lambda x: stem_to_keywords(x), axis = 1)
        # Open Jsonl file
        with open(jsonl_file, 'r') as json_file:
            # Open Outfile
            with open(output_files[jsonl_idx], 'w') as out_json:
                json_list = list(json_file)
                # Loop through examples
                rel_facts_dict_1 = {}
                rel_facts_dict_2 = {}
                rel_facts_dict_3 = {}
 
                for idx, example in enumerate(json_list):
                    ex_keywords = json_data['keywords'][idx]
                    try:
                        rel_facts = find_related_facts(ex_keywords, facts_keywords, 0.6)
                    except:
                       rel_facts = ['', '', ''] 
                    for fact_idx, fact in enumerate(rel_facts):
                            # Add to Fact
                        if fact_idx == 0:
                            rel_facts_dict_1[idx] = facts[fact] if fact != '' else ''
                        if fact_idx == 1:
                            rel_facts_dict_2[idx] = facts[fact] if fact != '' else ''
                        if fact_idx == 2:
                            rel_facts_dict_3[idx] = facts[fact] if fact != '' else ''

                # BP after both loops
                #df_facts_dict_1 = pd.DataFrame.from_dict(rel_facts_dict_1)
                #df_facts_dict_2 = pd.DataFrame.from_dict(rel_facts_dict_2)
                #df_facts_dict_3 = pd.DataFrame.from_dict(rel_facts_dict_3)
                    example = json.loads(example)
                    if rel_facts_dict_1[idx] != '':
                        example['fact2'] = rel_facts_dict_1[idx]
                    if rel_facts_dict_2[idx] != '':
                        example['fact3'] = rel_facts_dict_2[idx]
                    if rel_facts_dict_3[idx] != '':
                        example['fact4'] = rel_facts_dict_3[idx]
                    example = json.dumps(example)
                    out_json.write(example + '\n')


            out_json.close()
            json_file.close()

def find_related_facts(ex_keywords, fact_keywords, threshold):
    try:
        similarities = list(map(lambda x: trunc_and_calc_sim(x, ex_keywords), fact_keywords))
    except Exception as e: 
        print(Exception)
    good_rows = []
    for idx, sim in enumerate(similarities):
        if sim >= threshold:
            good_rows.append(idx)
    
    # Return Random Sample of 3
    return random.sample(good_rows, 3)

     

def stem_to_keywords(df_row):
    df_row = df_row.to_dict()
    question = df_row['question']['stem']

    return get_spacy_info(question, ['NOUN', 'PRON', 'ADJ', 'VERB', 'PROPN', 'ADV'])


def load_cs_facts(data_dict):
    file_name = 'data/OpenBookQA-V1-Sep2018/Data/Additional/crowdsourced-facts.txt'
    with open(file_name) as file:
        facts = file.readlines()
    file.close()
    facts = list(map(lambda x: x.replace('\n', ''), facts))
    facts_keywords = list(map(lambda x : get_spacy_info(x, ['NOUN', 'PRON', 'ADJ', 'VERB', 'PROPN', 'ADV']), facts))
    facts_keywords = list(map(lower_list, facts_keywords))

    return facts_keywords, facts

def lower_list(list_in):
    return_lst = []
    for item in list_in:
        return_lst.append(item.lower())
    return return_lst 


def process_json(jsonl_files):
    num_deleted = 0
    # Get JSONL as input
    output_files = ['train_complete_d_edited.jsonl','test_complete_d_edited.jsonl','dev_complete_d_edited.jsonl']
    # Load it
    for jsonl_idx, jsonl_file in enumerate(jsonl_files):
        with open(jsonl_file, 'r') as json_file:
            with open(output_files[jsonl_idx], 'w') as out_json:
                json_list = list(json_file)
                out_json_list = []
                for idx, json_data in enumerate(json_list):
                    # Load & Process
                    new_json_data = eliminate_choice_new(json_data) 
                    if new_json_data == 'delete':
                        # Skip example
                        num_deleted += 1
                        pass
                    else:
                        # Overwrite old jsonl file 
                        out_json.write(new_json_data + "\n")
                # Close Files
                json_file.close() 
                out_json.close()
                # Print Number of Examples Removed
                print(f"\nDeleted {num_deleted} from file: {jsonl_file}\n")
                # Reset num_deleted
                num_deleted = 0

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
        if "fact2" in examples:
            fact2_sentences = [[context] * 4 for context in examples["fact2"]] 
        else:
            fact2_sentences = None
        if "fact3" in examples:
            fact3_sentences = [[context] * 4 for context in examples["fact3"]] 
        else:
            fact3_sentences = None
        if "fact4" in examples:
            fact4_sentences = [[context] * 4 for context in examples["fact4"]] 
        else:
            fact4_sentences = None

        # Grab all second sentences possible for each context.
        question_headers = examples["question.stem"]
        second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
        
        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        if fact2_sentences:
            fact2_sentences = sum(fact2_sentences, [])
        if fact3_sentences:
            fact3_sentences = sum(fact3_sentences, [])
        if fact4_sentences:
            fact4_sentences = sum(fact4_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        # Un-flatten
        return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    
    
def main():
    
    facts = 1
    flag = '' 
    input_files = ['data/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl','data/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl','data/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl']
    if facts == 0:
        output_files = ['train_complete_d.jsonl','test_complete_d.jsonl','dev_complete_d.jsonl']
        # Eliminate least similar answer choice from .jsonl 
        load_glove_embeddings(50)
        process_json(output_files)
        flag = 0
    else:
        output_files = ['train_complete_e.jsonl','test_complete_e.jsonl','dev_complete_e.jsonl'] 
        load_glove_embeddings(50)
        process_jsonl_facts(output_files)
        flag = 0
    
    for io in range(3):
        file_name = input_files[io]
        with open(file_name) as json_file:
            json_list = list(json_file)
        for i in range(len(json_list)):
            json_str = json_list[i]
            #pdb.set_trace()
            result = json.loads(json_str)       
            #print(result['fact1'])
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
        openbookQA = load_dataset('json', data_files={'train': 'train_complete_d_edited.jsonl', 
                                                      'validation': 'dev_complete_d_edited.jsonl', 
                                                      'test': 'test_complete_d_edited.jsonl'})
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
    #print(len(features["input_ids"]), len(features["input_ids"][0]), [len(x) for x in features["input_ids"][0]])   
    
    idx = 3
    # Facts == 1
    if facts != 0:
        pdb.set_trace()
        [tokenizer.decode(features["input_ids"][idx][i]) for i in range(4)]    
    # Facts == 0
    else:
        [tokenizer.decode(features["input_ids"][idx][i]) for i in range(3)]    
    #show_one(updated['train'][idx])
    
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
   
    if facts != 0:
        [tokenizer.decode(batch["input_ids"][8][i].tolist()) for i in range(4)]
    else: 
        [tokenizer.decode(batch["input_ids"][8][i].tolist()) for i in range(3)]
    #show_one(updated["train"][8])

    

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
