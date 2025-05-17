from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import RandomRetriever, BM25Retriever, ConERetriever, TopkRetriever, PPLInferencer, AccEvaluator, DPPRetriever, MDLRetriever, TopicKRetriever
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
import os
import re
import json
import pprint
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from tqdm import tqdm
import pickle
from scipy.sparse import csr_matrix

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

import faiss
from utils import templates, input_columns, output_columns, test_split, score_mat_2_rank_mat

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

def main(output_file, CLF, query_clf_logit, topic_knowledge, task_name, template, train_path, test_path, model_path, sentence_model_path, input_columns_name, output_columns_name, ice_num, batch_size, batch_size_inf, seed, output_json_filepath):
    # load dataset
    combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})

    train_dataset = combined_dataset["train"]
    test_dataset = combined_dataset["test"]

    # Print some information about the datasets
    print(train_dataset)
    print(test_dataset)

    accelerator = Accelerator()

    # Construct the DatasetReader
    data = DatasetReader(combined_dataset, input_columns=input_columns_name, output_column=output_columns_name)

    topick_retriever = TopicKRetriever(data, CLF, query_clf_logit, topic_knowledge, task_name, ice_num=ice_num, sentence_transformers_model_name=sentence_model_path, tokenizer_name=sentence_model_path, seed=seed, batch_size=batch_size, test_split='test')
    print("Start inference....")
    inferencer = PPLInferencer(model_name=model_path, tokenizer=model_path, output_json_filepath=output_json_filepath, batch_size=batch_size_inf, accelerator=accelerator)
    topick_predictions = inferencer.inference(topick_retriever, ice_template=template, output_json_filename=output_file) #
    
    torch.cuda.empty_cache()
    
    return topick_predictions

# set the model and dataset path
model_dir = ''
sentence_transformer_path = 'sentence-transformers/all-mpnet-base-v2'
data_dir = 'data/'

for model_name in ['meta-llama/Llama-3.2-3B-Instruct']: # 'Qwen/Qwen2.5-0.5B-Instruct' 'meta-llama/Llama-3.2-1B-Instruct'
    model_path = model_dir + model_name
    sentence_model_path = sentence_transformer_path

    for seed in [1]:
        for task_name in ['cms']:
            print("@@@@@@@@@@@@@@@@@@@@@  ", model_name, task_name, "  @@@@@@@@@@@@@@@@@@@@@")
            train_path = data_dir + task_name + '/train.jsonl'
            test_name = test_split[task_name]
            test_path = data_dir + task_name + '/' + test_name + '.jsonl'

            output_json_filepath = 'result/' + model_name + '/' + task_name

            import os
            os.makedirs(output_json_filepath, exist_ok=True)

            batch_size = 1024
            batch_size_inf = 4
            select_time = 10
            
            with open(data_dir + task_name + "/topic_emb", 'rb') as f:
                topic_emb = pickle.load(f)
            
            with open(data_dir + task_name + "/query_clf_logit", 'rb') as fw: 
                query_clf_logit = pickle.load(fw)

            class Topic_predictor(nn.Module):
                def __init__(self, topic_emb):
                    super(Topic_predictor, self).__init__()
                    self.topic_emb = nn.Parameter(topic_emb, requires_grad=False)
                    self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
                    
                def forward(self, batch_X):
                    output = torch.mm(self.mlp(batch_X), self.topic_emb.T)
                    return output
        
            ## Load concept extractor
            CLF = Topic_predictor(torch.FloatTensor(topic_emb)).to('cuda')
            CLF.load_state_dict(torch.load(data_dir + task_name + "/topic_predictor", weights_only=True)) 

            ## grid search
            for shot in [8]: ######################################################################################################################################################
                ice_num = shot

                with open(data_dir + task_name + "/topic_knowledge_" + model_name.split("/")[-1], 'rb') as fw: ##############
                    topic_knowledge = pickle.load(fw)


                topk_model = main(f'TopicK_seed{seed}_{ice_num}_shot',
                                CLF, query_clf_logit, topic_knowledge, task_name,
                                templates[task_name], train_path, test_path, model_path, sentence_model_path, input_columns[task_name], output_columns[task_name],
                                ice_num, batch_size, batch_size_inf, seed, output_json_filepath)