from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import RandomRetriever, BM25Retriever, ConERetriever, TopkRetriever, PPLInferencer, AccEvaluator, DPPRetriever, MDLRetriever
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
import math
import os
import re
import json
from pprint import pprint
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from collections import Counter
from scipy.sparse import csr_matrix

from tqdm import tqdm
import pickle

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from openai import OpenAI
import openai
import faiss

from utils import templates, input_columns, output_columns, test_split, score_mat_2_rank_mat, omit_substrings

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"


#################### load dataset
print("loading dataset")

task = 'cms'
task_name = task
data_dir = 'data/'

train_path = data_dir + task_name + '/train.jsonl'
test_name = test_split[task_name]
test_path = data_dir + task_name + '/' + test_name + '.jsonl'

combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})

train_dataset = combined_dataset["train"]
test_dataset = combined_dataset["test"]

with open(data_dir + task_name + "/qid2tid_dic", 'rb') as f:
    qid2tid_dic = pickle.load(f)
    
with open(data_dir + task_name + "/topic_list", 'rb') as f:
    topic_list = pickle.load(f)

#################### query / topic embedding
print("computing query/topic embeddings")

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

model_id = 'sentence-transformers/all-mpnet-base-v2'  # #"sentence-transformers/multi-qa-mpnet-base-cos-v1" #
model = SentenceTransformer(model_id)
model = model.to("cuda")
model = model.eval()

from torch.utils.data import DataLoader
dataloader = DataLoader(train_dataset['text'], batch_size=1024)
emb_list = []
for _, entry in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        emb = model.encode(entry)
    emb_list.extend(emb)
query_emb = np.array(emb_list)

dataloader = DataLoader(topic_list, batch_size=1024)
emb_list = []
for _, entry in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        emb = model.encode(entry)
    emb_list.extend(emb)
topic_emb = np.array(emb_list)

with open(data_dir + task_name + "/query_emb", 'wb') as fw:
    pickle.dump(query_emb, fw, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(data_dir + task_name + "/topic_emb", 'wb') as fw:
    pickle.dump(topic_emb, fw, protocol=pickle.HIGHEST_PROTOCOL)

#################### covered topic prediction
print("computing required topics")
class Topic_predictor(nn.Module):
    def __init__(self, topic_emb):
        super(Topic_predictor, self).__init__()

        self.topic_emb = nn.Parameter(topic_emb, requires_grad=False)
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        
    def forward(self, batch_X):
        
        output = torch.mm(self.mlp(batch_X), self.topic_emb.T)
        return output

CLF = Topic_predictor(torch.FloatTensor(topic_emb)).to('cuda')
CLF.load_state_dict(torch.load(data_dir + task_name + "/topic_predictor", weights_only=True))

class CLF_dataset(data.Dataset):
    def __init__(self, X, Y):

        super(CLF_dataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx, self.X[idx]

    def get_labels(self, batch_indices):
        return self.Y[batch_indices]
    
num_topic = topic_emb.shape[0]
num_query = query_emb.shape[0]

train_X = torch.FloatTensor(query_emb)
train_Y = None

CLF_train_dataset = CLF_dataset(train_X, train_Y)
CLF_test_loader = data.DataLoader(CLF_train_dataset, batch_size=1024, shuffle=False)

with torch.no_grad():
    CLF_test = CLF.eval()
    c_clf_logit = []
    for _, mini_batch in enumerate(CLF_test_loader):
        batch_indices, batch_X = mini_batch
        batch_X = batch_X.to('cuda')
        output = CLF_test(batch_X)
        c_clf_logit.extend(output.cpu())

c_clf_logit = torch.stack(c_clf_logit)
with open(data_dir + task_name + "/query_clf_logit", 'wb') as fw:
    pickle.dump(c_clf_logit, fw, protocol=pickle.HIGHEST_PROTOCOL)