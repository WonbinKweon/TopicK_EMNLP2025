"""DPP Retriever"""

from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.logging import get_logger
from typing import Optional
import tqdm
import numpy as np
import math
from accelerate import Accelerator
import torch
from scipy import stats
import faiss
# import faiss.contrib.torch_utils
import time

logger = get_logger(__name__)

def filtering(score_mat, quantile=0.1):
    th_mat = torch.quantile(score_mat, quantile, dim=-1, keepdim=True)
    score_mat = torch.where(score_mat <= th_mat, torch.tensor(0.0, device=score_mat.device), score_mat)
    return score_mat

def score_mat_2_rank_mat(score_mat, weight=0.1):
    sorted_indices = torch.argsort(score_mat, dim=-1, descending=True)
    rank_mat = torch.zeros_like(score_mat, dtype=torch.long)
    rank_mat.scatter_(-1, sorted_indices, torch.arange(score_mat.size(-1), device=score_mat.device).expand_as(sorted_indices))
    rank_score_mat = (1.0 / (rank_mat + 1).float()) ** weight
    return rank_score_mat


class TopicKRetriever(TopkRetriever):
    model = None
    def __init__(self,
                 dataset_reader: DatasetReader,
                 CLF,
                 query_clf_logit,
                 topic_knowledge,
                 task_name,
                 norm_func = "rank",
                 pool_func = "emb",
                 score_func = "z",
                 weight_ens = 0.53,
                 weight_rel = 0.1,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 8,
                 candidate_num: Optional[int] = 100,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 seed: Optional[int] = 1,
                 scale_factor: Optional[float] = 0.1
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token,
                         sentence_transformers_model_name, ice_num, index_split, test_split, tokenizer_name, batch_size,
                         accelerator)
        self.candidate_num = candidate_num
        self.seed = seed
        self.scale_factor = scale_factor
        
        self.task_name = task_name
        self.CLF = CLF
        self.norm_func = norm_func
        self.pool_func = pool_func
        self.score_func = score_func
        self.weight_ens = weight_ens
        self.weight_rel = weight_rel

        topic_knowledge = score_mat_2_rank_mat(torch.FloatTensor(np.clip(topic_knowledge, 0.1, 0.9)), self.weight_ens)
        # topic_knowledge = torch.FloatTensor(np.clip(topic_knowledge, 0.1, 0.9))
        self.topic_knowledge = torch.tensor(topic_knowledge, dtype=torch.float32).cuda() # p(t): [T]
        self.query_clf_logit = query_clf_logit  ## exp p(t|d): [D, T]

        self.query_clf = torch.sigmoid(self.query_clf_logit.cuda()).cpu()  ## p(t|d): [D, T]
        self.query_score = score_mat_2_rank_mat(filtering(self.query_clf.cpu()) / (self.topic_knowledge.cpu()+1e-2), self.weight_ens) #.cuda() ## p(t|d)/p(t): [D, T]
              
    def topicK_search(self):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]

        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            embed = torch.tensor(entry['embed'], dtype=torch.float32).unsqueeze(0).cuda()
            with torch.no_grad():
                test_topic_prob = self.CLF(embed)
                test_topic_prob = score_mat_2_rank_mat(filtering(test_topic_prob), self.weight_ens)

            # top-K
            embed_np = embed.detach().cpu().numpy()
            near_ids = self.index.search(embed_np, self.candidate_num)[1][0].tolist()
            near_reps_np = np.array([self.index.index.reconstruct(int(i)) for i in near_ids], dtype=np.float32)
            near_reps = torch.from_numpy(near_reps_np).cuda()

            # Relevance score via dot product
            rel_scores = torch.matmul(embed, near_reps.T).squeeze(0)  # [K]

            ice_selected = []
            near_topic_prob = self.query_score[near_ids].cuda()  ## p(t|d)/p(t): [K, T]

            while len(ice_selected) < self.ice_num:
                if len(ice_selected) == 0:
                    next_topic_prob = near_topic_prob  # [K, T]
                else:
                    current_emb = near_reps[ice_selected].mean(dim=0, keepdim=True)  # [1, d]
                    next_emb = (near_reps + len(ice_selected) * current_emb) / (len(ice_selected) + 1)
                    with torch.no_grad():
                        next_topic_prob = torch.sigmoid(self.CLF(next_emb))
                        next_topic_prob = score_mat_2_rank_mat(filtering(next_topic_prob) / (self.topic_knowledge+1e-2), self.weight_ens)  # p(t|D*,d)/p(t): [K, T]
                topic_scores = torch.matmul(test_topic_prob, next_topic_prob.T).squeeze(0)  # [K]

                # z-score in torch
                rel_scores_z = (rel_scores - rel_scores.mean()) / (rel_scores.std(unbiased=False) + 1e-8)
                topic_scores_z = (topic_scores - topic_scores.mean()) / (topic_scores.std(unbiased=False) + 1e-8)
                total_scores = rel_scores_z * self.weight_rel + topic_scores_z  # [K]

                total_scores[ice_selected] = -float("inf")  # mask selected
                next_idx = torch.argmax(total_scores).item()
                ice_selected.append(next_idx)

            # Final index selection
            idx_selected = [near_ids[i] for i in ice_selected]

            if len(set(idx_selected)) != self.ice_num:
                print("duplicated ice", idx_selected)

            rtr_idx_list[idx] = idx_selected

        return rtr_idx_list

    def retrieve(self):
        return self.topicK_search()