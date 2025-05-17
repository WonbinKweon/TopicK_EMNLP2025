from openicl import PromptTemplate
import numpy as np
import torch

subj_tp_dict = {
    0: "</E>Input: </text> Type: objective",
    1: "</E>Input: </text> Type: subjective"
}
subj_template = PromptTemplate(subj_tp_dict, {'text': '</text>'}, ice_token='</E>')

sst2_tp_dict = {
    0: "</E>Review: </text> Sentiment: negative",
    1: "</E>Review: </text> Sentiment: positive"
}
sst2_template = PromptTemplate(sst2_tp_dict, {'text': '</text>'}, ice_token='</E>')

sst5_tp_dict = {
    0: "</E>Review: </text> Sentiment: terrible",
    1: "</E>Review: </text> Sentiment: bad",
    2: "</E>Review: </text> Sentiment: okay",
    3: "</E>Review: </text> Sentiment: good",
    4: "</E>Review: </text> Sentiment: great",
}
sst5_template = PromptTemplate(sst5_tp_dict, {'text': '</text>'}, ice_token='</E>')

cr_tp_dict = {
    0: "</E>Review: </text> Sentiment: negative",
    1: "</E>Review: </text> Sentiment: positive"
}
cr_template = PromptTemplate(cr_tp_dict, {'text': '</text>'}, ice_token='</E>')

ag_news_tp_dict = {
    0: "</E>Input: </text> Type: world",
    1: "</E>Input: </text> Type: sports",
    2: "</E>Input: </text> Type: business",
    3: "</E>Input: </text> Type: technology",
}
ag_news_template = PromptTemplate(ag_news_tp_dict, {'text': '</text>'}, ice_token='</E>')

medmcqa_tp_dict = {
    0: "</E>Question: </text> Answer: (a)",
    1: "</E>Question: </text> Answer: (b)",
    2: "</E>Question: </text> Answer: (c)",
    3: "</E>Question: </text> Answer: (d)",
}
medmcqa_template = PromptTemplate(medmcqa_tp_dict, {'text': '</text>'}, ice_token='</E>')

law_tp_dict = {
    0: "</E>Question: </text> Answer: (a)",
    1: "</E>Question: </text> Answer: (b)",
    2: "</E>Question: </text> Answer: (c)",
    3: "</E>Question: </text> Answer: (d)",
}
law_template = PromptTemplate(law_tp_dict, {'text': '</text>'}, ice_token='</E>')

law_tp_dict = {
    0: "</E>Question: </text> Answer: (a)",
    1: "</E>Question: </text> Answer: (b)",
    2: "</E>Question: </text> Answer: (c)",
    3: "</E>Question: </text> Answer: (d)",
}
law_template = PromptTemplate(law_tp_dict, {'text': '</text>'}, ice_token='</E>')

MMLU_tp_dict = {
    0: "</E>Question: </text> Answer: (A)",
    1: "</E>Question: </text> Answer: (B)",
    2: "</E>Question: </text> Answer: (C)",
    3: "</E>Question: </text> Answer: (D)",
}
MMLU_template = PromptTemplate(MMLU_tp_dict, {'text': '</text>'}, ice_token='</E>')

mnli_tp_dict = {
    0: "</E></text1> Can we know </text>? Yes.",
    1: "</E></text1> Can we know </text>? Maybe.",
    2: "</E></text1> Can we know </text>? No."
    }
mnli_template = PromptTemplate(mnli_tp_dict, {'text1': '</text1>', 'text2': '</text>'}, ice_token='</E>')

# qnli_tp_dict = {
#     0: "</E></text1> Can we know </text>? Yes.",
#     1: "</E></text1> Can we know </text>? No."
#     }
# qnli_template = PromptTemplate(qnli_tp_dict, {'text1': '</text1>', 'text2': '</text>'}, ice_token='</E>')
qnli_tp_dict = {
    0: "</E></text1> Can we know </text>? Yes.",
    1: "</E></text1> Can we know </text>? No."
    }
qnli_template = PromptTemplate(qnli_tp_dict, {'text1': '</text>', 'text2': '</text1>'}, ice_token='</E>')

cms_tp_dict = {
    0: "</E>Question: </text> Answer: (A)",
    1: "</E>Question: </text> Answer: (B)",
    2: "</E>Question: </text> Answer: (C)",
    3: "</E>Question: </text> Answer: (D)",
    4: "</E>Question: </text> Answer: (E)"
}
cms_template = PromptTemplate(cms_tp_dict, {'text': '</text>'}, ice_token='</E>')

templates = {'sst2': sst2_template,
        'subj': subj_template,
        "sst5": sst5_template,
        'cr': cr_template,
        "ag_news": ag_news_template,
        "mnli": mnli_template,
        "qnli": qnli_template,
        "medmcqa": medmcqa_template,
        'law': law_template,
        'MMLU_STEM': MMLU_template,
        'MMLU_Humanities': MMLU_template,
        'MMLU_Social_Sciences': MMLU_template,
        'MMLU_Other': MMLU_template,
        'qasc': MMLU_template,
        "sciq" : medmcqa_template,
        "cms": cms_template
        }

input_columns={'sst2': ["text"],
        'subj': ['text'],
        "sst5": ["text"],
        "cr": ["text"],
        "ag_news": ["text"],
        "law": ["text"],
        'mnli': ['text1', 'text2'],
        "qnli": ["text1", "text2"],
        "sciq": ["text"],
        "medmcqa": ["text"],
        "MMLU_STEM": ["text"],
        'MMLU_Humanities': ["text"],
        'MMLU_Social_Sciences': ["text"],
        'MMLU_Other': ["text"],
        "qasc": ["text"],
        "cms": ["text"]
        }

output_columns={'sst2': 'label',
            'subj': 'label',
            "sst5": 'label',
            'cr': 'label',
            "ag_news": 'label',
            'mnli': 'label',
            "qnli": 'label',
            "sciq": "label",
            "law": "label",
            "medmcqa": "label",
            "MMLU_STEM": "label",
            'MMLU_Humanities': "label",
            'MMLU_Social_Sciences': "label",
            'MMLU_Other': "label",
            "qasc": "label",
            "cms": "label"
        }

test_split={
        'sst2': 'test',
        "subj": 'test',
        "sst5": 'test',
        "law": 'test',
        "cr": 'test',
        "ag_news": 'test',
        'mnli': 'validation', # cannot get gold labels for the test split
        "qnli": 'validation',
        "sciq": "test",
        "medmcqa": "validation",
        "MMLU_STEM": "test",
        'MMLU_Humanities': "test",
        'MMLU_Social_Sciences': "test",
        'MMLU_Other': "test",
        "qasc": "validation",
        "cms": "validation"
}

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

def omit_substrings(terms):
    sorted_terms = sorted(terms, key=len)
    result = []

    for i, term in enumerate(sorted_terms):
        is_prefix = False
        for other_term in sorted_terms[i+1:]:
            if term in other_term:
                is_prefix = True
                break
        if not is_prefix:
            result.append(term)

    return result