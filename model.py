import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertCls(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.cls = nn.Linear(self.config.bert_hidden_size, self.config.num_class)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        emb = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = emb[1]
        out = self.dropout(out)
        out = self.cls(out)
        out = torch.sigmoid(out)
        return out
