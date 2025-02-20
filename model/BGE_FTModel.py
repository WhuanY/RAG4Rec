import torch
import torch.nn as nn
from transformers import AutoModel
from collections import defaultdict
import logging
class BGE_FTModel(torch.nn.Module):
    def __init__(self, config):
        super(BGE_FTModel, self).__init__()
        self.text_matcher = AutoModel.from_pretrained(config['bge_path']).to(f"cuda:{config['gpu_id']}")
        self.predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(f"cuda:{config['gpu_id']}")

        self.loss_fn = nn.BCELoss()
        # xavier initialization for predictor
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        self.frozen_target_parameters()

    def forward(self, sample):
        """sample: dict like {
            "user_id": list,
            "job_id": list,
            "model_input": {"input_ids": tensor, "attention_mask": tensor, "token_type_ids": tensor},
            "label": tensor
        }
        """
        text_input = {k: v.squeeze(1) for k, v in sample["model_input"].items()}
        text_output = self.text_matcher(**text_input)[0][:, 0] 
        output = self.predictor(text_output)
        return output

    def calculate_loss(self, output, label):
        #TODO: Apply more innovative loss functions.
        return self.loss_fn(output, label)

    def frozen_target_parameters(self):
        for param in self.text_matcher.parameters():
            param.requires_grad = False
        self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Trainable Params: {trainable_params}. Total Params: {total_params}. Trainable Paramaters Ratio: {trainable_params/total_params}")
