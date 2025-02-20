import torch
import torch.nn.functional as F
from torch import nn

class Model_PJF(nn.Module):
    """
    Base class for all models.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, batch):
        """
        Forward pass logic
        """
        raise NotImplementedError
    
    def calculate_loss(self, batch):
        """
        Calculate loss
        """
        raise NotImplementedError

class Model_PJFNN(Model_PJF):
    def __init__(self,config):
        super(Model_PJFNN, self).__init__(config)
        self.vocab_size = self.config['vocab_size']
        self.geek_emb_layer = nn.Embedding(self.vocab_size, 256)
        self.job_emb_layer = nn.Embedding(self.vocab_size, 64)
        self.geek_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU()
        ) # (bs*max_geek_item, 256, max_seq_len) -> (bs*max_geek_item, 256, max_seq_len-5+1)
        self.geek_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=5, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU()
        ) # (bs*max_geek_item, 256, max_seq_len-5+1) -> (bs*max_geek_item, 64, max_seq_len-5+1-5+1)
        self.job_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=64, kernel_size=5,padding=0), 
            nn.BatchNorm1d(64),
            nn.ReLU()
        ) # (bs*max_job_item, 64, max_seq_len) -> (bs*max_geek_item, 64, max_seq_len-5+1)
        self.job_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ) # (bs*max_job_item, 64, max_seq_len-5+1) -> (bs*max_geek_item, 64, max_seq_len-5+1-3+1)

    def forward(self, batch):
        geek_input_ids = batch["geek_texts"] # (bs, max_geek_item, max_seq_len)
        job_input_ids = batch["job_texts"] # (bs, max_geek_item, max_seq_len)

        geek_emb, job_emb = self._ids2embs(job_input_ids, geek_input_ids) # (bs, max_len, emb_size)
        geek_vector = self._forward_geek(geek_emb) # (bs, 64)
        job_vector = self._forward_job(job_emb) # (bs, 64)
    
        sim = F.cosine_similarity(job_vector, geek_vector) # (bs)

        return {
            "sim": sim, 
            "job_vec": job_vector,
            "geek_vec": geek_vector
        }

    def calculate_loss(self, outputs, batch):
        sim = outputs["sim"]
        labels = batch["label"]

        # split pos, neg
        pos = labels == 1
        neg = labels == 0

        pos_loss = -torch.mean(sim[pos]).mean()
        neg_loss = torch.mean(sim[neg]).mean()
        l2_reg = torch.tensor(0., device = sim.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)

        lambda_reg = 0.01
        total_loss = pos_loss + neg_loss + lambda_reg * l2_reg

        return total_loss

    @torch.no_grad()
    def _ids2embs(self, job_input_ids, geek_input_ids):
        geek_emb = self.geek_emb_layer(geek_input_ids) # (bs, max_geek_item, max_seq_len, 256) <- (bs, max_geek_item, max_seq_len) 
        job_emb = self.job_emb_layer(job_input_ids) # (bs, max_geek_item, max_seq_len, 64) <- (bs, max_geek_item, max_seq_len)
        print("geek_emb.size():", geek_emb.size(),"\n", "job_emb.size():", job_emb.size())
        return geek_emb, job_emb
    
    def _forward_geek(self, geek_emb):
        # geek_emb : (bs, max_geek_item, max_seq_len, 256)
        bs, max_geek_item, max_seq_len, emb_size = geek_emb.size()
        x = geek_emb.view(-1, max_seq_len, emb_size)
        x = x.permute(0, 2, 1)

        x = self.geek_conv1(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.geek_conv2(x)
        x = F.max_pool1d(x, kernel_size=x.size(2)) # (bs*max_geek_item, 64, 1)
        x = x.squeeze(-1)
        x = x.view(bs, max_geek_item, -1) # (bs, max_geek_item, 64)
        # mean at item dim
        x = x.mean(dim=1)
        return x
    
    def _forward_job(self, job_emb): 
        # job_emb : (bs, max_geek_item, max_seq_len, 64) 
        bs, max_geek_item, max_seq_len, emb_size = job_emb.size()
        x = job_emb.view(-1, max_seq_len, emb_size) # (bs*max_geek_item, max_seq_len, emb_size)
        x = x.permute(0, 2, 1) # (bs*max_geek_item, emb_size, max_seq_len) 

        x = self.job_conv1(x) # (bs*max_geek_item, 64, max_seq_len-5+1) = (bs*max_geek_item, 64, max_seq_len-4)
        x = F.max_pool1d(x, kernel_size=2) # (bs*max_geek_item, 64, max_seq_len-4-2+1) = (bs*max_geek_item, 64, max_seq_len-5)
        x = self.job_conv2(x) # (bs*max_geek_item, 64, max_seq_len-5-3+1) = (bs*max_geek_item, 64, max_seq_len-7)
        x = F.max_pool1d(x, kernel_size=x.size(2)) # (bs*max_geek_item, 64, 1)
        x = x.squeeze(-1) # (bs*max_geek_item, 64)
        x = x.view(bs, max_geek_item, -1) # (bs, max_geek_item, 64)
        # max at item dim
        x = x.max(dim=1)[0] # (bs, 64)
        return x
