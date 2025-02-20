# -----Data Preparation-----
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

bge_path = "/media/wuyuhuan/bge-small-zh"
logging.info(f"Using BGE model from {bge_path}")
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
logging.info(f"Torch Imported. Using PyTorch version {torch.__version__}")
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from peft import LoraModel, LoraConfig

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.info(f"Usable GPU: {torch.cuda.device_count()}, device using: {device}")  
tokenizer = AutoTokenizer.from_pretrained(bge_path)

def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    logging.info(f"Setting all seeds to {seed} to ensure reproducibility...")
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dict2device(data, device):
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = dict2device(v, device)
        elif isinstance(v, torch.Tensor):
            data[k] = v.to(device)
    return data

same_seed(42)

class BGE_FTDataset(Dataset):
    def __init__(self, mode: str, in_file: str, tokenizer: AutoTokenizer, ratio: float = 1.0):
        """
        mode: str, one of ['train', 'valid', 'test']
        in_file: str, path to the input csv file.
        tokenizer: AutoTokenizer, tokenizer for the model.
        ratio: float, the ratio of the data to be used. Default: 1.0. 
            set to 0.01 for functionality testing.
        """
        self.mode = mode
        self.dataset = pd.read_csv(in_file).sample(frac=ratio)
        self.tokenizer = AutoTokenizer.from_pretrained(bge_path)
        logging.info(f"Dataset {mode} Loaded. Shape: {self.dataset.shape}")

    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return {
            "user_id": row['user_id:token'],
            "job_id": row['job_id:token'],
            "text_pair": (row["cv"], row["jd"]),
            "label": row["browsed:label"]}

train_dataset = BGE_FTDataset('train', 'dataset/processed_train.csv', tokenizer,ratio=1)
valid_dataset = BGE_FTDataset('valid', "dataset/processed_valid.csv", tokenizer,ratio=1)
test_dataset = BGE_FTDataset('test', "dataset/processed_test.csv", tokenizer, ratio=1)

# train_dataset[0]
# valid_dataset[0]
# test_dataset[0]

def collate_fn(batch, tokenizer):
    user_ids = [item['user_id'] for item in batch]
    job_ids = [item['job_id'] for item in batch]
    cv_texts, jd_texts = zip(*[item['text_pair'] for item in batch])
    labels = [item['label'] for item in batch]

    tokenized_cv = tokenizer(text=cv_texts, 
                             text_pair=jd_texts,
                             padding='max_length',
                             truncation=True,
                             return_tensors='pt')

    return {
        'user_id':user_ids,
        'job_id':job_ids,
        'model_input': {
            'input_ids': tokenized_cv['input_ids'],
            'attention_mask': tokenized_cv['attention_mask'],
            'token_type_ids': tokenized_cv['token_type_ids']
            }, 
        'label': torch.tensor(labels, dtype=torch.float32)
    }

# Create dataloaders with parallel processing
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    num_workers=2,
    shuffle=True,
    collate_fn=lambda x: collate_fn(x, tokenizer),
    pin_memory=True,
    persistent_workers=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=256,
    num_workers=2,
    shuffle=False,
    collate_fn=lambda x: collate_fn(x, tokenizer),
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    num_workers=2,
    shuffle=False,
    collate_fn=lambda x: collate_fn(x, tokenizer),
    pin_memory=True,
    persistent_workers=True
)

# -----Evaluator-----
from typing import List
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
from typing import List
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
class Evaluator:
    """
    Evaluator for the BGE-FT model.
    """
    def __init__(self):
        #TODO: the label values are currently only assumed to be binary
        # For further experiments, we need to make the label values more general.  
        self.uid2topk = {} # {uid: [(score, label), ...]}  
        
        self.topk = 10
        self.metric2func = {
            "ndcg": self._ndcg,
            "precision": self._precision,
            "recall": self._recall,
            "map": self._map,
            "mrr": self._mrr,
            "auc": self._auc,
            "logloss": self._logloss,
        }
        self.cls_metrics = ["auc", "logloss"]
        self.rkg_metrics = ["ndcg", "precision", "recall", "map", "mrr"]
    
    def collect(self, uid, score, label):
        """
        Process a batch of data. Save the data to the evaluator. 
        Input params are lists of same length as batch size.
        After this func, uid2topk will look like: {uid: [(score, label), ...]}
        where each uid has interaction list sorted by score

        Args:
            uid: list, list of user ids.  
            score: list, list of scores.
            label: list, list of labels.
            
        Returns:
            None
        """
        for u, s, l in zip(uid, score, label):
            if u not in self.uid2topk:
                self.uid2topk[u] = []
            self.uid2topk[u].append((s, l)) 

        for u in self.uid2topk:
            self.uid2topk[u] = sorted(self.uid2topk[u], key=lambda x: x[0], reverse=True)
         
    def evaluate(self, K: List[int]):
        """
        Evaluate the model using the collected data and the pass value k.
        Args:
            K: List[int], a list of k values for ranking metrics.
        
        return:
            result: dict, a dictionary of evaluation results.
            result_str: str, a formatted string of the evaluation results.
        """
        result = {} # {cls_m1: value1, cls_m2: value2, ..., rkg_m1@k1: value1, rkg_m2@k2: value2, ...}

        # Calculate the metrics
        for cls_metric in self.cls_metrics:
            matric_val = self.metric2func[cls_metric]()
            result[cls_metric] = matric_val

        for rkg_metric in self.rkg_metrics:
            for k in K:
                result[rkg_metric + '@' + str(k)] = self.metric2func[rkg_metric](k)
        
        result_str = self._format_str(result)
        return result, result_str
    
    # below are the ranking metric functions. With most of are indirect copy from the recbole.metrics.
    def _ndcg(self, k):
        base = []
        idcg = []

        # save base and idcg(Ideal DCG) for each position
        for i in range(k):
            base.append(np.log(2) / np.log(i + 2)) # np.log(2) / np.log(i + 2) = log_{i + 2}(2)
            if i > 0:
                idcg.append(base[i] + idcg[i - 1])
            else:
                idcg.append(base[i])

        # calculate the dcg
        tot = 0
        for uid in self.uid2topk:
            dcg = 0
            pos = 0
            for i, (score, label) in enumerate(self.uid2topk[uid][:k]):
                dcg += (2 ** label - 1) * base[i] # 2^rel - 1 / log_(2)(i + 1)
                pos += label # TODO: If label is not binary, this should be modified.
            tot += dcg / idcg[int(pos) - 1]
        return tot / len(self.uid2topk)

    def _precision(self, k):
        tot = 0
        valid_length = 0
        for uid in self.uid2topk:
            rec = 0
            rel = 0
            for i, (score, label) in enumerate(self.uid2topk[uid][:k]):
                rec += 1
                rel += label # TODO: If label is not binary, this should (maybe) be modified.
            try:
                tot += rel / rec
                valid_length += 1
            except:
                continue
        return tot / valid_length
    
    def _recall(self, k):
        tot = 0
        valid_length = 0
        for uid in self.uid2topk:
            rec = 0
            rel = 0
            for i, (score, label) in enumerate(self.uid2topk[uid]):
                if i < k:
                    rec += label
                rel += label #TODO: If label is not binary, this should (maybe) be modified.
            try:
                tot += rec / rel
                valid_length += 1
            except:
                continue
        return tot / valid_length

    # TODO: The MAP and MRR functions are not understood yet.
    def _map(self,k):
        tot = 0
        for uid in self.uid2topk:
            tp = 0
            pos = 0
            ap = 0
            for i, (score, label) in enumerate(self.uid2topk[uid][:k]):
                if label == 1:
                    tp += 1
                    pos += 1
                    ap += tp / (i + 1)
            if pos > 0:
                tot += ap / pos
        return tot / len(self.uid2topk)

    def _mrr(self, k):
        tot = 0
        for uid in self.uid2topk:
            for i, (score, label) in enumerate(self.uid2topk[uid]):
                if label == 1:
                    tot += 1 / (i + 1)
                    break
        return tot / len(self.uid2topk)
           
    # below are the classification metric functions
    def _auc(self):
        """
        Calculate the AUC score.
        """
        total_auc = 0
        valid_auc_num = 0
        for uid, topk in self.uid2topk.items():
            score, labels = zip(*topk)
            try:
                auc = roc_auc_score(labels, score)
                total_auc += auc
                valid_auc_num += 1
            except:
                continue
        return total_auc / valid_auc_num
        
    def _logloss(self):
        """
        Calculate the logloss.
        """
        total_logloss = 0
        valid_logloss_num = 0
        for uid, topk in self.uid2topk.items():
            score, labels = zip(*topk)
            try:
                logloss = log_loss(labels, score)
                valid_logloss_num += 1
                total_logloss += logloss
            except:
                continue
        return total_logloss / valid_logloss_num

    # other utility functions for evaluator
    def _format_str(self, result):
        res = ''
        for metric in result.keys():
            res += '\n\t{}:\t{:.4f}'.format(metric, result[metric])
        return res
    
#-----Model Architecture-----
from collections import defaultdict

class BGE_FTModel(torch.nn.Module):
    def __init__(self, rag_model):
        super(BGE_FTModel, self).__init__()
        logging.info(f"Initializing Model Based on path: {rag_model}")
        self.text_matcher = AutoModel.from_pretrained(rag_model).to(device)
        self.predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        ).to(device)

        self.loss_fn = nn.BCEWithLogitsLoss()
        # xavier initialization for predictor
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        config = LoraConfig(r=4, lora_alpha=4, target_modules=["query", "key"])
        self.text_matcher = LoraModel(self.text_matcher, config, adapter_name="default").to(device)
        
        # logging.info(f"Frozing Parameters...")
        # self.frozen_target_parameters()
        # logging.info(f"Model Initialized.")
        self.print_trainable_parameters()

        self.timing_stats = defaultdict(list)  # For storing timing information

    def forward(self, sample):
        """sample: dict like {
            "user_id": list,
            "job_id": list,
            "model_input": {"input_ids": tensor, "attention_mask": tensor, "token_type_ids": tensor},
            "label": tensor
        }
        """
        # Create CUDA events for timing
        start = torch.cuda.Event(enable_timing=True)
        step1 = torch.cuda.Event(enable_timing=True)
        step2 = torch.cuda.Event(enable_timing=True)
        step3 = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        text_input = {k: v.squeeze(1) for k, v in sample["model_input"].items()}
        step1.record()
        text_output = self.text_matcher(**text_input)[0][:, 0] 
        step2.record()
        output = self.predictor(text_output)
        step3.record()
        end.record()

        # Synchronize CUDA operations
        torch.cuda.synchronize()

        # Record timing statistics
        self.timing_stats['data_prep'].append(start.elapsed_time(step1))
        self.timing_stats['text_matcher'].append(step1.elapsed_time(step2))
        self.timing_stats['predictor'].append(step2.elapsed_time(step3))

        return output

    
    def calculate_loss(self, output, label):
        #TODO: Apply more innovative loss functions.
        return self.loss_fn(output, label)

    def frozen_target_parameters(self):
        logging.info(f"Frozing Parameters...")
        for param in self.text_matcher.parameters():
            param.requires_grad = False
        self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Trainable Params: {trainable_params}. Total Params: {total_params}. Trainable Paramaters Ratio: {trainable_params/total_params}")

    def get_average_timings(self):
        """Get average timing statistics"""
        return {
            step: sum(times)/len(times) if times else 0 
            for step, times in self.timing_stats.items()
        }

    def reset_timings(self):
        """Reset timing statistics"""
        self.timing_stats.clear()

#-----Trainer-----
class Trainer(object):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, optimizer, eval_step, verbose=True):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.clip_grad_norm = None
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler() # for mixed precision training 
        self.eval_step = 1

        self.verbose = verbose

    def train(self, epochs, early_stopping_epochs=5):
        train_loss = valid_loss = float('inf')
        
        # the below init values are for early stopping
        best_valid = cur_best_valid = float('inf')
        cur_step_from_best_val = 0
        
        for epoch_idx in range(epochs):
            # train
            train_loss = self._train_epoch(epoch_idx, self.train_dataloader) # mean loss of this epoch

            # valid
            valid_loss = self._valid_epoch(epoch_idx, self.valid_dataloader) # mean loss of this epoch
            if self.verbose:
                logging.info(f"Epoch {epoch_idx} Train mean Loss: {train_loss:.4f}, Valid mean Loss: {valid_loss:.4f}")
                
            # early stopping
            if (epoch_idx + 1) % self.eval_step == 0:
                if self.verbose:
                    logging.info(f"Epoch {epoch_idx + 1} starts early stopping check.")
                cur_best_valid, cur_step_from_best_val, stop_flag, update_flag = self._early_stopping(
                    valid_loss, cur_best_valid, cur_step_from_best_val, early_stopping_epochs, lower_is_better=True) # -> best, cur_step, stop_flag, update_flag
                
                if update_flag:
                    best_valid = cur_best_valid
            
                if stop_flag:
                    if self.verbose:
                        logging.info(f"Early stopping at epoch {epoch_idx}")
                    break
        
        return best_valid

    @torch.no_grad()
    def eval(self, evaluator):
        """
        Using the test dataloader to evaluate the model.
        For each (cv, jd) pair, we predict the probability of the cv being browsed.
        The evaluation results are saved to the save_path as the following format:

        The evaluation matriceare all based on top-k selection. for each cv_i, the 
        top-k are selected from all (cv_i, jd) pairs that appear in the test set. 
        After consideration, due to the context of precise-recommendation matching 
        task, we decide if jd_j are in the testset records but not being recorded 
        with cv_i in the testset, we will not consider jd_j in the top-k selection
        for cv_i.
        
        params:
            evaluator: Evaluator, the evaluator for the model.

        return:
            result: list
        """
        # set model to eval mode
        if self.verbose:
            logging.info("Start evaluating on test set")
        self.model.eval()


        pbar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), desc="Matrices Evaluation     ")
       
        # predicting scores, while saving the predictions records.
        for step, sample in pbar:
            uid = sample["user_id"] # List of length bs
            sample = dict2device(sample, device) # {"model_input_jd": dict, "model_input_cv": dict, "label": tensor}
            scores = self.model(sample).squeeze(-1).cpu().tolist()
            labels = sample["label"].squeeze(-1).cpu().tolist()
            evaluator.collect(uid, scores, labels)

        # evaluate the results
        results, results_str = evaluator.evaluate([1, 5, 10])
        return results, results_str

    # below is indirect copy from https://github.com/hyp1231/SHPJF/tree/master/model
    def _train_epoch(self, epoch_idx: int, train_dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        total_batches = len(train_dataloader)
        
        pbar = tqdm(enumerate(train_dataloader), total=total_batches, desc=f"Epoch {epoch_idx} Train")

        for step, sample in enumerate(train_dataloader):
            # Create new events for each iteration
            start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            loss_end = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)

            start.record()
            sample = dict2device(sample, device)
            label = sample["label"].unsqueeze(1).to(device)
            self.optimizer.zero_grad()
            
            # 模型前向传播
            with autocast(device_type=str(device),dtype=torch.float16):
                output = self.model(sample)
                forward_end.record()
                # 损失计算
                loss = self.model.calculate_loss(output, label)
                loss_end.record()
            # Use scaler for backward pass
            self.scaler.scale(loss).backward() 
            backward_end.record()
            
            # 梯度裁剪（如果启用）
            if self.clip_grad_norm:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            
            # 优化器步骤
            self.scaler.step(self.optimizer)
            self.scaler.update()
            torch.cuda.synchronize()

            # 更新进度条显示
            forward_elapsed_time = start.elapsed_time(forward_end)
            loss_elapsed_time = forward_end.elapsed_time(loss_end)
            backward_elapsed_time = loss_end.elapsed_time(backward_end)
            if (step + 1) % 100 == 0:
                logging.info(f"Epoch {epoch_idx} Step {step + 1}/{total_batches} | loss: {loss.item():.4f} | forward: {forward_elapsed_time:.2f} ms | loss: {loss_elapsed_time:.2f} ms | backward: {backward_elapsed_time:.2f} ms")
            # pbar.set_postfix_str(f"loss: {loss.item():.4f} | forward: {forward_elapsed_time:.2f} ms | loss: {loss_elapsed_time:.2f} ms | backward: {backward_elapsed_time:.2f} ms")
            
            total_loss += loss.item()
            self._check_nan(loss)
            
        return total_loss / total_batches
    
    @torch.no_grad()
    def _valid_epoch(self, epoch_idx: int, valid_dataloader: DataLoader):
        """valid the model with valid data by calculate the loss
        """
        
        # set model to eval mode
        self.model.eval()
        total_loss = 0
        total_batches = len(valid_dataloader)
        pbar = tqdm(enumerate(valid_dataloader), total=total_batches, desc=f"Epoch {epoch_idx} Valid")

        # calculate loss on validation set
        for step, sample in enumerate(valid_dataloader):
            sample = dict2device(sample, device) # batch: {"model_input_jd": dict, "model_input_cv": dict, "label": tensor}
            label = sample["label"].unsqueeze(1) # (bs, 1)
            output = self.model(sample) # (bs, 1)
            loss = self.model.calculate_loss(output, label) # output: (bs, 1), label: (bs, 1)
            if (step + 1) % 10 == 0:
                logging.info(f"Epoch {epoch_idx} Valid Step {step + 1}/{total_batches} | loss: {loss.item():.4f}")
            # pbar.set_postfix(loss=loss.item())
            total_loss += loss.item()
            self._check_nan(loss)

        return total_loss / total_batches
    
    def _early_stopping(self, value, best, cur_step, max_step, lower_is_better=True):
        """validation-based early stopping

        Args:
            value (float): current result
            best (float): best result
            cur_step (int): the number of consecutive steps that did not exceed the best result
            max_step (int): threshold steps for stopping

        Returns:
            tuple:
            - best: float,
            best result after this step
            - cur_step: int,
            the number of consecutive steps that did not exceed the best result after this step
            - stop_flag: bool,
            whether to stop
            - update_flag: bool,
            whether to update
        """
        stop_flag = False
        update_flag = False

        better = value < best if lower_is_better else value > best
        if better:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
        return best, cur_step, stop_flag, update_flag

    def _check_nan(self, loss):
        if torch.isnan(loss).any():
            raise ValueError("Model diverged with loss = NaN")
        return
    
#-----Main-----
model = BGE_FTModel(bge_path).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()
trainer = Trainer(model, train_loader, valid_loader, test_loader, optimizer, eval_step=1)
best_valid = trainer.train(epochs = 1000)
evaluator = Evaluator()
result, result_str = trainer.eval(evaluator)

print(result)