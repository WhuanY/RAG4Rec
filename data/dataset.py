import os
from tqdm import tqdm
from logging import getLogger
import pandas as pd
import torch
from torch.utils.data import Dataset

from typing import Tuple, Dict

class Dataset_PJF(Dataset):
    def __init__(self, config, pool, phase):
        assert phase in ['train', 'test', 'valid']
        super(Dataset_PJF, self).__init__()
        self.config = config
        self.pool = pool
        self.phase = phase
        self.logger = getLogger()

        self._init_attributes(pool)
        self._load_inters()
        

    def _init_attributes(self, pool):
        self.geek_num = pool.geek_num
        self.job_num = pool.job_num
        self.geek_token2id = pool.geek_token2id
        self.job_token2id = pool.job_token2id

    def _load_inters(self):
        filepath = os.path.join(self.config['dataset_path'], f'dataset-{self.phase}.inter')  # dataset-{mode}.inter
        self.logger.info(f'Loading {self.phase} dataset from {filepath}')

        self.geek_ids, self.job_ids, self.labels = [], [], []
        with open(filepath, 'r', encoding='utf-8') as file:
            next(file)  # skip the first line
            for line in tqdm(file):
                geek_token, job_token, b, d, s=  line.strip().split('\t')
                geek_id = self.geek_token2id[geek_token]
                self.geek_ids.append(geek_id)
                job_id = self.job_token2id[job_token]
                self.job_ids.append(job_id)
                self.labels.append([float(b), float(d), float(s)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            'geek_id': self.geek_ids[index],
            'job_id': self.job_ids[index],
            'label': self.labels[index]
        }
    
class Dataset_PJFNN(Dataset_PJF):
    def __init__(self, config, pool, phase):
        super(Dataset_PJFNN, self).__init__(config, pool, phase)

    def __getitem__(self, index):
        geek_id = self.geek_ids[index]
        job_id = self.job_ids[index]
        label = self.labels[index][0] # only use the first label
        return {
            'geek_texts': self.pool.get_texts("geek", geek_id), 
            'job_texts': self.pool.get_texts("job", job_id),
            'label': label
        }

    def __len__(self):
        return super().__len__()

class Dataset_BGEFT(Dataset_PJF):
    def __init__(self, mode:str, config: Dict):
        """
        mode: str, the mode of the dataset, one of "train", "valid" or "test".
        in_file: str, path to the input csv file.
        tokenizer: AutoTokenizer, tokenizer for the model.
        ratio: float, the ratio of the data to be used. Default: 1.0. 
            set to for functionality testing.
        """
        if mode == "train":
            self.dataset = pd.read_csv(config['train_ds_path']).sample(frac=config['ratio'])
        elif mode == "valid":
            self.dataset = pd.read_csv(config['valid_ds_path']).sample(frac=config['ratio'])
        elif mode == "test":
            self.dataset = pd.read_csv(config['test_ds_path']).sample(frac=config['ratio'])

    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return {
            "user_id": row['user_id:token'],
            "job_id": row['job_id:token'],
            "text_pair": (row["cv"], row["jd"]),
            "label": row["browsed:label"]}

if __name__ == "__main__":
    test_config = {'dataset_path': '/root/RAG4Rec/dataset'}
    from pool import PJFNNPool
    test_pool = PJFNNPool(test_config)
    train_dataset = Dataset_PJFNN(test_config, test_pool, 'train')
    print(train_dataset)
    print(train_dataset[0])