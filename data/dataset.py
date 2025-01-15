import pandas as pd
from torch.utils.data import Dataset

from typing import Tuple, Dict

class BGE_FTDataset(Dataset):
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

def create_datasets(config: Dict) -> Tuple[BGE_FTDataset]:
    """
    Create datasets for training, validation and testing.
    """
    return BGE_FTDataset("train", config), BGE_FTDataset("valid", config), BGE_FTDataset("test", config)
    