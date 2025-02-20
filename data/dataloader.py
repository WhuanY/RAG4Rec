import torch 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def construct_dataloader(config, datasets):
    """
    construct dataloader for training, validation and testing.
    config: dict, configuration for the dataloader.
    datasets: list, list of datasets for training, validation and testing.
    """
    
    param_list = [
        [*datasets],
        [config['train_batch_size']] + [config['eval_batch_size']] * 2,
        [True, False, False],
        [config['num_workers']] * 3,
        [config['pin_memory']] * 3,
    ]
    
    dataloaders = [
        DataLoader(
            dataset=ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=pm,
            collate_fn = get_collator(config)
        ) for ds, bs, shuffle, nw, pm in zip(*param_list)
    ]
    return dataloaders

def get_collator(config):
    """Get collator instance based on model name"""
    collator_map = {
        "PJFNN": Collator_PJFNN,
        # TODO: Add more collator.
    }
    
    model_name = config['model']
    
    collator_class = collator_map.get(model_name)
    if collator_class is None:
        raise ValueError(f"No collator found for model: {model_name}")
        
    return collator_class(config)

class Collator(object):
    """
    collator for dataloader. In the context of PJF, texts are usually used, 
    therefore, the collator is by default to handle texts.
    """
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

    def __call__(self, batch):
        return batch

class Collator_PJFNN(Collator):
    """Collator for PJFNN model"""
    def __init__(self, config):
        super().__init__(config)
    
    def __call__(self, batch):
        geek_texts = [item['geek_texts'] for item in batch]
        job_texts = [item['job_texts'] for item in batch]
        labels = [item['label'] for item in batch]

        max_geek_items = max([len(geek_text) for geek_text in geek_texts])
        max_job_items = max([len(job_text) for job_text in job_texts])

        # prepare batch
        batch_geek_encoded = []
        batch_job_encoded = []

        for geek_text, job_text in zip(geek_texts, job_texts):
            geek_encoded = self.tokenizer(geek_text, padding="max_length", truncation=True, max_length=256)['input_ids'] # (geek_item, max_seq_len)
            geek_encoded = torch.tensor(geek_encoded, dtype=torch.int64)  # Convert to tensor

            if len(geek_text) < max_geek_items:
                padding = torch.full((max_geek_items - len(geek_text), 256), self.tokenizer.pad_token_id, dtype=torch.int16) # (max_geek_item - geek_item, max_seq_len)
                geek_encoded = torch.cat((geek_encoded, padding), dim=0) # (max_geek_items, max_seq_len)

            if len(job_text) == 0:
                job_encoded = torch.full((max_job_items, 256), self.tokenizer.pad_token_id, dtype=torch.int16) # (max_job_items, max_seq_len)
            else:
                job_encoded = self.tokenizer(job_text, padding="max_length", truncation=True, max_length=256)['input_ids']
                job_encoded = torch.tensor(job_encoded, dtype=torch.int64) # (job_item, max_seq_len)
                if len(job_text) < max_job_items:
                    padding = torch.full((max_job_items - len(job_text), 256), self.tokenizer.pad_token_id, dtype=torch.int16)
                    job_encoded = torch.cat((job_encoded, padding), dim=0) # (max_job_items, max_seq_len)
            
            batch_geek_encoded.append(geek_encoded)
            batch_job_encoded.append(job_encoded)

        return {
            "geek_texts": torch.stack(batch_geek_encoded), # (bs, max_geek_item, max_seq_len)
            "job_texts": torch.stack(batch_job_encoded),    # (bs, max_geek_item, max_seq_len)
            "labels": torch.tensor(labels) # (bs, 1)
        }

if __name__ == "__main__":
    config = {
        'tokenizer_path': '/media/wuyuhuan/bge-small-zh',
        'model': 'PJFNN',
        'shuffle': False,
        'train_batch_size': 1,
        'eval_batch_size': 1,
        'num_workers': 1,
        'pin_memory': True
    }
    

    