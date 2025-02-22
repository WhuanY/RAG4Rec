import os
import pandas as pd
import re
from typing import Union, List
import glob
from logging import getLogger
from tqdm import tqdm
import torch
import torch.nn.functional as F

class Pool_PJF(object):
    """
    PJFPool simulates a database that stores all the data needed for the project. 
    Also simulates useful API for easy data access.
    """
    def __init__(self, config):
        self.logger = getLogger()
        self.config = config
        self._load_ids()

    def get_geek_id(self, token):
        return self.geek_token2id[token]

    def get_job_id(self, token):
        return self.job_token2id[token]
        
    def _load_ids(self):
        """
        load the token2id and id2token mappings for geeks and jobs.
        """
        for target in ['geek', 'job']:
            token2id = {}
            id2token = []
            filepath = os.path.join(self.config['dataset_path'], f'{target}.token')
            self.logger.info(f'Generating {target} ids from {filepath}...')
            with open(filepath, 'r') as file:
                next(file) # skip the first line
                for i, line in enumerate(file):
                    token = line.strip()
                    token2id[token] = i
                    id2token.append(token)
            setattr(self, f'{target}_token2id', token2id) # geek_token2id, job_token2id
            setattr(self, f'{target}_id2token', id2token) # geek_id2token, job_id2token
            setattr(self, f'{target}_num', len(id2token)) # geek_num, job_num
        
    def __str__(self):
        return '\n\t'.join(['Pool:'] + [
            f'{self.geek_num} geeks',
            f'{self.job_num} jobs'
        ])

    def __repr__(self):
        return self.__str__()
    
class Pool_Text(Pool_PJF):
    """
    TextPool is a subclass of PJFPool that loads the text data.
    Sets up the following attributes:
    geek_texts: list of geek texts. 
    job_texts: list of job texts
    """
    def __init__(self, config):
        super(Pool_Text, self).__init__(config)
        self._load_texts()

    def set_texts(self, target: str, target_id:int, text:Union[str, List[str]]):
        """
        Change the text description of an entity.
        target: 'geek' or 'job'
        target_id: the id of the target entity
        text: the new text description. Can be a string or a list of strings.
        """
        assert target in ['geek', 'job']
        li = getattr(self, f'{target}_id2texts')
        li[target_id] = text
        setattr(self, f'{target}_id2texts', li)
    
    def get_texts(self, target: str, target_id: int):
        """
        Get the text description of an entity.
        target: 'geek' or 'job'
        target_id: the id of the target entity
        """
        assert target in ['geek', 'job']
        return getattr(self, f'{target}_id2texts')[target_id]

    def _load_texts(self):
        for target in ['geek', 'job']:
            id2texts = [""] * getattr(self, f'{target}_num')

            if target == 'geek':
                filepath = os.path.join(self.config['dataset_path'], f'*.udoc')
            elif target == 'job':
                filepath = os.path.join(self.config['dataset_path'], f'*.idoc')
                
            self.logger.info(f'Load {target} texts from {filepath}')
            file = glob.glob(filepath)[0]
            temp_df = pd.read_csv(file, sep = '\t')
            if target == 'geek':
                for i, row in temp_df.iterrows():
                    geek_id = self.get_geek_id(row['user_id:token'])
                    id2texts[geek_id] = self._gen_cv(row)
            elif target == 'job':
                for i, row in temp_df.iterrows():
                    job_id = self.get_job_id(row['job_id:token'])
                    id2texts[job_id] = row['job_doc:token_seq']
            setattr(self, f'{target}_id2texts', id2texts)

    def __str__(self):
        return '\n\t'.join(['TextPool:'] + [
            f'{self.geek_num} geeks',
            f'{self.job_num} jobs',
            f'{len(self.geek_id2texts)} geek texts',
            f'{len(self.job_id2texts)} job texts'
        ])

    def __repr__(self):
        return self.__str__()
    
    # functions below are utility functions
    def _clean_job_text(self, text):
        illegal_set = ',.;?!~[]\'"@#$%^&*()-_=+{}\\`～·！¥（）—「」【】|、“”《<》>？，。…：'   # 定义非法字符

        for c in illegal_set:
            text = text.replace(c, ' ')     # 非法字符 替换为 空格
        for pattern in ['岗位职责', '职位描述', '工作内容', '岗位描述', '岗位说明', '工作职责', '你的职责']:
            text = text.replace(pattern, '')   # 内容头部替换为空格
        text = ' '.join([_ for _ in text.split(' ') if len(_) > 0])
        return text    # 空格间隔
        
    def _gen_cv(self, row):
        """
        Generates a list of strings summarizing certain fields from a row.

        This function uses predefined headers to label specific fields and constructs
        a combined string for each field that exists in the provided row data. The row
        is expected to be a dictionary- or series-like structure where fields can be
        checked for existence. If a field is missing or set to NaN, a default placeholder
        is used.

        Args:
            row: A dictionary or series-like object containing the fields to extract.

        Returns:
            A list of formatted strings, where each string consists of a header and
            the corresponding field value.
        """
        # Field mappings with their headers
        field_mappings = {
            "experience:token_seq": "【工作经历】:",
            "desire_jd_industry_id:token_seq": "【期望行业】: ",
            "cur_industry_id:token_seq": "【当前行业】:",
            "cur_jd_type:token_seq": "【职位标签】:"
        }
        li = []
        # Add each field with its header
        for field, header in field_mappings.items():
            if field in row:
                _ = row[field] if pd.notna(row[field]) else "无 "
                item = header + _
                li.append(item)
        return li # [工作经历, 期望行业, 当前行业, 职位标签]

class Pool_PJFNN(Pool_Text):
    def __init__(self, config):
        super().__init__(config)
        # split original texts
        self._split_jobs()
    
    def _split_jobs(self):
        """
        Split the job sentences into list of sentences.
        """
        for job_id, job_text in enumerate(self.job_id2texts):
            job_text = self.__clean_job_text(job_text)
            job_items = self.__split_job_sent(job_text) 
            self.set_texts('job', job_id, job_items)

    def __clean_job_text(self, text):
        illegal_set = ',.;?!~[]\'"@#$%^&*()-_=+{}\\`～·！¥（）—「」【】|、“”《<》>？，。…：'   # 定义非法字符

        for c in illegal_set:
            text = text.replace(c, ' ')     # 非法字符 替换为 空格
        for pattern in ['岗位职责', '职位描述', '工作内容', '岗位描述', '岗位说明', '工作职责', '你的职责']:
            text = text.replace(pattern, '')   # 内容头部替换为空格
        text = ' '.join([_ for _ in text.split(' ') if len(_) > 0])
        return text    # 空格间隔

    def __split_job_sent(self, text, max_items=20):
        text = re.split('(?:[0-9][.;。：．•）\)])', text)  # 按照数字分割包括  1.  1;  1。  1：  1) 等
        ans = []
        for t in text:
            for tt in re.split('(?:[\ ][0-9][、，])', t):  #
                for ttt in re.split('(?:^1[、，])', tt):   # 1、
                    for tttt in re.split('(?:\([0-9]\))', ttt):   # (1)
                        ans += re.split('(?:[。；…●])', tttt)
                        
        ans = ans[:max_items]
        return [_.strip() for _ in ans if len(_.strip()) > 0] 

if __name__ == "__main__":
    test_config = {'dataset_path': '~/RAG4Rec/dataset'}
    print("Testing PJFNN")
    pjfnnpool = Pool_PJFNN(test_config)
    print(pjfnnpool)