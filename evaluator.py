from typing import List
import unittest
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
from logging import getLogger

class Evaluator:
    """
    Evaluator for the BGE-FT model.
    """
    def __init__(self, config):
        #TODO: the label values are currently only assumed to be binary.
        # For further experiments, we need to make the label values more general.  
        self.logger = getLogger()
        self.topk = config['topk']
        self.uid2topk = {} # {uid: [(score, label), ...]}  
        self.metric2func = {
            "ndcg": self._ndcg,
            "precision": self._precision,
            "recall": self._recall,
            "map": self._map,
            "mrr": self._mrr,
            "auc": self._auc,
            "logloss": self._logloss,
            "f1": self._f1,
        }
        self.cls_metrics = ["auc", "logloss", "f1",]
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
            # s, l to cpu list
            s = s.cpu().tolist()
            l = l.cpu().tolist()
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
            for k in self.topk:
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
        for uid in self.uid2topk:
            rec = 0
            rel = 0
            for i, (score, label) in enumerate(self.uid2topk[uid][:k]):
                rec += 1
                rel += label # TODO: If label is not binary, this should (maybe) be modified.
            tot += rel / rec
        return tot / len(self.uid2topk)

    def _recall(self, k):
        tot = 0
        for uid in self.uid2topk:
            rec = 0
            rel = 0
            for i, (score, label) in enumerate(self.uid2topk[uid]):
                if i < k:
                    rec += label
                rel += label #TODO: If label is not binary, this should (maybe) be modified.
            if rel > 0:
                tot += rec / rel
        return tot / len(self.uid2topk)

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
        total_cnt = len(self.uid2topk)
        for uid, topk in self.uid2topk.items():
            score, labels = zip(*topk)
            if len(set(labels)) < 2:
                total_cnt -= 1
                continue 
            auc = roc_auc_score(labels, score)
            total_auc += auc
        return total_auc / total_cnt
        

    def _logloss(self):
        """
        Calculate the logloss.
        """
        total_logloss = 0
        total_cnt = len(self.uid2topk)
        for uid, topk in self.uid2topk.items():
            score, labels = zip(*topk)
            if len(set(labels)) < 2:
                total_cnt -= 1
                continue
            logloss = log_loss(labels, score)
            total_logloss += logloss
        return total_logloss / total_cnt

    def _f1(self):
        """
        Calculate the F1 score.
        """
        total_f1 = 0
        total_cnt = len(self.uid2topk)
        for uid, topk in self.uid2topk.items():
            score, labels = zip(*topk)
            if len(set(labels)) < 2:
                total_cnt -= 1
                continue
            # Convert scores to binary predictions (1 if score >= 0.5, else 0)
            predictions = [1 if s >= 0.5 else 0 for s in score]
            f1 = f1_score(labels, predictions)
            total_f1 += f1
        return total_f1 / total_cnt

    # other utility functions for evaluator
    def _format_str(self, result):
        res = ''
        for metric in result.keys():
            res += '\n\t{}:\t{:.4f}'.format(metric, result[metric])
        return res

if __name__ == '__main__':
    unittest.main()
