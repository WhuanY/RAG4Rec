import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from logging import getLogger
from util import dict2device

class Trainer(object):
    """The Trainer for training and evaluation strategies.

    Initializing the Trainer needs two parameters: `config` and `model`.
    - `config` records the parameters information for controlling training and evaluation,
    such as `learning_rate`, `epochs`, `eval_step` and so on.
    - `model` is the instantiated object of a Model Class.
    """
    def __init__(self, config, model):
        self.config = config 
        self.model = model



        self.logger = getLogger()
        self.learner = config['learner'].lower()
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = config['eval_step']
        self.early_stopping_epochs = config['early_stopping_epochs']
        self.clip_grad_norm = config['clip_grad_norm']
        self.device = config['device']
        self.optimizer = self._build_optimizer()

        self.start_epoch = 0

    def train(self, train_data, valid_data):
        """Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        train_loss = valid_loss = float('inf')
        # the below init values are for early stopping
        best_valid = cur_best_valid = float('inf')
        cur_step_from_best_val = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            train_loss = self._train_epoch(epoch_idx, train_data) # mean loss of this epoch
            
            # valid
            valid_loss = self._valid_epoch(epoch_idx, valid_data) # mean loss of this epoch
            if self.verbose:
                self.logger.info(f"Epoch {epoch_idx} Train mean Loss: {train_loss:.4f}, Valid mean Loss: {valid_loss:.4f}")
                
            # early stopping
            if (epoch_idx + 1) % self.eval_step == 0:
                if self.verbose:
                    self.logger.info(f"Epoch {epoch_idx + 1} starts early stopping check.")
                cur_best_valid, cur_step_from_best_val, stop_flag, update_flag = self._early_stopping(
                    valid_loss, cur_best_valid, cur_step_from_best_val, self.early_stopping_epochs, lower_is_better=True) # -> best, cur_step, stop_flag, update_flag
                
                if update_flag:
                    best_valid = cur_best_valid
            
                if stop_flag:
                    if self.verbose:
                        self.logger.info(f"Early stopping at epoch {epoch_idx}")
                    break
        
        return best_valid

    @torch.no_grad()
    def evaluate(self, eval_data):
        """Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            save_score (bool): Save .score file to running dir if ``True``. Defaults to ``False``.
            group (str): Which group to evaluate, can be ``all``, ``low``, ``high``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()
        batch_matrix_list = []
        iter_data = (
            tqdm(
                enumerate(eval_data),
                total=len(eval_data),
                desc=f"Evaluate   ",
            )
        )
        for batch_idx, batched_data in iter_data:
            interaction = batched_data
            scores = self.model.predict(dict2device(interaction, self.device))
            i

            batch_matrix = self.evaluator.collect(interaction, scores)
            batch_matrix_list.append(batch_matrix)
        result, result_str = self.evaluator.evaluate(batch_matrix_list)

        return result, result_str

    def _build_optimizer(self):
        """Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        opt2method = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adagrad': optim.Adagrad,
            'rmsprop': optim.RMSprop,
            'sparse_adam': optim.SparseAdam
        }

        if self.learner in opt2method:
            optimizer = opt2method[self.learner](self.model.parameters(), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, epoch_idx:int, train_loader: DataLoader):
        self.model.train()
        total_loss= 0
        total_batches = len(train_loader)

        for step, sample in enumerate(train_loader):
            sample = dict2device(sample, self.device)
            label = sample['labels'].to(self.device) 
            self.optimizer.zero_grad()
            
            output = self.model(sample)
            loss = self.model.calculate_loss(output, sample)
            loss.backward()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            return total_loss / total_batches

    def _valid_epoch(self, epoch_idx:int, valid_loader: DataLoader):
        """Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result, valid_result_str = self.evaluate(valid_loader, load_best_model=False)
        valid_score = valid_result[self.valid_metric]
        return valid_score, valid_result, valid_result_str

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
