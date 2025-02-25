import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from logging import getLogger
from util import ensure_dir, get_local_time, dict2device
from tqdm import tqdm

from evaluator import Evaluator

class Trainer(object):
    """The Trainer for training and evaluation strategies.

    Initializing the Trainer needs two parameters: `config` and `model`.
    - `config` records the parameters information for controlling training and evaluation,
    such as `learning_rate`, `epochs`, `eval_step` and so on.
    - `model` is the instantiated object of a Model Class.
    """
    def __init__(self, config, model):
        # overall attributes
        self.logger = getLogger()
        self.config = config 
        # train and evaluate
        self.model = model
        self.learner = config['learner'].lower()
        self.learning_rate = config['learning_rate']
        self.max_epochs = config['max_epochs']
        self.eval_step = config['eval_step']
        self.stopping_steps = config['stopping_steps']
        self.lower_is_better = config['lower_is_better']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_matric = config['valid_metric']
        self.device = config['device']
        self.optimizer = self._build_optimizer()
        self.evaluator = Evaluator(config)
        # save and continue training
        self.start_epoch = 0
        self.checkpoint_dir = config['checkpoint_dir']
        self.best_model_path = None
        # other
        self.verbose = True

    def train(self, train_data, valid_data, should_save=True):
        """Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            should_save (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        train_loss = valid_score = float('inf')
        # the below init values are for early stopping
        self.best_valid_score = cur_best_valid_score = float('inf') if self.lower_is_better else float('-inf')
        cur_step_from_best_val = 0

        for epoch_idx in range(self.start_epoch, self.max_epochs):
            self.cur_epoch = epoch_idx
            # train
            self.logger.info(f"Train Epoch {self.cur_epoch}/{self.max_epochs}")
            train_loss = self._train_epoch(epoch_idx, train_data) # mean loss of this epoch
            
            # valid
            valid_score, valid_result = self._valid_epoch(epoch_idx, valid_data) # mean loss of this epoch
            if self.verbose:
                self.logger.info(f"Epoch {epoch_idx} Train mean Loss: {train_loss:.4f}, Valid score: {valid_score:.4f}")
                
            # early stopping
            if (epoch_idx + 1) % self.eval_step == 0:
                if self.verbose:
                    self.logger.info(f"Epoch {epoch_idx + 1} starts early stopping check.")
                cur_best_valid_score, cur_step_from_best_val, stop_flag, update_flag = self._early_stopping(
                    valid_score, 
                    cur_best_valid_score, 
                    cur_step_from_best_val, 
                    self.stopping_steps, 
                    self.lower_is_better) # -> best, cur_step, stop_flag, update_flag
                
                if update_flag:
                    if should_save:
                        self._save_checkpoint(epoch_idx)
                    self.best_valid_score = cur_best_valid_score
                    self.best_valid_result = valid_result
            
                if stop_flag:
                    if self.verbose:
                        self.logger.info(f"Early stopping at epoch {epoch_idx}")
                    break
        
        return self.best_valid_score, self.best_valid_result
    
    def resume_checkpoint(self, resume_file):
        """Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file
        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning('Architecture configuration given in config file is different from that of checkpoint. '
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=False):
        """Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.

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
        for batch_idx, interaction in iter_data:
            scores = self.model.predict(dict2device(interaction, self.device)) # (bs)
            uid, score, label = interaction['geek_id'], scores, interaction['labels']
            batch_matrix = self.evaluator.collect(uid, score, label)
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
        for step, sample in tqdm(enumerate(train_loader), 
                                 total= total_batches):
            sample = dict2device(sample, self.device)
            self.optimizer.zero_grad()
            output = self.model(sample)
            loss = self.model.calculate_loss(output, sample)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        
        self.logger.info(f"Train Epoch {epoch_idx} loss: {total_loss / total_batches}")
        return total_loss / total_batches

    def _valid_epoch(self, epoch_idx:int, valid_loader: DataLoader):
        """Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.

        Returns:
            valid_score (float): the valid score
        """
        valid_result, valid_result_str = self.evaluate(valid_loader, load_best_model=False)
        valid_score = valid_result[self.valid_matric]
        return valid_score, valid_result

    def _early_stopping(self, value, best, cur_step, max_step, lower_is_better):
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

        better = (value <= best if lower_is_better else value > best)
        if better:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
        return best, cur_step, stop_flag, update_flag
    
    def _save_checkpoint(self, epoch):
        """Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
        """
        ensure_dir(self.checkpoint_dir)
        saved_model_filepath = os.path.join(self.checkpoint_dir, f'{self.config["model"]}-{get_local_time()}.pth')
        
        if self.best_model_path and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)
        
        self.best_model_path = saved_model_filepath
        
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_epoch': self.cur_epoch,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, saved_model_filepath)

    def _check_nan(self, loss):
        if torch.isnan(loss).any():
            raise ValueError("Model diverged with loss = NaN")
        return
