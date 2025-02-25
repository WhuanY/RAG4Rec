import os
import random
import logging
import datetime
import importlib
import numpy as np
import torch


def init_seed(seed, reproducibility):
    """ init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_local_time():
    """Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> init_logger(config)
        >>> logger = logging.getLogger()
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)

    logfilename = '{}-{}.log'.format(config['model'], get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M:%S"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        handlers=[fh, sh]
    )

def dynamic_load(config, module_path, base_class):
    """
    Load the required class based on the model name and base class.
    
    Args:
        config (dict): Configuration dictionary containing the model name.
        module_path (str): Path to the module containing the class.
        base_class (str): The base class type (e.g., "Dataset", "Dataloader", "Model").
    
    Returns:
        class: The dynamically loaded class.
    """
    module = importlib.import_module(module_path)
    class_name = base_class + '_' + config['model']
    return getattr(module, class_name)

def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def dict2device(data, device):
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = dict2device(v, device)
        elif isinstance(v, torch.Tensor):
            data[k] = v.to(device)
    return data

if __name__ == "__main__":
    print(get_local_time())