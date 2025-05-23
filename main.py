from config import Config
from logging import getLogger
from data.dataloader import construct_dataloader
from util import init_seed, init_logger, dynamic_load
from data.preprocess import preprocess
from trainer import Trainer
import argparse


def main(model, config_dict=None, saved=True):
    # running envs initialization
    config = Config(model, config_dict=config_dict)
    import torch
    init_seed(config['seed'], True)
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # data preparation
    logger.info("Data Preprocessing...")
    preprocess(config)

    logger.info("Data Loading...")
    pool = dynamic_load(config, 'data', "Pool")(config)
    train_ds, valid_ds, test_ds = \
        dynamic_load(config, 'data', "Dataset")(config, pool, 'train'), \
        dynamic_load(config, 'data', "Dataset")(config, pool, 'valid'), \
        dynamic_load(config, 'data', "Dataset")(config, pool, 'test')
    
    train_loader, valid_loader, test_loader = construct_dataloader(config, [train_ds, valid_ds, test_ds])
    # model loading
    logger.info('Model Loading...')
    model = dynamic_load(config, 'model', "Model")(config).to(config['device'])
    logger.info('Model loading finished.')
    print(model)

    # trainer loading and initialization
    logger.info('Trainer loading...')
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.train(train_loader, valid_loader)
    logger.info('best valid result: {}'.format(best_valid_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='PJFNN', help='Model to test.')
    args = parser.parse_args()

    main(model=args.model)