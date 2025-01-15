import argparse
from logging import getLogger

from config import Config
from data.dataset import create_datasets
from util import init_seed, init_logger, dynamic_load

def main(model, config_dict=None, saved=True):

    # running envs initialization
    config = Config(model, config_dict=config_dict)
    init_seed(42, True)
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # # data preparation
    # logger.info('Data preparation...')
    # train_dataset, valid_dataset, test_dataset = create_datasets(config)
    # logger.info(f"Data preparation finished.")

    # model loading
    logger.info('Model loading...')
    model = dynamic_load(config, 'model', 'Model')(config).to(f"cuda:{config['gpu_id']}")
    logger.info('Model loading finished.')
    print(model)
    # trainer loading and initialization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BGE_FT', help='Model to test.')
    args = parser.parse_args()

    main(model=args.model)