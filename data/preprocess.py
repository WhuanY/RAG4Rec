import random
import pandas as pd
from logging import getLogger
from sklearn.model_selection import train_test_split





def preprocess(config):
    INPUT_FILE = './dataset/zhilian.inter'
    TRAIN_FILE = './dataset/dataset-train.inter'
    VALID_FILE = './dataset/dataset-valid.inter'
    TEST_FILE = './dataset/dataset-test.inter'

    random.seed(config['seed'])
    logger = getLogger()

    # Read the input file
    inter_df = pd.read_csv(INPUT_FILE, sep="\t")
    logger.info(f"Read file from \"{INPUT_FILE}\", Total records: {len(inter_df)}")

    # Dirty Data Cleaning
    #     1.  All same records. For each group of same records, keep the first one
    inter_df = inter_df.drop_duplicates(subset=None, keep="first")
    logger.info(f"Drop duplicates, remaining interaction records: {len(inter_df)}")
    #     2.  Drop Records Having same (geek_id:token, job_id:token) pair, but different labels
    inter_df = inter_df.drop_duplicates(subset=['user_id:token', 'job_id:token'], keep=False)
    logger.info(f"Drop confict records, remaining interaction records: {len(inter_df)}")

    # Split the data into training, validation, and test sets
    train_df, _ = train_test_split(inter_df, test_size=0.2, random_state=config['seed'])
    valid_df, test_df = train_test_split(_, test_size=0.5, random_state=config['seed'])
    logger.info(f"Train-Valid-Test Split. Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # write the data to files
    train_df.to_csv(TRAIN_FILE, sep="\t", index=False)
    valid_df.to_csv(VALID_FILE, sep="\t", index=False)
    test_df.to_csv(TEST_FILE, sep="\t", index=False)

    # with open(INPUT_FILE, 'r') as f:
    #     header = f.readline()
    #     lines = f.readlines()

    #     # Shuffle the lines
    #     random.shuffle(lines)

    #     # Calculate split index
    #     split_index_train = int(0.8 * len(lines))
    #     split_index_valid = split_index_train + int(0.1 * len(lines))

    #     # Split the lines into training and validation sets
    #     train_lines = lines[:split_index_train]
    #     valid_lines = lines[split_index_train:split_index_valid]
    #     test_lines = lines[split_index_valid:]
    #     with open(TRAIN_FILE, 'w') as f:
    #         f.write(header)
    #         f.writelines(train_lines)
    #     print(f"[INFO] Train set: {len(train_lines)} lines")
    #     with open(VALID_FILE, 'w') as f:
    #         f.write(header)
    #         f.writelines(valid_lines)
    #     print(f"[INFO] Valid set: {len(valid_lines)} lines")
    #     with open(TEST_FILE, 'w') as f:
    #         f.write(header)
    #         f.writelines(test_lines)
    #     print(f"[INFO] Test set: {len(test_lines)} lines")

    # print("[INFO] Finish Spliting. Done!")

if __name__ == "__main__":
    config = {
        'seed': 2021
    }
    preprocess(config)