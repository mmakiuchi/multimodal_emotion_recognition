"""
    Name: main.py
    Description: Main function to load the data, define the model and train.
"""

from solver import Solver

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from speech.data_loader import get_dataloaders
from speech.parser_helper import get_config

from torch.backends import cudnn

# Name of the train and test pkl files
train_pkl = "train"
test_pkl = "test"

def get_solver(train_loader, test_loader, train_eval, train_1batch):
    """ Return the solver object """

    with open(config.out_dir + "/test_results.txt", "a") as txt_f:
        txt_f.write("[INFO] Getting the model... \n")
    solver = Solver(train_loader, test_loader, train_eval, train_1batch, config)

    return solver

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # get the data loaders
    print("[INFO] Getting the data loader...")
    with open(config.out_dir + "/test_results.txt", "a") as txt_f:
        txt_f.write("[INFO] Getting the data loaders... \n")
    
    # get spectrogram crops and corresponding features
    train_loader, test_loader, train_eval, train_1batch = get_dataloaders(config=config,
                                                                          train_pkl=train_pkl,
                                                                          test_pkl=test_pkl)
    print("[INFO] Finished getting the data loaders...")

    # solver
    solver = get_solver(train_loader, test_loader, train_eval, train_1batch)

    # train
    with open(config.out_dir + "/test_results.txt", "a") as txt_f:
        txt_f.write("[INFO] Training... \n")
    solver.train()

if __name__ == '__main__':
    config = get_config()
    main(config)
