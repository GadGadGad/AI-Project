# directory
from __future__ import annotations
import os

DATA_SAVE_DIR = "data/"
TRAINED_MODEL_DIR = "trained_models/"
TENSORBOARD_LOG_DIR = "tensorboard_log/"
RESULTS_DIR = "results/"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2015-01-01"  
TRAIN_END_DATE = "2020-01-01"

TEST_START_DATE = "2020-01-01"
TEST_END_DATE = "2022-01-01"

TRADE_START_DATE = "2022-01-01"
TRADE_END_DATE = "2024-06-01"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}

ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  
}

def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs("./" + directory)

