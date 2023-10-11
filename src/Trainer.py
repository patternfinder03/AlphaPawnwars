import numpy as np
print(np.__version__)


import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

from tqdm.notebook import trange

import random
import math

import chess

import json

import os
from RsNet import ResNet
from AlphaZeroStockfish import AlphaZeroSF
from ChessEnvPawn import ChessEPW

game = ChessEPW()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ResNet(game, 10, 256, device)
model.load_state_dict(torch.load("model_15_ChessPawnWarsParallel3.pt", map_location=device))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 1,
    'num_iterations': 15,
    'num_selfPlay_iterations': 750,
    'num_parallel_games' : 10,
    'num_epochs': 5,
    'batch_size': 128,
    'iteration_num': 1,
    'temperature': 1.25,
    'dirichlet_epsilon': .3,
    'dirichlet_alpha': .25
}

alphaZero = AlphaZeroSF(model, optimizer, game, args)
alphaZero.learn()
