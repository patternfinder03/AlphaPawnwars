import numpy as np
print(np.__version__)


import torch
print(torch.__version__)
torch.manual_seed(0)

from Resnet.resnet import ResNet
from alphazero.alphaZeroParallel import AlphaZeroParallel
from envs.ChessEnvPawnParallel import ChessEPWP


# Trains model with parametesr


game = ChessEPWP()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ResNet(game, 5, 256, device)

model.load_state_dict(torch.load("./Models/model_4_ChessPawnWarsParallel7No-1OtherOneAccIs-1withThreads5blocksLessMul2000.pt", map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 250,
    'num_iterations': 10,
    'num_selfPlay_iterations': 4000,
    'num_parallel_games' : 250,
    'num_epochs': 7,
    'batch_size': 128,
    'iteration_num': 6,
    'temperature': 1.25,
    'dirichlet_epsilon': .3,
    'dirichlet_alpha': .25
}

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()
