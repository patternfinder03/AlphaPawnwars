import numpy as np
print(np.__version__)


import torch
print(torch.__version__)
torch.manual_seed(0)

from RsNet import ResNet
from alphazero import AlphaZeroParallel
from ChessEnvPawnParalle import ChessEPWP


# Trains model with parametesr


game = ChessEPWP()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ResNet(game, 10, 256, device)

# model.load_state_dict(torch.load("model_5_ChessPawnWarsParallel2.pt", map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 200,
    'num_iterations': 1,
    'num_selfPlay_iterations': 100,
    'num_parallel_games' : 100,
    'num_epochs': 4,
    'batch_size': 128,
    'iteration_num': 0,
    'temperature': 1.25,
    'dirichlet_epsilon': .3,
    'dirichlet_alpha': .25
}

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()
