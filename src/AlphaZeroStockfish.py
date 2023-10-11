import pandas as pd

from MCTS import MCTS, MCTSParallel
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange
import json
from stockfish import Stockfish

class AlphaZeroSF:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.stockfish = Stockfish(path=r"C:\Users\theal\PycharmProjects\AlphaPawnwars\src\stockfish\stockfish-windows-x86-64-avx2.exe")



    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def playAgainstStockfish(self):
        memory = []
        player = 1

        stockfish_player = np.random.choice([-1, 1])

        print(stockfish_player, 'STOCKFISH COLOR')

        state = self.game.get_initial_state()

        self.stockfish.set_fen_position("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1", True)

        while True:
            if player != stockfish_player:
                neutral_state = self.game.change_perspective(state, player)
                action_probs = self.mcts.search(neutral_state)

                # print((neutral_state, action_probs, player), 'ai')
                memory.append((neutral_state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                # print(self.game.int_to_move[str(action)], 'AI')

                action_str =  self.game.int_to_move[str(action)]
            else:
                # Getting neutral state --------------------
                neutral_state = self.game.change_perspective(state, player)
                current_fen = state.fen()
                parts = current_fen.split(' ')
                if stockfish_player == 1:
                    parts[1] = 'w'
                elif stockfish_player == -1:
                    parts[1] = 'b'
                new_fen = ' '.join(parts)
                self.stockfish.set_fen_position(new_fen)
                # print(self.stockfish.get_fen_position(), self.stockfish.get_board_visual(), 'WHAT STOCKFISH SEE')
                # -------------------------------------------

                # Make lower if too slow
                action_str = self.stockfish.get_best_move()
                action_probs = np.zeros(self.game.action_size)
                action = self.game.move_to_int[action_str]
                action_probs[action - 1] = 1

                # print((neutral_state, action_probs, player))
                memory.append((neutral_state, action_probs, player))


                # print(action_str, 'STOCKFISH')


            state = self.game.get_next_state(state, action, player)

            self.stockfish.make_moves_from_current_position([action_str])

            # print("State after", self.stockfish.get_board_visual(), state, self.stockfish.get_fen_position())

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)



    def stockfishAgainstStockfish(self):
        memory = []
        player = 1


        state = self.game.get_initial_state()

        self.stockfish.set_fen_position("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1", True)

        while True:
            if player == 1:
                neutral_state = self.game.change_perspective(state, player)
                current_fen = state.fen()
                parts = current_fen.split(' ')

                parts[1] = 'w'
                new_fen = ' '.join(parts)
                self.stockfish.set_fen_position(new_fen)
                # print(self.stockfish.get_fen_position(), self.stockfish.get_board_visual(), 'WHAT STOCKFISH SEE')
                # -------------------------------------------

                # Make lower if too slow
                if random.randint(1, 2) == 1:
                    action_str = self.stockfish.get_best_move(np.random.randint(100, 25001))
                else:
                    action_str = self.stockfish.get_best_move()
                action_probs = np.zeros(self.game.action_size)
                action = self.game.move_to_int[action_str]
                action_probs[action - 1] = 1

                # print((neutral_state, action_probs, player))
                memory.append((neutral_state, action_probs, player))
            else:
                # Getting neutral state --------------------
                neutral_state = self.game.change_perspective(state, player)
                current_fen = state.fen()
                parts = current_fen.split(' ')

                parts[1] = 'b'
                new_fen = ' '.join(parts)
                self.stockfish.set_fen_position(new_fen)
                # print(self.stockfish.get_fen_position(), self.stockfish.get_board_visual(), 'WHAT STOCKFISH SEE')
                # -------------------------------------------

                # Make lower if too slow
                if random.randint(1, 2) == 1:
                    action_str = self.stockfish.get_best_move(np.random.randint(100, 25001))
                else:
                    action_str = self.stockfish.get_best_move()
                action_probs = np.zeros(self.game.action_size)
                action = self.game.move_to_int[action_str]
                action_probs[action - 1] = 1

                # print((neutral_state, action_probs, player))
                memory.append((neutral_state, action_probs, player))


                # print(action_str, 'STOCKFISH')


            state = self.game.get_next_state(state, action, player)

            self.stockfish.make_moves_from_current_position([action_str])

            # print("State after", self.stockfish.get_board_visual(), self.stockfish.get_fen_position())

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)


    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx+self.args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            print(policy_loss, value_loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.stockfishAgainstStockfish()
                print("played")

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration + self.args['iteration_num']}_{self.game}PW.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration + self.args['iteration_num']}_{self.game}PW.pt")

