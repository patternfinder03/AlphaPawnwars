import pandas as pd
from MCTS import MCTS, MCTSParallel
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm.notebook import trange
import json
from stockfish import Stockfish
from concurrent.futures import ThreadPoolExecutor

class AlphaZeroStockfishParallel:
    # Initialize the AlphaZero module
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self.stockfish = Stockfish(path=r"/src/stockfish/stockfish-windows-x86-64-avx2.exe")


    # Play self play games. At each move create a tree search using the policy model.
    # After creating tree, pick branch with highest likelihood of leading to a win
    # with some randomless to encourage exploration. Then, if the game is over
    # add it to the memory to learn after the iteration
    def selfPlay(self):

        # Create games and memory
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        # While games are still going on, get the states and make an action.
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size,
                                          p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                # Add to memory
                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
            # Update Player
            player = self.game.get_opponent(player)

        return return_memory

    def makeStockFishMovesFromBoardState(self, state):
        fen_string = state.fen()
        self.stockfish.set_fen_position(fen_string)

        action_str = self.stockfish.get_best_move()
        action = self.game.move_to_int[action_str]

        action_probs = np.zeros(self.game.action_size)
        action_probs[action - 1] = 1  # Set the probability of the best move to 1

        return action, action_probs

    def makeStockFishMovesFromBoardStates(self, states):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.makeStockFishMovesFromBoardState, states))

        # Unzip the combined results into separate lists
        moves, probs = zip(*results)

        return list(moves), np.array(probs)



    def playAgainstStockfish(self):

        # Create games and memory
        return_memory = []
        player = 1
        stockfish_player = np.random.choice([-1, 1])
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        # While games are still going on, get the states and make an action.
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)

            if player != stockfish_player:
                self.mcts.search(neutral_states, spGames)

                for i in range(len(spGames))[::-1]:
                    spg = spGames[i]

                    action_probs = np.zeros(self.game.action_size)
                    for child in spg.root.children:
                        action_probs[child.action_taken] = child.visit_count
                    action_probs /= np.sum(action_probs)

                    spg.memory.append((spg.root.state, action_probs, player))

                    temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                    temperature_action_probs /= np.sum(temperature_action_probs)
                    action = np.random.choice(self.game.action_size,
                                              p=temperature_action_probs)

                    spg.state = self.game.get_next_state(spg.state, action, player)

                    value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                    # Add to memory
                    if is_terminal:
                        for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                            hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                            return_memory.append((
                                self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome
                            ))
                        del spGames[i]
            else:
                all_actions, all_action_probs = self.makeStockFishMovesFromBoardStates(neutral_states)
                for i in range(len(spGames))[::-1]:
                    spg = spGames[i]

                    action_probs = all_action_probs[i]
                    action = all_actions[i]

                    spg.memory.append((spg.root.state, action_probs, player))

                    spg.state = self.game.get_next_state(spg.state, action, player)

                    value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                    # Add to memory
                    if is_terminal:
                        for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                            hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                            return_memory.append((
                                self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome
                            ))
                        del spGames[i]



            # Update Player
            player = self.game.get_opponent(player)

        return return_memory

    # Train model based off of memory data
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args[
                'batch_size'])]  # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error

            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            state = state.squeeze(dim=1)
            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            print(policy_loss, value_loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # Plays self play games then trains off that data
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(
                    self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.playAgainstStockfish()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration + self.args['iteration_num']}_{self.game}Parallel.pt")
            torch.save(self.optimizer.state_dict(),
                       f"optimizer_{iteration + self.args['iteration_num']}_{self.game}Parallel.pt")


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None




# def makeStockFishMovesFromBoardState(self, state):
#     fen_string = state.fen()
#     self.stockfish.set_fen_position(fen_string)
#
#     top_moves = self.stockfish.get_top_moves(5)
#     action_str = top_moves[0]['Move']
#     action = self.game.move_to_int[action_str]
#
#     action_probs = np.zeros(self.game.action_size)
#     weights = 1 / np.arange(1, 5 + 1)
#
#     iterator_weights = 0
#     for move in top_moves:
#         action_int = self.game.move_to_int[move['Move']]
#         action_probs[action_int - 1] = weights[iterator_weights]
#         iterator_weights += 1
#
#     return action, action_probs
#
# def makeStockFishMovesFromBoardStates(self, states):
#     with ProcessPoolExecutor() as executor:
#         results = list(executor.map(self.makeStockFishMovesFromBoardState, states))
#
#     # Unzip the combined results into separate lists
#     moves, probs = zip(*results)
#
#     return list(moves), np.array(probs)