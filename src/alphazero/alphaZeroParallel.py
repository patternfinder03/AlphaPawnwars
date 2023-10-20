import pandas as pd
from mcts.mctsParallel import MCTSParallel
from mcts.mctsSingle import MCTS
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm.notebook import trange
import json
import chess
import concurrent.futures

class AlphaZeroParallel:
    # Initialize the AlphaZero module
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

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
                num_pawns = len([p for p in spg.state.piece_map().values() if str(p) == "P" or str(p) == "p"])
                for child in spg.root.children:
                    move = chess.Move.from_uci(self.game.int_to_move[str(child.action_taken)])

                    if not spg.state.piece_at(move.to_square) is None:
                        action_probs[child.action_taken] = child.visit_count * 20
                    elif str(spg.state.piece_at(move.from_square)).upper() == "K":
                        from_rank = move.from_square // 8  # Convert square number to rank (0-7)
                        to_rank = move.to_square // 8  # Convert square number to rank (0-7)

                        from_file = move.from_square % 8  # Convert square number to file (0-7)
                        to_file = move.to_square % 8  # Convert square number to file (0-7)

                        # Check direction of movement based on player perspective
                        if player == 1:  # White's perspective
                            if to_rank > from_rank:  # King moved forward
                                multiplier = 5
                            elif to_rank == from_rank:  # King moved sideways
                                multiplier = 3
                            else:  # King moved backward
                                multiplier = 2
                        else:  # Black's perspective
                            if to_rank < from_rank:  # King moved forward
                                multiplier = 5
                            elif to_rank == from_rank:  # King moved sideways
                                multiplier = 3
                            else:  # King moved backward
                                multiplier = 2

                        if num_pawns <= 4:  # Fewer or equal to 4 pawns
                            multiplier *= 0.25
                        elif num_pawns <= 6:  # Between 5 to 6 pawns
                            multiplier *= 0.5
                        elif num_pawns <= 8:  # Between 7 to 8 pawns
                            multiplier *= 0.66

                        action_probs[child.action_taken] = child.visit_count * multiplier
                    else:
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


    def selfPlayWithThreadsWithMultipliers(self):
        def process_single_game(spg, args, game):
            action_probs = np.zeros(game.action_size)
            # num_pawns = len([p for p in spg.state.piece_map().values() if str(p) == "P" or str(p) == "p"])

            for child in spg.root.children:

                move = chess.Move.from_uci(self.game.int_to_move[str(child.action_taken)])

                if not spg.state.piece_at(move.to_square) is None:
                    action_probs[child.action_taken] = child.visit_count * 20
                elif str(spg.state.piece_at(move.from_square)).upper() == "K":
                    from_rank = move.from_square // 8  # Convert square number to rank (0-7)
                    to_rank = move.to_square // 8  # Convert square number to rank (0-7)


                    # Check direction of movement based on player perspective
                    if player == 1:  # White's perspective
                        if to_rank > from_rank:  # King moved forward
                            multiplier = 2
                        elif to_rank == from_rank:  # King moved sideways
                            multiplier = 1.1
                        else:  # King moved backward
                            multiplier = 1
                    elif player == -1:  # Black's perspective
                        if to_rank < from_rank:  # King moved forward
                            multiplier = 2
                        elif to_rank == from_rank:  # King moved sideways
                            multiplier = 1.1
                        else:  # King moved backward
                            multiplier = 1
                    else:
                        print("ERROR")

                    # if num_pawns <= 4:  # Fewer or equal to 4 pawns
                    #     multiplier *= 0.25
                    # elif num_pawns <= 6:  # Between 5 to 6 pawns
                    #     multiplier *= 0.5
                    # elif num_pawns <= 8:  # Between 7 to 8 pawns
                    #     multiplier *= 0.66

                    action_probs[child.action_taken] = child.visit_count * multiplier
                else:
                    action_probs[child.action_taken] = child.visit_count

            action_probs /= np.sum(action_probs)

            spg.memory.append((spg.root.state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(game.action_size, p=temperature_action_probs)

            spg.state = game.get_next_state(spg.state, action, player)
            value, is_terminal = game.get_value_and_terminated(spg.state, action)

            if is_terminal:
                memories_for_this_game = []
                for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                    hist_outcome = value if hist_player == player else game.get_opponent_value(value)
                    memories_for_this_game.append((
                        game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return None, memories_for_this_game  # Return None to indicate the game ended
            else:
                return spg, []  # Game continues, so return the updated spg and an empty memory list

        # Create games and memory
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        while spGames:  # Continue processing games as long as there are non-terminated games
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spGames)
            updated_spGames = []

            # Use ThreadPoolExecutor to process games concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(process_single_game, spGames, [self.args] * len(spGames),
                                       [self.game] * len(spGames))

            # Collect results from each thread
            for updated_spg, new_memories in results:
                if updated_spg is not None:  # If the game wasn't terminated, add it to the updated list
                    updated_spGames.append(updated_spg)
                else:  # Game terminated, so we handle the memories
                    return_memory.extend(new_memories)

            spGames = updated_spGames  # Update spGames with the list of continuing games
            # Update Player
            player = self.game.get_opponent(player)

        return return_memory
    def selfPlayWithThreads(self):

        def process_single_game(spg, args, game):
            action_probs = np.zeros(game.action_size)
            # num_pawns = len([p for p in spg.state.piece_map().values() if str(p) == "P" or str(p) == "p"])

            for child in spg.root.children:

                # move = chess.Move.from_uci(self.game.int_to_move[str(child.action_taken)])

                action_probs[child.action_taken] = child.visit_count
                # if not spg.state.piece_at(move.to_square) is None:
                #     action_probs[child.action_taken] = child.visit_count * 20
                # elif str(spg.state.piece_at(move.from_square)).upper() == "K":
                #     from_rank = move.from_square // 8  # Convert square number to rank (0-7)
                #     to_rank = move.to_square // 8  # Convert square number to rank (0-7)
                #
                #
                #     # Check direction of movement based on player perspective
                #     if player == 1:  # White's perspective
                #         if to_rank > from_rank:  # King moved forward
                #             multiplier = 2
                #         elif to_rank == from_rank:  # King moved sideways
                #             multiplier = 1.1
                #         else:  # King moved backward
                #             multiplier = 1
                #     elif player == -1:  # Black's perspective
                #         if to_rank < from_rank:  # King moved forward
                #             multiplier = 2
                #         elif to_rank == from_rank:  # King moved sideways
                #             multiplier = 1.1
                #         else:  # King moved backward
                #             multiplier = 1
                #     else:
                #         print("ERROR")
                #
                #     # if num_pawns <= 4:  # Fewer or equal to 4 pawns
                #     #     multiplier *= 0.25
                #     # elif num_pawns <= 6:  # Between 5 to 6 pawns
                #     #     multiplier *= 0.5
                #     # elif num_pawns <= 8:  # Between 7 to 8 pawns
                #     #     multiplier *= 0.66
                #
                #     action_probs[child.action_taken] = child.visit_count * multiplier
                # else:
                #     action_probs[child.action_taken] = child.visit_count

            action_probs /= np.sum(action_probs)

            spg.memory.append((spg.root.state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(game.action_size, p=temperature_action_probs)

            spg.state = game.get_next_state(spg.state, action, player)
            value, is_terminal = game.get_value_and_terminated(spg.state, action)

            if is_terminal:
                memories_for_this_game = []
                for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                    hist_outcome = value if hist_player == player else game.get_opponent_value(value)
                    memories_for_this_game.append((
                        game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return None, memories_for_this_game  # Return None to indicate the game ended
            else:
                return spg, []  # Game continues, so return the updated spg and an empty memory list

        # Create games and memory
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        while spGames:  # Continue processing games as long as there are non-terminated games
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spGames)
            updated_spGames = []

            # Use ThreadPoolExecutor to process games concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(process_single_game, spGames, [self.args] * len(spGames),
                                       [self.game] * len(spGames))

            # Collect results from each thread
            for updated_spg, new_memories in results:
                if updated_spg is not None:  # If the game wasn't terminated, add it to the updated list
                    updated_spGames.append(updated_spg)
                else:  # Game terminated, so we handle the memories
                    return_memory.extend(new_memories)

            spGames = updated_spGames  # Update spGames with the list of continuing games
            # Update Player
            player = self.game.get_opponent(player)

        return return_memory

    # Train model based off of memory data
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error

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
                memory += self.selfPlayWithThreads()
                # memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"./Models/model_{iteration + self.args['iteration_num']}_{self.game}withThreads5blocksNoMul4000.pt")
            torch.save(self.optimizer.state_dict(), f"./Models/optimizer_{iteration + self.args['iteration_num']}_{self.game}withThreads5blocksNoMul4000.pt")

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

