import torch
import torch.nn as nn
import torch.nn.functional as F
from Node.node import Node
import numpy as np
import chess
class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                 * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


    @torch.no_grad()
    def search_biased(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                 * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)

        player = 1 if state.turn == chess.WHITE else -1

        for child in root.children:
            move = chess.Move.from_uci(self.game.int_to_move[str(child.action_taken)])

            if not state.piece_at(move.to_square) is None:
                action_probs[child.action_taken] = child.visit_count * 20
            elif str(state.piece_at(move.from_square)).upper() == "K":
                from_rank = move.from_square // 8  # Convert square number to rank (0-7)
                to_rank = move.to_square // 8  # Convert square number to rank (0-7)

                # Check direction of movement based on player perspective
                if player == 1:  # White's perspective
                    if to_rank > from_rank:  # King moved forward
                        multiplier = 5
                    elif to_rank == from_rank:  # King moved sideways
                        multiplier = 1
                    else:  # King moved backward
                        multiplier = 1
                elif player == -1:  # Black's perspective
                    if to_rank < from_rank:  # King moved forward
                        multiplier = 5
                    elif to_rank == from_rank:  # King moved sideways
                        multiplier = 1
                    else:  # King moved backward
                        multiplier = 1
                else:
                    print("ERROR")


                action_probs[child.action_taken] = child.visit_count * multiplier
            else:
                action_probs[child.action_taken] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs