import torch
import torch.nn as nn
import torch.nn.functional as F
from Node import Node
import numpy as np

# Runs Monte-Carlo Tree search on board states
class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):

        # Get current policies for action probabilites from encoded board states
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        # Turns logits into probabilites
        policy = torch.softmax(policy, axis=1).cpu().numpy()

        # Applies arg parameters to focus more on exploration vs exploitation
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                 * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])


        # Make root Node and expand once based on current policy
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)

        # In each game state, search a specified number of moves to add to tree
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                # While there are still child nodes, keep traversing tree until arriving at a new leaf Node
                while node.is_fully_expanded():
                    node = node.select()

                # See if said leaf Node is done, and if so what result
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)

                # Because tree goes player->opponent->player->opponent we need to reverse the value for tree backpropogation
                value = self.game.get_opponent_value(value)

                # If done back_propogate, otherwise update the new Node
                if is_terminal:
                    node.backpropagate(value)

                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if
                                  spGames[mappingIdx].node is not None]

            # Update the trees for all games
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]

                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)

