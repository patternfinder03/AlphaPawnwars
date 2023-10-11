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
                move = chess.Move.from_uci(self.game.int_to_move[str(child.action_taken)])

                if not spg.state.piece_at(move.to_square) is None:
                    action_probs[child.action_taken] = child.visit_count * 20
                elif str(spg.state.piece_at(move.from_square)) == "K":
                    # If king moves forward or side to side give bonus

                    # Don't really think being on first or second row should be beneficial

                    # If it's moving towards enemy pawns could probably be good but would probably be a nightmare to code

                    # Opposition could be lategame idea

                    action_probs[child.action_taken] = child.visit_count * 4
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