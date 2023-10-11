import time

import chess
import json
import numpy as np
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool



class ChessEPWP:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 1968 # Don't Change



        with open('../data/move_to_int.json', 'r') as json_file:
            self.move_to_int = json.load(json_file)

        with open('../data/int_to_move.json', 'r') as json_file:
            self.int_to_move = json.load(json_file)

        self.turn = 1

        self.PIECE_VALUES = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }


        self.count_stuffs = 0
        self.win_counts = 0

        self.game_count = 0

    def __repr__(self):
        return "ChessPawnWarsParallel7No-1OtherOneAccIs-1"


    def _convert_to_list(self, items):
        if isinstance(items, np.ndarray):
            return items.tolist()
        elif not isinstance(items, list):
            return [items]
        return items

    def _flatten_nested_list(self, nested_list):
        if isinstance(nested_list, np.ndarray):
            nested_list = nested_list.tolist()

        if isinstance(nested_list, chess.Board):  # replace 'Board' with your actual Board class
            nested_list = [nested_list]  # If it is a 'Board' object, we encapsulate it in a list

        return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]



    # Get's a 10 x 8 x 8 matrix to represent current state. Currently using ThreadPoolExecutor,
    # and have tried changing it to multiprocessing pool but that breaks everything
    def get_encoded_state(self, states):
        states = self._flatten_nested_list(states)

        squares_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        piece_to_index = {chess.PAWN: 0, chess.KING: 1}

        def isPassedPawn(state, square):
            # Determine the pawn's color
            pawn = state.piece_at(square)
            if pawn is None or pawn.piece_type != chess.PAWN:
                return False  # No pawn on the given square

            file = chess.square_file(square)
            rank = chess.square_rank(square)
            pawn_color = pawn.color

            enemy_pawns = state.pieces(chess.PAWN, not pawn_color)

            # Define the rank direction based on the pawn's color
            rank_direction = 1 if pawn_color == chess.WHITE else -1

            # Check every rank in front of the pawn
            current_rank = rank + rank_direction
            while 0 <= current_rank <= 7:
                for f in [file, file - 1, file + 1]:  # Check the pawn's file and adjacent files
                    if 0 <= f <= 7:  # Ensure we are within board boundaries
                        square_to_check = chess.square(f, current_rank)
                        # If there's an enemy pawn that can attack this square, our pawn is not passed
                        if state.is_attacked_by(not pawn_color, square_to_check):
                            return False
                current_rank += rank_direction

            return True

        def square_to_index(square):
            letter = chess.square_name(square)
            return 8 - int(letter[1]), squares_index[letter[0]]


        def encode_state(state):
            board3d = np.zeros((10, 8, 8))

            passed_pawn_white = False
            passed_pawn_black = False

            if state.turn:
                for piece in [chess.PAWN, chess.KING]:
                    for square in state.pieces(piece, chess.WHITE):
                        idx = np.unravel_index(square, (8, 8))
                        board3d[piece_to_index[piece]][7 - idx[0]][idx[1]] = 1

                        if not passed_pawn_white and piece == chess.PAWN:
                            if isPassedPawn(state, square):
                                column_index = chess.square_file(square)
                                board3d[8][:, column_index] = 1


                    for square in state.pieces(piece, chess.BLACK):
                        idx = np.unravel_index(square, (8, 8))
                        board3d[piece_to_index[piece] + 2][7 - idx[0]][idx[1]] = 1

                        if piece == chess.PAWN:
                            if isPassedPawn(state, square):
                                column_index = chess.square_file(square)
                                board3d[9][:, column_index] = 1

                    aux = state.turn
                    state.turn = chess.WHITE
                    for move in state.legal_moves:
                        if state.is_capture(move):
                            column_index = move.from_square % 8
                            board3d[6][:, column_index] = 1

                        if state.piece_at(move.from_square).piece_type == chess.KING or state.is_capture(move):
                            i, j = square_to_index(move.to_square)
                            board3d[4][i][j] = 1
                    state.turn = chess.BLACK
                    for move in state.legal_moves:
                        if state.is_capture(move):
                            column_index = move.from_square % 8
                            board3d[7][:, column_index] = 1
                        if state.piece_at(move.from_square).piece_type == chess.KING or state.is_capture(move):
                            i, j = square_to_index(move.to_square)
                            board3d[5][i][j] = 1

                    state.turn = aux


            else:
                for piece in [chess.PAWN, chess.KING]:
                    for square in state.pieces(piece, chess.BLACK):
                        idx = np.unravel_index(square, (8, 8))
                        board3d[piece_to_index[piece]][7 - idx[0]][idx[1]] = 1

                        if not passed_pawn_black and piece == chess.PAWN:
                            if isPassedPawn(state, square):
                                column_index = chess.square_file(square)
                                board3d[8][:, column_index] = 1

                    for square in state.pieces(piece, chess.WHITE):
                        idx = np.unravel_index(square, (8, 8))
                        board3d[piece_to_index[piece] + 2][7 - idx[0]][idx[1]] = 1

                        if not passed_pawn_white and piece == chess.PAWN:
                            if isPassedPawn(state, square):
                                column_index = chess.square_file(square)
                                board3d[9][:, column_index] = 1

                    aux = state.turn
                    state.turn = chess.BLACK
                    for move in state.legal_moves:
                        if state.is_capture(move):
                            column_index = move.from_square % 8
                            board3d[6][:, column_index] = 1

                        if state.piece_at(move.from_square).piece_type == chess.KING or state.is_capture(move):
                            i, j = square_to_index(move.to_square)
                            board3d[4][i][j] = 1
                    state.turn = chess.WHITE
                    for move in state.legal_moves:
                        if state.is_capture(move):
                            column_index = move.from_square % 8
                            board3d[7][:, column_index] = 1

                        if state.piece_at(move.from_square).piece_type == chess.KING or state.is_capture(move):
                            i, j = square_to_index(move.to_square)
                            board3d[5][i][j] = 1
                    state.turn = aux


                board3d = board3d[:, ::-1, ::-1].copy()
            return board3d.astype(np.float32)

        with ThreadPoolExecutor() as executor:
            encoded_states = list(executor.map(encode_state, states))

        return np.array(encoded_states)

    # Reset Board and return state
    def get_initial_state(self):
        board = chess.Board()
        board.clear_board()
        board.turn=chess.WHITE
        board.castling_rights=False
        pawnsW=[8,9,10,11,12,13,14,15]
        pawnsB=[48,49,50,51,52,53,54,55]
        for square in range(8):
            board.set_piece_at(pawnsW[square], chess.Piece(chess.PAWN, chess.WHITE))
            board.set_piece_at(pawnsB[square], chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(4, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(60, chess.Piece(chess.KING, chess.BLACK))

        print("GOT INITIAL STATE ", self.game_count)
        print(board, board.turn)
        self.game_count += 1
        return board

    # Make move on board state and return that state
    def get_next_state(self, states, actions, players):
        states = self._convert_to_list(states)
        actions = self._convert_to_list(actions)

        next_states = []
        for state, action in zip(states, actions):
            move = self.int_to_move[str(action)]
            next_state = state.copy()
            next_state.push_uci(move)
            next_states.append(next_state)


        return next_states if len(next_states) > 1 else next_states[0]

    # Get legal chess moves in position
    def get_valid_moves(self, states):
        states = self._convert_to_list(states)

        results = []
        for state in states:
            legal_moves = state.legal_moves
            move_ints = [self.move_to_int[str(move)] for move in legal_moves]

            moves = np.zeros(self.action_size, dtype=np.uint8)
            moves[move_ints] = 1

            results.append(moves)

        return results if len(results) > 1 else results[0]

    # See if win. As it's pawn wars a pawn promotion wins which can be calculated of move notation is length 5 because only pawn promotions are length 5
    def check_win(self, states, actions):
        if isinstance(states, np.ndarray):
            states = list(states)
        if not isinstance(states, list):
            states = [states]

        if isinstance(actions, np.ndarray):
            actions = list(actions)
        if not isinstance(actions, list):
            actions = [actions]

        # print(states)
        results = []
        for state, action in zip(states, actions):
            if action is None:
                results.append(False)
                continue

            outcome = state.outcome()
            move = self.int_to_move[str(action)]
            if len(move) == 5 or (outcome and outcome.winner):
                results.append(True)
            else:
                results.append(False)

        return results

    # See if board is/can be a draw
    def check_draw(self, states, actions):
        states = self._convert_to_list(states)
        actions = self._convert_to_list(actions)

        results = []
        for state, action in zip(states, actions):
            if action is None:
                results.append(False)
                continue

            outcome = state.outcome()
            if outcome and not outcome.winner or state.can_claim_draw():
                results.append(True)
            else:
                results.append(False)

        return results

    # See if win or draw and return Value, terminated values
    def get_value_and_terminated(self, states, actions):
        states = self._convert_to_list(states)
        actions = self._convert_to_list(actions)

        win_results = self.check_win(states, actions)
        draw_results = self.check_draw(states, actions)
        valid_moves_results = [np.sum(self.get_valid_moves(state)) == 0 for state in states]

        results = []
        for win, draw, valid_moves, state, action in zip(win_results, draw_results, valid_moves_results, states,
                                                         actions):
            if win:
                self.win_counts += 1
                if self.win_counts % 250 == 0:
                    print('draw', self.count_stuffs, 'wins', self.win_counts)
                results.append((1, True))
            elif valid_moves or draw:
                if self.count_stuffs % 250 == 0:
                    print('draw', self.count_stuffs, 'wins', self.win_counts)
                self.count_stuffs += 1
                results.append((0, True))
            else:
                results.append((0, False))

        return results if len(results) > 1 else results[0]

    # Get opponent
    def get_opponent(self, player):
        return -player
        
    # Get opponent value
    def get_opponent_value(self, value):
        return -value

    # Change board perspective
    def change_perspective(self, states, player):
        if not isinstance(states, (list, np.ndarray)):
            states = [states]

        new_states = []
        for state in states:
            new_state = state.copy()  # Create a copy to avoid modifying the original state
            if player == 1:
                new_state.turn = chess.WHITE
            else:
                new_state.turn = chess.BLACK
            new_states.append(new_state)

        return np.array(new_states)
