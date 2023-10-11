import time

import chess
import json
import numpy as np
import pandas as pd
import random

class ChessEPW:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 1968 # Don't Change



        with open('move_to_int.json', 'r') as json_file:
            self.move_to_int = json.load(json_file)

        with open('int_to_move.json', 'r') as json_file:
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
        return "ChessPawnWarsStockvStockNOPREV"


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

        return encode_state(states[0])

    #ben function
    # Reset Board and return state
    def get_initial_state(self):
        board = chess.Board()
        board.clear_board()
        board.turn=chess.WHITE
        board.castling_rights=False
        pawnsW=[8,9,10,11,12,13,14,15]
        pawnsB=[48,49,50,51,52,53,54,55]
        # pawnsW=[16,26,22,11,12,13,14,15]
        # pawnsB=[48,42,50,51,32,53,54,55]
        for square in range(8):
            board.set_piece_at(pawnsW[square], chess.Piece(chess.PAWN, chess.WHITE))
            board.set_piece_at(pawnsB[square], chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(4, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(60, chess.Piece(chess.KING, chess.BLACK))

        print("GOT INITIAL STATE ", self.game_count)
        print(board, board.turn)
        self.game_count += 1
        return board

    def get_next_state(self, state, action, player):
        actual_move = self.int_to_move[str(action)]
        # print(state, action, actual_move)
        state.push_uci(actual_move)
        return state

    def get_valid_moves(self, state):
        legal_moves = state.legal_moves
        move_ints = []
        for move in legal_moves:
            move_ints.append(self.move_to_int[str(move)])

        moves = np.array([], dtype=np.uint8)
        for i in range(self.action_size):  # Start from 0
            if i in move_ints:
                moves = np.append(moves, 1)
            else:
                moves = np.append(moves, 0)

        # print(moves)
        return moves


    def get_valid_moves_weighted(self, state):
        legal_moves = state.legal_moves
        move_ints = []
        for move in legal_moves:
            move_ints.append(self.move_to_int[str(move)])

        moves = np.array([], dtype=np.uint8)
        for i in range(self.action_size):  # Start from 0
            if i in move_ints:
                moves = np.append(moves, 1)
            else:
                moves = np.append(moves, 0)

        # print(moves)
        return moves

    def check_win(self, state, action):
        if action == None:
            return False

        outcome = state.outcome()
        move = self.int_to_move[str(action)]
        if len(move)==5:
            return True

        if outcome and outcome.winner:
            return True

        return False

    def check_draw(self, state, action):
        if action == None:
            return False

        outcome = state.outcome()

        if outcome and not outcome.winner or state.can_claim_draw():
            return True

        # if state.can_claim_draw():
        #     print("22222222222222222DRAWWWWWWW")
        #     return 12940129049012409
        #     return True

        return False


    def get_postion_eval(self, state, action):
        white_value = 0
        black_value = 0

        # Iterate over all squares on the board
        for square in chess.SQUARES:
            piece = state.piece_at(square)
            if piece:
                # Add the value of the piece to the respective total (white or black)
                if piece.color:
                    white_value += self.PIECE_VALUES[piece.piece_type]
                else:
                    black_value += self.PIECE_VALUES[piece.piece_type]

        # if black_value != white_value:
        #     pass
        #     print(white_value - black_value)
        if not state.turn:
            evaluation = (white_value - black_value) / (white_value + black_value) * .05
            # if evaluation < .2:
            #     evaluation = .2
            # if evaluation > .8:
            #     evaluation = .8
            return evaluation
        else:
            evaluation = (black_value - white_value) / (black_value + white_value) * .05
            # if evaluation < .2:
            #     evaluation = .2
            # if evaluation > .8:
            #     evaluation = .8
            return evaluation

    def get_value_and_terminated(self, state, action):
        # print(self.check_win(state, action), 'IS END')


        if self.check_win(state, action):
            # print("WIIIIIIIIIIIIIIIIN")
            # print("CUUREVAL", 1, self.int_to_move[str(action)], state.turn)
            self.win_counts += 1

            # time.sleep(10)
            if self.win_counts % 250 == 0:
                print('draw', self.count_stuffs, 'wins', self.win_counts)
                # print('WIN')
                # print(state)
            # if state.turn:
            #     print(self.count)
            #     dff = pd.read_csv('sssss')
            #     return 1, True, 1
            # self.count += 1
            return 1, True

        if np.sum(self.get_valid_moves(state)) == 0 or self.check_draw(state, action):
            # print("DRAWWWWW")
            # print("CUUREVAL", 0, self.int_to_move[str(action)], state.turn)
            # print(state)
            # print(state, 'draw')
            if self.count_stuffs % 250 == 0:
                print('draw', self.count_stuffs, 'wins', self.win_counts)
                # print("DRAW")
                # print(state)
            self.count_stuffs += 1

            # Maybe set to 0
            return 0, True
        # print("CCCCCCCCCCCCCCCC")

        # cur_eval = self.get_postion_eval(state, action)
        # if cur_eval < .5 or cur_eval > .5:
        #     pass
            # print("NORMAL")
            # print("CUUREVAL", cur_eval, self.int_to_move[str(action)], state.turn)
            # print(state)
        # print(state)
        return 0, False
        # return cur_eval, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        if player == 1:
            state.turn = chess.WHITE
        else:
            state.turn = chess.BLACK
        return state

#chess_new = ChessEQK()
#state = chess_new.get_initial_state()


# chess_new = ChessEPW()
# player = 1
#
# state = chess_new.get_initial_state()
#
# while True:
#     print(state)
#     valid_moves = chess_new.get_valid_moves(state)
#
#     # print(valid_moves)
#     print("valid_moves", [i for i in range(chess_new.action_size) if valid_moves[i] == 1])
#     valid_moves_compare = [i for i in range(chess_new.action_size) if valid_moves[i] == 1]
#     print(state)
#     moves_str = []
#     for v in valid_moves_compare:
#         moves_str.append(chess_new.int_to_move[str(v)])
#     print(moves_str)
#     action = str(input(f"{player}:"))
#     action = chess_new.move_to_int[action]
#     if action not in valid_moves_compare:
#         print("action not valid")
#         continue
#
#     state = chess_new.get_next_state(state, action, player)
#     chess_new.change_perspective(state, player)
#
#     print(chess_new.get_encoded_state(state))
#
#     value, is_terminal = chess_new.get_value_and_terminated(state, action)
#
#     if is_terminal:
#         print(state)
#         if value == 1:
#             print(player, "won")
#         else:
#             print("draw")
#         break
#
#     player = chess_new.get_opponent(player)