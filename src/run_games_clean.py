import pickle
import pygame
import chess
import random
import time
from RsNet import ResNet
from MCTS import MCTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ChessEnvPawn import ChessEPW

# Initialize the pygame
pygame.init()

# Set the dimensions of the window
window_size = (800, 800)
window = pygame.display.set_mode(window_size)

# Load chess piece images
pieces = {}
for color in ['w', 'b']:
    for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
        image = pygame.image.load(f'./resources/{color + piece}.png')
        pieces[color + piece] = pygame.transform.scale(image, (100, 100))

# Define colors
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (0, 255, 0)


def draw_board(window, board):
    window.fill(LIGHT_GREEN)
    sq_size = 100  # size of each square on the chess board
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                pygame.draw.rect(window, LIGHT_GREEN, pygame.Rect(col * sq_size, row * sq_size, sq_size, sq_size))
            else:
                pygame.draw.rect(window, DARK_GREEN, pygame.Rect(col * sq_size, row * sq_size, sq_size, sq_size))

            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image_key = piece.symbol().upper()
                if piece.color:
                    piece_image_key = 'w' + piece_image_key
                else:
                    piece_image_key = 'b' + piece_image_key
                window.blit(pieces[piece_image_key], pygame.Rect(col * sq_size, row * sq_size, sq_size, sq_size))

def run_vs_self():
    game_moves = []

    vgame = ChessEPW()

    player = 1

    args = {
        'C': 1,
        'num_searches': 50,
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.1
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(vgame, 10, 256, device)
    model.load_state_dict(torch.load("./Models/model_20_ChessPawnWarsParallel7withThreads1000.pt", map_location=device))
    model.eval()
    mcts = MCTS(vgame, args, model)

    state = vgame.get_initial_state()
    neutral_state = state
    board = vgame.get_initial_state()

    # Run the game loop
    game_ended = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw the board with pieces
        draw_board(window, board)
        pygame.display.flip()

        if not game_ended:
            time.sleep(0)

            # Automated move logic
            if board.legal_moves:
                mcts_probs = mcts.search(neutral_state)
                action = np.argmax(mcts_probs)
                value, is_terminal = vgame.get_value_and_terminated(state, action)

                # if is_terminal and value == 0:
                #     second_best_idx = mcts_probs.argsort()[-2]  # Get second highest value's index
                #     action = second_best_idx

                action_str = vgame.int_to_move[str(action)]
                game_moves.append(action_str)
                # print(action_str)
                board.push_san(action_str)
                state = vgame.get_next_state(state, action, player)
            else:
                game_ended = True

            # Draw the board with pieces
            draw_board(window, board)
            pygame.display.flip()

            # Check if the game has ended
            if vgame.get_value_and_terminated(state, action)[1]:
                game_ended = True
                print("Game Over! Closing in 5 seconds...")
                pygame.time.wait(5000)
                running = False

            player = vgame.get_opponent(player)  # Alternate turns
            # print("here")
            neutral_state = vgame.change_perspective(state, player)

        # Draw the board with pieces at the end of the game
        if game_ended:
            with open('game_moves.pkl', 'wb') as f:
                pickle.dump(game_moves, f)
            draw_board(window, board)
            pygame.display.flip()
            pygame.time.wait(5000)  # Additional delay to allow the last move to be displayed



    # Quit pygame
    pygame.quit()


def load_and_navigate():
    pygame.init()

    with open('game_moves.pkl', 'rb') as f:
        saved_moves = pickle.load(f)
    vgame = ChessEPW()
    index = 0
    board = vgame.get_initial_state()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:  # arrow right
                    if index < len(saved_moves):
                        board.push_san(saved_moves[index])
                        index += 1
                if event.key == pygame.K_LEFT:  # arrow left
                    if index > 0:
                        board.pop()  # undo last move
                        index -= 1

        draw_board(window, board)
        pygame.display.flip()

    pygame.quit()


# run_vs_self()
load_and_navigate()
