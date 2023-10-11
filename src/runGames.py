import pickle
import pygame
import chess
import torch
import numpy as np

from Resnet.resnet import ResNet
from mcts.mctsSingle import MCTS
from envs.ChessEnvPawn import ChessEPW
import argparse

# Constants
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (0, 255, 0)
WINDOW_SIZE = (800, 800)
SQUARE_SIZE = 100
RESOURCE_PATH = 'Resources/'

pygame.init()
window = pygame.display.set_mode(WINDOW_SIZE)


def load_pieces():
    pieces = {}
    for color in ['w', 'b']:
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
            image_path = RESOURCE_PATH + color + piece + '.png'
            image = pygame.image.load(image_path)
            pieces[color + piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
    return pieces


pieces = load_pieces()


def draw_board(window, board):
    window.fill(LIGHT_GREEN)
    for row in range(8):
        for col in range(8):
            color = LIGHT_GREEN if (row + col) % 2 == 0 else DARK_GREEN
            pygame.draw.rect(window, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                window.blit(pieces[piece_image_key], pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def run_vs_self(model1_path, model2_path):
    game_moves = []
    vgame = ChessEPW()
    player = 1

    args = {
        'C': 1,
        'num_searches': 100,
        'dirichlet_epsilon': 0.1,
        'dirichlet_alpha': 0.1
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [ResNet(vgame, 10, 256, device) for _ in range(2)]
    models[0].load_state_dict(torch.load(model1_path, map_location=device))
    models[1].load_state_dict(torch.load(model2_path, map_location=device))

    for model in models:
        model.eval()

    mcts = [MCTS(vgame, args, model) for model in models]

    state = vgame.get_initial_state()
    board = state.copy()

    running, game_ended = True, False
    while running:
        handle_pygame_events()

        draw_board_and_update(window, board)

        if not game_ended:
            game_ended = automated_move_logic(board, state, mcts[player - 1], vgame, game_moves, player)

            player = vgame.get_opponent(player)

    save_game_moves(game_moves)
    pygame.quit()


def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()


def draw_board_and_update(window, board):
    draw_board(window, board)
    pygame.display.flip()


def automated_move_logic(board, state, current_mcts, vgame, game_moves, player):
    game_ended = False
    if board.legal_moves:
        mcts_probs = current_mcts.search(state)
        action = np.argmax(mcts_probs)
        value, is_terminal = vgame.get_value_and_terminated(state, action)

        action_str = vgame.int_to_move[str(action)]
        game_moves.append(action_str)
        board.push_san(action_str)
        state = vgame.get_next_state(state, action, player)

        if is_terminal:
            game_ended = True
            print("Game Over! Closing in 5 seconds...")
            pygame.time.wait(5000)
            exit()

    return game_ended


def save_game_moves(game_moves):
    with open('game_moves1.pkl', 'wb') as f:
        pickle.dump(game_moves, f)


def load_and_navigate():
    pygame.init()

    with open('data/game_moves.pkl', 'rb') as f:
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


def main():
    parser = argparse.ArgumentParser(description="Chess Simulation.")
    parser.add_argument("mode", type=str, choices=["sp", "load"],
                        help="Mode to run the script in. 'sp' for self-play and 'load' to load a saved game.")

    args = parser.parse_args()

    if args.mode == "sp":
        MODEL_PATH_1 = "./Models/model_18_ChessPawnWarsParallel7withThreads.pt"
        MODEL_PATH_2 = "./Models/model_19_ChessPawnWarsParallel7withThreads.pt"
        run_vs_self(MODEL_PATH_1, MODEL_PATH_2)
    elif args.mode == "load":
        load_and_navigate()


if __name__ == '__main__':
    main()