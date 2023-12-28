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


def player_move_logic(board, player_color):
    running = True
    move_made = False
    move = None  # Initialize move as None
    start_square = None  # Keep track of the starting square
    end_game = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not start_square:
                    # Get the start square on the first click
                    start_square = (event.pos[1] // SQUARE_SIZE, event.pos[0] // SQUARE_SIZE)
                    print(f"Selected start square: {start_square}")
                else:
                    # Get the end square on the second click
                    end_square = (event.pos[1] // SQUARE_SIZE, event.pos[0] // SQUARE_SIZE)

                    # Adjust for player color
                    if player_color == -1:  # if player is white
                        move = chess.Move(chess.square(start_square[1], 7 - start_square[0]),
                                          chess.square(end_square[1], 7 - end_square[0]))
                    else:  # if player is black
                        move = chess.Move(chess.square(7 - start_square[1], start_square[0]),
                                          chess.square(7 - end_square[1], end_square[0]))
                    print(f"Selected end square: {end_square}")

                    piece = board.piece_at(move.from_square)

                    if move in board.legal_moves:
                        board.push(move)
                        move_made = True
                        running = False
                    else:
                        # If the move is not legal, reset the start_square and wait for another pair of clicks
                        start_square = None

                    if end_square[0] == 0 and str(piece).upper() == "P":
                        end_game = True

    return move_made, move, end_game


def draw_board(window, board, player_color):
    window.fill(LIGHT_GREEN)

    for row in range(8):
        for col in range(8):
            color = LIGHT_GREEN if (row + col) % 2 == 0 else DARK_GREEN
            pygame.draw.rect(window, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            # Adjust the piece_square calculation based on player's color
            if player_color == -1:  # if player is white
                piece_square = chess.square(col, 7 - row)
            else:  # if player is black
                piece_square = chess.square(7 - col, row)

            piece = board.piece_at(piece_square)
            if piece:
                piece_image_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                window.blit(pieces[piece_image_key],
                            pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def run_vs_self(model1_path, model2_path):
    game_moves = []
    vgame = ChessEPW()
    player = 1

    args = {
        'C': 1,
        'num_searches': 50,
        'dirichlet_epsilon': 0,
        'dirichlet_alpha': 0.1
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [ResNet(vgame, 5, 256, device) for _ in range(2)]
    models[0].load_state_dict(torch.load(model1_path, map_location=device))
    models[1].load_state_dict(torch.load(model2_path, map_location=device))

    for model in models:
        model.eval()

    mcts = [MCTS(vgame, args, model) for model in models]

    state = vgame.get_initial_state()
    board = state.copy()

    running, game_ended = True, False
    while running:
        # handle_pygame_events()

        draw_board_and_update(window, board)

        if not game_ended:
            game_ended = automated_move_logic(board, state, mcts[player - 1], vgame, game_moves, player)

            player = vgame.get_opponent(player)

    print("SAVING")
    save_game_moves(game_moves)
    pygame.quit()


def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()


def draw_board_and_update(window, board, player_color=1):
    draw_board(window, board, player_color)
    pygame.display.flip()




def automated_move_logic(board, state, current_mcts, vgame, game_moves, player):
    game_ended = False
    if board.legal_moves:
        mcts_probs = current_mcts.search_biased(state)
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
            save_game_moves(game_moves)
            exit()

    return game_ended


def save_game_moves(game_moves):
    with open('./data/game_moves.pkl', 'wb') as f:
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

        draw_board(window, board, player_color=1)
        pygame.display.flip()

    pygame.quit()


def run_vs_computer(model_path, player=-1):
    game_moves = []
    vgame = ChessEPW()
    player = player
    player_color = player
    flip = True if player_color == 1 else False

    args = {
        'C': 1,
        'num_searches': 50,
        'dirichlet_epsilon': 0,
        'dirichlet_alpha': 0.1
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(vgame, 5, 256, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mcts = MCTS(vgame, args, model)

    state = vgame.get_initial_state()
    board = state.copy()

    running, game_ended = True, False
    while running:
        handle_pygame_events()
        draw_board_and_update(window, board, player_color)

        if not game_ended:
            if player == 1:  # Computer's turn
                game_ended = automated_move_logic(board, state, mcts, vgame, game_moves, player)
            else:  # User's turn
                move_made, move, game_ended = player_move_logic(board, player_color)
                if move_made:
                    move = vgame.move_to_int[str(move)]
                    state = vgame.get_next_state(state, move, player)
                else:
                    continue
            player = vgame.get_opponent(player)

    save_game_moves(game_moves)
    pygame.quit()



def main():
    parser = argparse.ArgumentParser(description="Chess Simulation.")
    parser.add_argument("mode", type=str, choices=["sp", "load", "play"],
                        help="Mode to run the script in. 'sp' for self-play, 'play' for user vs computer, and 'load' to load a saved game.")
    parser.add_argument("--color", type=str, choices=['white', 'black'], default='white',
                        help="Player's color. Can be 'white' or 'black'. Default is 'white'.")

    args = parser.parse_args()

    if args.mode == "sp":
        MODEL_PATH_1 = "./Models/model_8_ChessPawnWarsParallel7No-1OtherOneAccIs-1withThreads5blocksNoMul4000.pt"
        MODEL_PATH_2 = "./Models/model_8_ChessPawnWarsParallel7No-1OtherOneAccIs-1withThreads5blocksNoMul4000.pt"
        run_vs_self(MODEL_PATH_1, MODEL_PATH_2)

    if args.mode == "play":
        MODEL_PATH_1 = "./Models/model_6_ChessPawnWarsParallel7No-1OtherOneAccIs-1withThreads5blocksNoMul4000.pt"
        player_color = 1 if args.color == 'black' else -1  # Set player color based on the argument
        run_vs_computer(MODEL_PATH_1, player_color)


    elif args.mode == "load":
        load_and_navigate()


if __name__ == '__main__':
    main()