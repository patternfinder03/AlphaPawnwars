import imageio
import pygame
import pickle
import os
from envs.ChessEnvPawn import ChessEPW
import chess
import time

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

def convert_to_gif():
    pygame.init()

    with open('data/game_moves.pkl', 'rb') as f:
        saved_moves = pickle.load(f)
    vgame = ChessEPW()
    board = vgame.get_initial_state()

    images = []  # List to store each frame's image data

    for move in saved_moves:
        board.push_san(move)
        draw_board(window, board)
        pygame.display.flip()

        # Introduce a delay
        time.sleep(0.5)  # You can adjust this value

        # Pump the event queue to keep window responsive
        pygame.event.pump()

        # Capture the current state as an image and append to images list
        pygame.image.save(window, "temp_image.png")
        images.append(imageio.imread("temp_image.png"))

    # Create GIF from images
    imageio.mimsave('game_replay.gif', images, duration=10000)

    pygame.quit()

    # Optionally, remove the temporary image
    os.remove("temp_image.png")



if __name__ == "__main__":
    convert_to_gif()