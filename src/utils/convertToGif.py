import imageio
import pygame
import pickle
import os
from envs.ChessEnvPawn import ChessEPW
import chess
import math
import time

DARK_GRAY = (169, 169, 169)
LIGHT_GRAY = (211, 211, 211)
WINDOW_SIZE = (900, 900)  # Increased size
SQUARE_SIZE = 100
BORDER_SIZE = 50  # Space for letters and numbers
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

BORDER_SIZE = 50
WINDOW_SIZE = (8 * SQUARE_SIZE + 2 * BORDER_SIZE, 8 * SQUARE_SIZE + 2 * BORDER_SIZE)


def draw_arrow(surface, color, start, end):
    pygame.draw.line(surface, color, start, end, 3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_length = 20  # Increased the size
    arrow_width = 15  # Increased the size
    left_point = (end[0] - arrow_length * math.cos(angle - math.pi / 6),
                  end[1] - arrow_length * math.sin(angle - math.pi / 6))
    right_point = (end[0] - arrow_length * math.cos(angle + math.pi / 6),
                   end[1] - arrow_length * math.sin(angle + math.pi / 6))
    pygame.draw.polygon(surface, color, [end, left_point, right_point])

def draw_board(window, board):
    window.fill((255, 255, 255))  # Fill the entire window with white color

    for row in range(8):
        for col in range(8):
            color = LIGHT_GRAY if (row + col) % 2 == 0 else DARK_GRAY
            pygame.draw.rect(window, color,
                             pygame.Rect(BORDER_SIZE + col * SQUARE_SIZE, BORDER_SIZE + row * SQUARE_SIZE, SQUARE_SIZE,
                                         SQUARE_SIZE))

            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                window.blit(pieces[piece_image_key],
                            pygame.Rect(BORDER_SIZE + col * SQUARE_SIZE, BORDER_SIZE + row * SQUARE_SIZE, SQUARE_SIZE,
                                        SQUARE_SIZE))

    # Add numbers and letters
    font = pygame.font.SysFont(None, 36)
    for row in range(8):
        label = font.render(str(8 - row), True, (0, 0, 0))
        window.blit(label, (BORDER_SIZE / 2 - label.get_width() / 2,
                            BORDER_SIZE + row * SQUARE_SIZE + SQUARE_SIZE / 2 - label.get_height() / 2))

    for col in range(8):
        label = font.render(chr(97 + col), True, (0, 0, 0))
        window.blit(label, (BORDER_SIZE + col * SQUARE_SIZE + SQUARE_SIZE / 2 - label.get_width() / 2,
                            WINDOW_SIZE[1] - BORDER_SIZE / 2 - label.get_height() / 2))

    if board.move_stack:
        last_move = board.peek()  # Get the last move
        start_x = BORDER_SIZE + (chess.square_file(last_move.from_square) + 0.5) * SQUARE_SIZE
        start_y = BORDER_SIZE + (7 - chess.square_rank(last_move.from_square) + 0.5) * SQUARE_SIZE
        end_x = BORDER_SIZE + (chess.square_file(last_move.to_square) + 0.5) * SQUARE_SIZE
        end_y = BORDER_SIZE + (7 - chess.square_rank(last_move.to_square) + 0.5) * SQUARE_SIZE
        draw_arrow(window, (128, 128, 128), (start_x, start_y), (end_x, end_y))


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
    imageio.mimsave('game_replay.gif', images, duration=1500)

    pygame.quit()

    # Optionally, remove the temporary image
    os.remove("temp_image.png")



if __name__ == "__main__":
    convert_to_gif()