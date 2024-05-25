# AlphaPawnwars

Fitting the AlphaZero model with some tweaks to the chess variant game Pawn Wars.

## Results

### First Decent Results
- **White's Strategy**: Induced to play more king moves (2000 games).
- **Black's Strategy**: Pure self-play (2000 games).

![First Results](https://github.com/lordyabu/AlphaPawnwars/assets/92772420/d9b3d3e6-724b-4689-9b17-a539ae6b5eff)

### Second Decent Results
- **Extended Training**: Built upon the initial White model with an additional 8000 self-play games, resulting in a total of 10000 games.
- **Emergent Strategy**: The model employs the "Fox in the Chicken Coop" strategy.

![Second Results](https://github.com/lordyabu/AlphaPawnwars/assets/92772420/5378b95f-1ef3-47f6-8efa-67bbfa80b856)

## Overview

AlphaPawnwars is an implementation of the AlphaZero algorithm tailored to the Pawn Wars variant of chess. The project explores various techniques to improve the model's performance, including incentivized self-play and competitive games against Stockfish.

## Technical Details

### State Representation

The state of the game is represented using a 3D array with the dimensions (10, 8, 8). Each plane in the array captures specific information about the board configuration and potential moves.

- **Dimensions**: The state is encoded into a 3D array of size (10, 8, 8).
  - The first dimension (10) represents different channels or planes.
  - The second and third dimensions (8, 8) represent the chessboard with 8 rows and 8 columns.

#### Planes Description

1. **White Pawns**: Plane 0
   - Represents the positions of white pawns on the board.
2. **White Kings**: Plane 1
   - Represents the positions of white kings on the board.
3. **Black Pawns**: Plane 2
   - Represents the positions of black pawns on the board.
4. **Black Kings**: Plane 3
   - Represents the positions of black kings on the board.
5. **White King Moves and Captures**: Plane 4
   - Indicates squares to which the white king can move or capture.
6. **Black King Moves and Captures**: Plane 5
   - Indicates squares to which the black king can move or capture.
7. **White Potential Captures**: Plane 6
   - Marks columns where white pieces can potentially capture.
8. **Black Potential Captures**: Plane 7
   - Marks columns where black pieces can potentially capture.
9. **Passed Pawns (White)**: Plane 8
   - Indicates columns with passed white pawns.
10. **Passed Pawns (Black)**: Plane 9
   - Indicates columns with passed black pawns.

### Move Probability Changes

The move probability adjustment is designed to incentivize certain strategic behaviors based on the state of the game. Here are the key aspects of these adjustments:

#### King Movement

- **Direction-Based Multiplier**:
  - **Forward Movement**: Highest priority with a multiplier of 5.
  - **Sideways Movement**: Medium priority with a multiplier of 3.
  - **Backward Movement**: Lowest priority with a multiplier of 2.

- **Player Perspective**:
  - The multipliers are adjusted based on whether the player is White or Black to account for the different perspectives.

#### Number of Pawns

The number of pawns on the board further influences the multipliers:

- **Fewer or Equal to 4 Pawns**: Multiplier is reduced to 0.25.
- **Between 5 to 6 Pawns**: Multiplier is reduced to 0.5.
- **Between 7 to 8 Pawns**: Multiplier is reduced to 0.66.

### Example Adjustments

Here is an example of how the move probabilities are adjusted:

- **If the White King moves forward and there are 3 pawns**:
  - Base multiplier: 5
  - Adjustment for pawns: 0.25
  - Final multiplier: 5 * 0.25 = 1.25

- **If the Black King moves sideways and there are 6 pawns**:
  - Base multiplier: 3
  - Adjustment for pawns: 0.5
  - Final multiplier: 3 * 0.5 = 1.5

