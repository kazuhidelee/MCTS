# Monte Carlo Tree Search (MCTS) Algorithm

### Overview
In this assignment, I have implemented the Monte Carlo Tree Search (MCTS) component of AlphaZero for the game Othello.

### Setup
- The assignment was completed and ran on **Google Colab notebook**.

**To try it youself:**
- Make a copy of the Colab notebook before starting to code (`File > Save a copy in Drive`).

### Othello
Othello is a strategy board game for two players (Black and White), played on an 8 by 8 board. The game traditionally begins with four discs placed in the middle of the board as shown below. Black moves first.

<img width="312" alt="Screenshot 2025-02-27 at 10 37 00 AM" src="https://github.com/user-attachments/assets/3a867e37-4c02-4bc2-945c-09984074c0ed" />

Black must place a black disc on the board, in such a way that there is at least one straight (horizontal, vertical, or diagonal) occupied line between the new disc and another black disc, with one or more contiguous white pieces between them. In the starting position, Black has the following 4 options indicated by translucent discs:

<img width="311" alt="Screenshot 2025-02-27 at 10 37 34 AM" src="https://github.com/user-attachments/assets/60fdc93c-bfeb-4871-b2f8-df508b00be9c" />

After placing the disc, Black flips all white discs lying on a straight line between the new disc and any existing black discs. All flipped discs are now black. If Black decides to place a disc in the topmost location, one white disc gets flipped, and the board now looks like this:

<img width="311" alt="Screenshot 2025-02-27 at 10 37 55 AM" src="https://github.com/user-attachments/assets/f229315c-c3d4-4324-9783-5b7e382c96dc" />

Now White plays. This player operates under the same rules, with the roles reversed: White lays down a white disc, causing black discs to flip. Possibilities at this time would be:

<img width="310" alt="Screenshot 2025-02-27 at 10 38 10 AM" src="https://github.com/user-attachments/assets/89e94d36-3bab-4383-85fb-9465f90d58f5" />

If White plays the bottom left option and flips one disc:

<img width="312" alt="Screenshot 2025-02-27 at 10 38 26 AM" src="https://github.com/user-attachments/assets/ee31bc75-dffa-464d-a67d-859abc0d9df4" />

Players alternate taking turns. If a player does not have any valid moves, play passes back to the other player. When neither player can move, the game ends. A game of Othello may end before the board is completely filled.

The player with the most discs on the board at the end of the game wins. If both players have the same number of discs, then the game is a draw.

### AlphaZero Algorithm (High-Level)
1. Run **`numIters`** iterations, each with **`numEps`** episodes.
2. Use **MCTS** to determine actions until reaching a terminal state.
3. Generate training samples from searches.
4. Train the neural network using these samples.
5. Compare the new neural network against the old one **`arenaCompare`** times.
6. Replace the old network if the new one performs better.
7. Repeat the process.

### Code Structure
#### `MCTS.ipynb`
- Main file for running and testing MCTS.
- Contains the **Coach class** which trains the model.
- Calls `MCTsearch()` indirectly through `executeEpisode()`.

#### `MCTS class`
Funcitons in MCTS class
- `select(self, state, board)`: Selection phase of MCTS.
- `simulate(self, state, board)`: Simulation phase using a neural network.
- `backpropagate(self, seq)`: Backpropagation phase.
- `MCTsearch(self, initial_board)`: Main MCTS function.

