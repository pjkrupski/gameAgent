# author Vincent Kubala; GameUI implementation by Eli Zucker

###############################################################################
# State and Action Representations:
#
# - A (game) state is a TTTState that contains game board and the index of the
#   player to move
#
# - An action is a pair that represents the location at which a player
#   would like to draw an X or an O: (<index of row>, <index of column>),
#   where the indices start at 0.
#
# - See the main function at the bottom of this file for an example.
#
###############################################################################


from typing import Tuple
from adversarialsearchproblem import AdversarialSearchProblem, GameState, GameUI
from searches import search_L, search_Line, search_V, search_Star, search_T
import time
import numpy as np 
import heuristicsearch as heuristic
from math import tanh


SPACE = 0
X = 1  # Player 0 is X
O = -1  # Player 1 is O
PLAYER_SYMBOLS = [X, O]


class TTTState:
    def __init__(self, board, ptm):
        """
        Inputs:
                board - represented as a 2D List of character strings.
                Each character in the board is X, O, or SPACE (see above
                for global definition), where SPACE indicates that the
                corresponding cell of the tic-tac-toe board is empty.

                ptm- the index of the player to move, which will be 0 or 1,
                where 0 corresponds to the X player, who moves first, and
                1 to the O player, who moves second.
        """
        self.board = board
        self.ptm = ptm

    def player_to_move(self):
        return self.ptm


# In TTT, an action consists of placing a piece on a 2D grid. Thus, our actions need two pieces of
# data: both row and column. So the type of our action is tuple with two ints.
Action = Tuple[int, int]


class TTTProblem(AdversarialSearchProblem[TTTState, Action]):
    def __init__(self, dim, pattern="line", winlength=3, board=None, player_to_move=0):
        """
        Inputs:
                dim- the number of cells in one row or column.
                board - 2d list of character strings (as in TTTState)
                player_to_move- index of player to move (as in TTTState).

                The board and player_to_move together constitute the start state
                of the game
        """
        self._dim = dim
        self._winlength = winlength
        if board == None:
            board = np.full(shape=(dim,dim), fill_value=SPACE)
        self._start_state = TTTState(board, player_to_move)
        self.pattern = pattern
        searches = {'l': search_L, 'line': search_Line, 'v': search_V, 'star': search_Star, 't': search_T}
        self.search_win = searches[pattern]
        searches = {'l': heuristic.search_L, 'line': heuristic.search_Line, 'v': heuristic.search_V, 'star': heuristic.search_Star, 't': heuristic.search_T}
        self.search_heuristic = searches[pattern]

    def heuristic_func(self, state: TTTState, player_index: int) -> float:
        """
        TODO: Fill this out with your own heuristic function! You should make sure that this
        function works with boards of any size; if it only works for 3x3 boards, you won't be
        able to properly test ab-cutoff for larger board sizes!
        """
        dict = {0:X, 1:O}
        possibilities = self.search_heuristic(state.board, self._winlength)
        x = possibilities[dict[player_index]] - possibilities[dict[player_index]]
        x = tanh(x/50)/4 + .75
        # if x <= .5 or x >= 1:
        #     print(x)
        return x

    def get_available_actions(self, state):
        actions = set()
        # for r in range(self._dim):
        #     for c in range(self._dim):
        #         if state.board[r][c] == SPACE:
        #             actions.add((r, c))
        
        for i in np.argwhere(state.board == SPACE):
            actions.add((i[0], i[1]))
        
        
        return actions

    def transition(self, state, action):
        assert not (self.is_terminal_state(state))
        assert action in self.get_available_actions(state)

        # make deep copy of board
        board = np.copy(state.board)

        board[action[0]][action[1]] = PLAYER_SYMBOLS[state.ptm]
        return TTTState(board, 1 - state.ptm)

    def is_terminal_state(self, state):
        return not (self.internal_evaluate_terminal(state) == "non-terminal")
            

    def evaluate_terminal(self, state):
        internal_val = self.internal_evaluate_terminal(state)
        if internal_val == "non-terminal":
            raise ValueError("attempting to evaluate a non-terminal state")
        else:
            return internal_val
        
    def internal_evaluate_terminal(self, state):
        """
        If state is terminal, returns its evaluation;
        otherwise, returns 'non-terminal'.
        """
        board = state.board
        # rows = self._dim
        # cols = self._dim
        # 
        # for row in range(rows):
        #     for col in range(cols):
        #         if board[row][col] == X:
        #             if self.search_win(self, X, board, row, col):
        #                 return [1.0, 0.0]
        #         elif board[row][col] == O:
        #             if self.search_win(self, O, board, row, col):
        #                 return [0.0, 1.0]
        x = self.search_win(board, winlength=self._winlength)
        if x == X:
            return [1.0, 0.0]
        elif x == O:
            return [0.0, 1.0]
        elif self.get_available_actions(state) == set():
            # all spaces are filled up
            return [0.5, 0.5]
        else:
            return "non-terminal"
   
   
    @staticmethod
    def board_to_pretty_string(board):
        """
        Takes in a tile game board and outputs a pretty string representation
        of it for printing.
        """
        hbar = "-"
        vbar = "|"
        corner = "+"
        dim = len(board)
        valToString = [" ", "X", "O"]

        s = corner
        for _ in range(2 * dim - 1):
            s += hbar
        s += corner + "\n"

        for r in range(dim):
            s += vbar
            for c in range(dim):
                s += valToString[board[r][c]] + " "
            s = s[:-1]
            s += vbar
            s += "\n"

        s += corner
        for _ in range(2 * dim - 1):
            s += hbar
        s += corner
        return s


# Basic TTT GameUI implementation (prints board states to console)
class TTTUI(GameUI):
    def __init__(self, asp: TTTProblem, delay=0.2):
        self._asp = asp
        self._delay = delay
        self._state = TTTState(
            [[SPACE for _ in range(asp._dim)] for _ in range(asp._dim)], 0
        )  # empty state

    def render(self):
        print(TTTProblem.board_to_pretty_string(self._state.board))
        time.sleep(self._delay)

    def get_user_input_action(self):
        """
        Output- Returns an action obtained through the GameUI input itself.
        """
        user_action = None
        available_actions = self._asp.get_available_actions(self._state)

        #while not user_action in available_actions:
        #    row = int(input("Enter row index: "))
        #    col = int(input("Enter column index: "))
        #    user_action = (row, col)
        while user_action not in available_actions:
            user_input = input("Enter row and column indices separated by a space eg, 3 3:\n")
            row, col = map(int, user_input.split())  # Split input string and convert to integers
            user_action = (row, col)
        return user_action


def main():
    """
    Provides an example of the TTTProblem class being used.
    """
    t = TTTProblem()
    # A state in which an X is in the center cell, and O moves next.
    s0 = TTTState([[" ", " ", " "], [" ", "X", " "], [" ", " ", " "]], 1)
    # The O player puts down an O at the top-left corner, and now X moves next
    s1 = t.transition(s0, (0, 0))
    assert s1.board == [["O", " ", " "], [" ", "X", " "], [" ", " ", " "]]
    assert s1.ptm == 0


if __name__ == "__main__":
    main()
