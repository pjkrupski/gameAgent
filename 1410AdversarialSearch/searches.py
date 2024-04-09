from scipy.signal import correlate2d
import numpy as np

kernel = np.array(((1, 0, 0), (1, 0, 0), (1, 1, 1)))
L_Pattern = [kernel, np.sum(kernel)]

kernel = np.array(((1, 0, 1), (0, 1, 0)))
V_Pattern = [kernel, np.sum(kernel)]

kernel = np.array(((1, 1, 1), (0, 1, 0), (0, 1, 0)))
T_Pattern = [kernel, np.sum(kernel)]


def search_L(self, board):
    #Searches for a len 5 L pattern of c's 
    return search_By_Pattern(self, board, *L_Pattern)

#TODO
def search_V(self, board):
    return search_By_Pattern(self, board, *V_Pattern)

#TODO
def search_Star(self, board):
    pass

#TODO
def search_T(self, board):
#Searches for a len 5 L pattern of c's 
    kernel = np.array(((1, 1, 1), (0, 1, 0), (0, 0, 1)))
    return search_By_Pattern(self, board, *T_Pattern)
    

def search_By_Pattern(self, board, kernel, size):
    x = correlate2d(board, kernel, mode="valid")
    if -size in x:
        return -1
    if size in x:
        return 1
    return 0


def search_Line(self, board):
#Searches for a len(board) line of c's 
   #transposition to check rows, then columns
    for TBoard in [board, np.transpose(board)]:
        # Check all the rows
        for row in TBoard:
            if len(set(row)) == 1:
                return row[0]
    
    # Check all the Diagonals
    if len(set([board[i][i] for i in range(len(board))])) == 1:
        return board[0][0]
    if len(set([board[i][len(board)-i-1] for i in range(len(board))])) == 1:
        return board[0][len(board)-1]
    return 0

    


def _all_same(cell_list, c):
    """
    Given a list of cell contents, e.g. ['x', ' ', 'X'],
    returns [1.0, 0.0] if they're all X, [0.0, 1.0] if they're all O,
    and False otherwise.
    """
    lst = [cell == c for cell in cell_list]
    if all(lst):
        return True

    return False
