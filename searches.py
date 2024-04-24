from scipy.signal import correlate2d
from scipy.ndimage import rotate
import numpy as np

kernel = np.array(((1, 0), (1, 0), (1, 1)))
kernel2 = np.array(((0, 1), (0, 1), (1, 1)))
L_Pattern = [[kernel, kernel2], [np.sum(kernel), np.sum(kernel2)]]

kernel = np.array(((1, 0, 1), (0, 1, 0)))
V_Pattern = [[kernel], [np.sum(kernel)]]

kernel = np.array(((1, 1, 1), (0, 1, 0), (0, 1, 0)))
T_Pattern = [[kernel], [np.sum(kernel)]]


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
    

def search_By_Pattern(self, board, kernels, sizes):
    for i, kernel in enumerate(kernels):
        for x in range(4):
            x = correlate2d(board, kernel, mode="valid")
            if -sizes[i] in x:
                return -1
            if sizes[i] in x:
                return 1
            kernel = rotate(kernel, 90)
    return 0


def search_Line(self, board):
   
    diagonal1 = [board[i][i] for i in range(self._dim)]
    if _all_same(diagonal1, -1) or _all_same(diagonal1, 1): 
        return diagonal1[0]

    diagonal2 = [board[i][self._dim - 1 - i] for i in range(self._dim)]
    if _all_same(diagonal2, -1) or _all_same(diagonal2, 1):
        return diagonal2[0]

    for row in board:
        if _all_same(row, -1) or _all_same(row, 1):
            return row[0]

    for c in range(self._dim):
        col = [board[r][c] for r in range(self._dim)]
        if _all_same(col, -1) or _all_same(col, 1):
            return col[0]

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
