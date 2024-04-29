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


def search_L(self, board, winlength):
    #Searches for a len 5 L pattern of c's 
    return search_By_Pattern(self, board, *L_Pattern)

#TODO
def search_V(self, board, winlength):
    return search_By_Pattern(self, board, *V_Pattern)

#TODO
def search_Star(self, board, winlength):
    pass

#TODO
def search_T(self, board, winlength):
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

def search_Line(self, board, winlength):
    # Check rows
    for row in board:
        if self.helper(row, winlength, -1) or self.helper(row, winlength, 1):
            return row[0]

    # Check columns
    for c in range(self._dim):
        col = [board[r][c] for r in range(self._dim)]
        if self.helper(col, winlength, -1) or self.helper(col, winlength, 1):
            return col[0]

    # Check diagonals
    for k in range(self._dim - winlength + 1):
        if k == 0:
            diag = np.diag(board)
        else:
            diag = np.diag(board, k)
        if self.helper(diag, winlength, -1) or self.helper(diag, winlength, 1):
            return diag[0]

        diag = np.diag(board,-k)
        if self.helper(diag, winlength, -1) or self.helper(diag, winlength, 1):
            return diag[0]

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

def helper(dim, winlength, char):
    count = 0
    for i in dim:
        if count == winlength:
            return True
        if dim[i] == char:
            count += 1
        else:
            count = 0
    return False