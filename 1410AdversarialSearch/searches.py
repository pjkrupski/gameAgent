

def search_L(self, c, board, row, col):
        #Searches for a len 5 L pattern of c's 
        rows = self._dim
        cols = self._dim
        if (row - 2 <= 0 and col + 2 < cols and
            board[row-1][col] == c and
            board[row-2][col] == c and
            board[row][col+1] == c and
            board[row][col+2] == c):
            return True
        if (row + 2 < rows and col + 2 < cols and
            board[row + 1][col] == c and
            board[row + 2][col] == c and
            board[row][col+1] == c and
            board[row][col+2] == c):
            return True
        if (row - 2 <= 0 and col - 2 >= 0 and
            board[row-1][col] == c and
            board[row-2][col] == c and
            board[row][col-1] == c and
            board[row][col-2] == c):
            return True
        if (row + 2 < rows and col + 2 < cols and
            board[row + 1][col] == c and
            board[row + 2][col] == c and
            board[row][col-1] == c and
            board[row][col-2] == c):
            return True
        return False

#TODO
def search_V():
    pass

#TODO
def search_Star():
    pass

#TODO
def search_T(self, c, board, row, col):
#Searches for a len 5 L pattern of c's 
    rows = self._dim
    cols = self._dim
    if (col + 2 < cols and row + 2 < rows and
        board[row][col+1] == c and
        board[row][col+2] == c and
        board[row+1][col+1] == c and
        board[row+2][col+1] == c):
        return True
    if (row + 1 < rows and row - 1 <= 0 and col + 2 < cols and
        board[row - 1][col] == c and
        board[row + 1][col] == c and
        board[row+1][col+1] == c and
        board[row+1][col+2] == c):
        return True
    if (row - 2 <= 0 and col + 2 < cols and
        board[row][col+1] == c and
        board[row][col+2] == c and
        board[row-1][col+1] == c and
        board[row-2][col+1] == c):
        return True
    if (row + 1 < rows and row - 1 >= 0 and col + 2 < cols and
        board[row][col+1] == c and
        board[row][col+2] == c and
        board[row-1][col+1] == c and
        board[row+1][col+2] == c):
        return True
    return False

def search_Line(self, c, board, row, col):
#Searches for a len(board) line of c's 
   
    diagonal1 = [board[i][i] for i in range(self._dim)]
    
    # bind the output of _all_same for diagonal1 for its two uses
    if _all_same(diagonal1, c):  # #onlyinpython / #imissoptions
        return True

    diagonal2 = [board[i][self._dim - 1 - i] for i in range(self._dim)]
    if _all_same(diagonal2, c):
        return True

    for row in board:
        if _all_same(row, c):
            return True

    for c in range(self._dim):
        # why oh why didn't I just use numpy arrays?
        col = [board[r][c] for r in range(self._dim)]
        if _all_same(col, c):
            return True
    return False

    


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
