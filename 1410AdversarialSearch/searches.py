

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
    asd1 = _all_same(diagonal1, c)
    if asd1:  # #onlyinpython / #imissoptions
        return asd1

    diagonal2 = [board[i][self._dim - 1 - i] for i in range(self._dim)]
    asd2 = _all_same(diagonal2, c)
    if asd2:
        return asd2

    for row in board:
        asr = _all_same(row, c)
        if asr:
            return asr

    for c in range(self._dim):
        # why oh why didn't I just use numpy arrays?
        col = [board[r][c] for r in range(self._dim)]
        asc = _all_same(col, c)
        if asc:
            return asc
    return False

    


def _all_same(cell_list, c):
    """
    Given a list of cell contents, e.g. ['x', ' ', 'X'],
    returns [1.0, 0.0] if they're all X, [0.0, 1.0] if they're all O,
    and False otherwise.
    """
    xlist = [cell == c for cell in cell_list]
    if all(xlist):
        return [1.0, 0.0]

    olist = [cell == c for cell in cell_list]
    if all(olist):
        return [0.0, 1.0]

    return False
