

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
