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
        eval_board = self.find_n_inarow(row, winlength)
        if eval_board != 0:
          return eval_board

    # Check columns
    for c in range(self._dim):
        col = [board[r][c] for r in range(self._dim)]
        eval_board = self.find_n_inarow(col, winlength)
        if eval_board != 0:
          return eval_board


    # Check diagonals
    for b in [board, np.fliplr(board), np.flipud(board)]:
      for k in range(self._dim - winlength + 1):

          if k == 0:   #left to right diag
              diag1 = np.diag(b)
          else:  #left to right diag
              diag1 = np.diag(b, k)

          eval_board = self.find_n_inarow(diag1, winlength)
          if eval_board != 0:
            return eval_board


          diag2 = np.diag(b,-k) #right to left drag

          eval_board = self.find_n_inarow(diag2, winlength)
          if eval_board != 0:
            return eval_board
    
    return 0


def find_n_inarow(dim, winlength):
    xs = 0
    os = 0
    for i in range(0, len(dim)):
        if xs == winlength:
          return 1
        elif os == winlength:
          return -1

        if dim[i] == 1:
            xs += 1
            os = 0
        elif dim[i] == -1:
            os += 1
            xs = 0
        else:
            os = 0
            xs = 0

    return 0
