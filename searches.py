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

        eval_x = self.find_n_inarow(row, winlength, 1)

        if eval_x[0]: 
          print(eval_x[1], " won in row ")
          return eval_o[1]

        eval_o = self.find_n_inarow(row, winlength, -1)
        
        if eval_o[0]: 
          print(eval_o[1], " won in row ")
          return eval_o[1]

    # Check columns
    for c in range(self._dim):
        col = [board[r][c] for r in range(self._dim)]

        eval_x = self.find_n_inarow(col, winlength, 1)

        if eval_x[0]: 
          print(eval_x[1], " won in col ")
          return eval_o[1]

        eval_o = self.find_n_inarow(col, winlength, -1)
        
        if eval_o[0]: 
          print(eval_o[1], " won in col ")
          return eval_o[1]





    # Check diagonals
    for k in range(self._dim - winlength + 1):

        if k == 0:   #left to right diag
            diag1 = np.diag(board)
        else:  #left to right diag
            diag1 = np.diag(board, k)


        eval_x = self.find_n_inarow(diag1, winlength, 1)
        if eval_x[0]:
          print(eval_x[1], " won in diag 1 ")
          return eval_o[1]


        eval_o = self.find_n_inarow(diag1, winlength, -1)
        if eval_o[0]:
          print(eval_o[1], " won in diag 1 ")
          return eval_o[1]

        diag2 = np.diag(board,-k) #right to left drag

        eval_x = self.find_n_inarow(diag2, winlength, 1)
        if eval_x[0]:
          print(eval_x[1], " won in diag 2 ")
          return eval_x[1]

        eval_o = self.find_n_inarow(diag2, winlength, -1)
        if eval_o[0]:
          print(eval_o[1], " won in diag 2 ")
          return eval_o[1]
    
   # print(" returning 0 from search line ", flush=True)
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

def find_n_inarow(dim, winlength, char):
    #print(char, " is char")
    count = 0
    for i in range(0, len(dim)):
        if count == winlength:
            return (True, char)
        if dim[i] == char:
            #print("dim at ", i, " is = to ", dim[i], " and char is ", char, " count becomes ", count+1)
            count += 1
        else:
            count = 0
    return (False, char)





















