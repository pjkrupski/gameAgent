from scipy.signal import correlate2d
import numpy as np
from searches import L_Pattern, V_Pattern, T_Pattern


def search_L(board, winlength):
    #Searches for a len 5 L pattern of c's 
    return search_By_Pattern(board, *L_Pattern)

#TODO
def search_V(board, winlength):
    return search_By_Pattern(board, *V_Pattern)

#TODO
def search_Star(board, winlength):
    pass

#TODO
def search_T(board, winlength):
#Searches for a len 5 L pattern of c's 
    kernel = np.array(((1, 1, 1), (0, 1, 0), (0, 0, 1)))
    return search_By_Pattern(board, *T_Pattern)

def search_Line(board, winlength):
  kernel = np.ones((1,winlength))
  kernel2 = np.identity(winlength)
  Line_Pattern = [[kernel, kernel2], [np.sum(kernel), np.sum(kernel2)]]
  return search_By_Pattern(board, *Line_Pattern, rotations=2)
    

def search_By_Pattern(board, kernels, sizes, rotations=4):
    possibility = {-1:0, 1:0}
    for i, kernel in enumerate(kernels):
        for j in range(rotations):
            for y in [-1, 1]:
              tempBoard = np.copy(board)
              tempBoard[tempBoard == 0] = y
              output = correlate2d(tempBoard, kernel, mode="valid")
              possibility[y] += np.sum((output == y*sizes[i]).astype(np.float32))
            kernel = np.rot90(kernel)
    return possibility
    