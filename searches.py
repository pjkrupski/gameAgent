from scipy.signal import correlate2d
import numpy as np

kernel = np.array(((1, 0), (1, 0), (1, 1)))
kernel2 = np.array(((0, 1), (0, 1), (1, 1)))
L_Pattern = [[kernel, kernel2], [np.sum(kernel), np.sum(kernel2)]]

kernel = np.array(((1, 0, 1), (0, 1, 0)))
V_Pattern = [[kernel], [np.sum(kernel)]]

kernel = np.array(((1, 1, 1), (0, 1, 0), (0, 1, 0)))
T_Pattern = [[kernel], [np.sum(kernel)]]


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
#Searches for a len 5 T pattern of c's 
    kernel = np.array(((1, 1, 1), (0, 1, 0), (0, 0, 1)))
    return search_By_Pattern(board, *T_Pattern)
    

def search_By_Pattern(board, kernels, sizes,  rotations=4):
    for i, kernel in enumerate(kernels):
        for x in range(rotations):
            x = correlate2d(board, kernel, mode="valid")
            if -sizes[i] in x:
                return -1
            if sizes[i] in x:
                return 1
            kernel = np.rot90(kernel)
    return 0

def search_Line(board, winlength):
    # Check rows
    kernel = np.ones((1,winlength))
    kernel2 = np.identity(winlength)
    Line_Pattern = [[kernel, kernel2], [np.sum(kernel), np.sum(kernel2)]]
    return search_By_Pattern(board, *Line_Pattern, rotations=2)
