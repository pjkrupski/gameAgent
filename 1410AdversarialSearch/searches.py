
#look at this
def search_L(self, target_char, board, row, col):
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == target_char:
                #work on caching and storing visited nodes
                num_neighbors = 0
                if c+1 < len(board[0]) and board[r][c+1] == target_char:
                    num_neighbors += 1
                if c-1 >= 0 and board[r][c-1] == target_char:
                    num_neighbors += 1
                if r+1 < len(board) and board[r+1][c] == target_char:
                    num_neighbors += 1
                if r-1 >= 0 and board[r-1][c] == target_char:
                    num_neighbors += 1
                if num_neighbors >= 2:
                    return True


    # # Define offsets for all possible L-shapes
    # l_shape_offsets = [
    #     ((0, 1), (1, 0)),   # Top left
    #     ((0, -1), (1, 0)),  # Top right
    #     ((-1, 0), (0, 1)),  # Bottom left
    #     ((-1, 0), (0, -1))  # Bottom right
    # ]

    # # Iterate through all cells in the board
    # for r in range(len(board)):
    #     for c in range(len(board[0])):
    #         for offset1, offset2 in l_shape_offsets:
    #             # Calculate coordinates for the two offsets
    #             r1, c1 = r + offset1[0], c + offset1[1]
    #             r2, c2 = r + offset2[0], c + offset2[1]

    #             # Ensure the offsets are within the board bounds
    #             if 0 <= r1 < len(board) and 0 <= c1 < len(board[0]) and \
    #                0 <= r2 < len(board) and 0 <= c2 < len(board[0]):
    #                 # Check if the cells form an L-shape with the target character
    #                 if _all_same([board[r][c], board[r1][c1], board[r2][c2]], target_char):
    #                     return True

    # return False
# def search_L(self, c, board, row, col):
#         # rows = self._dim
#         # cols = self._dim
#         top_left_corner = [board[0][0], board[1][0], board[0][1]]
#         top_right_corner = [board[0][-1], board[0][-2], board[1][-1]]
#         bottom_left_corner = [board[-1][0], board[-2][0], board[-1][1]]
#         bottom_right_corner = [board[-1][-1], board[-1][-2], board[-2][-1]]
        
#         if _all_same(top_left_corner, c):
#             return True
#         if _all_same(top_right_corner, c):
#             return True
#         if _all_same(bottom_left_corner, c):
#             return True
#         if _all_same(bottom_right_corner, c):
#             return True
#         return False
        #Searches for a len 5 L pattern of c's 
        # rows = self._dim
        # cols = self._dim
        # if (row - 2 <= 0 and col + 2 < cols and
        #     board[row-1][col] == c and
        #     board[row-2][col] == c and
        #     board[row][col+1] == c and
        #     board[row][col+2] == c):
        #     return True
        # if (row + 2 < rows and col + 2 < cols and
        #     board[row + 1][col] == c and
        #     board[row + 2][col] == c and
        #     board[row][col+1] == c and
        #     board[row][col+2] == c):
        #     return True
        # if (row - 2 <= 0 and col - 2 >= 0 and
        #     board[row-1][col] == c and
        #     board[row-2][col] == c and
        #     board[row][col-1] == c and
        #     board[row][col-2] == c):
        #     return True
        # if (row + 2 < rows and col + 2 < cols and
        #     board[row + 1][col] == c and
        #     board[row + 2][col] == c and
        #     board[row][col-1] == c and
        #     board[row][col-2] == c):
        #     return True
        # return False

        #searches for L in the corners

#TODO
def search_V(self, target_char, board, row, col):


    v_shape_offsets = [
        ((0, 1), (1, 0), (1, 2)),   # V-shape pattern
        ((1, 0), (2, 1), (0, 1)),   # Inverted V-shape pattern
        ((0, 1), (1, 1), (2, 1)),   # Sideways V-shape pattern
        ((1, 0), (1, 1), (1, 2))    # Sideways inverted V-shape pattern
    ]

    # Iterate through potential starting points for V-shape patterns
    for r in range(row - 2):  # Adjust range for potential V-shape patterns
        for c in range(col - 2):  # Adjust range for potential V-shape patterns
            # Iterate through V-shape patterns
            for offsets in v_shape_offsets:
                # Check if all cells in the V-shape pattern contain the target character
                if _all_same([board[r + dr][c + dc] for dr, dc in offsets], target_char):
                    return True

    return False
#TODO
def search_Star(self, target_char, board, row, col):

    # Define offsets for all neighboring cells
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    # Iterate through all cells in the board
    for r in range(len(board)):
        for c in range(len(board[0])):
            # Check if the current cell contains the target character
            if board[r][c] == target_char:
                # Check if all neighboring cells contain the target character
                if all(_all_same([board[r + dr][c + dc] for dr, dc in neighbor_offsets if 0 <= r + dr < len(board) and 0 <= c + dc < len(board[0])], target_char) for r, c in neighbor_offsets):
                    return True

    return False


#TODO
def search_T(self, target_char, board, row, col):
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

def search_Line(self, target_char, board, row, col):
#Searches for a len(board) line of c's 
   
    diagonal1 = [board[i][i] for i in range(self._dim)]
    
    # bind the output of _all_same for diagonal1 for its two uses
    asd1 = _all_same(diagonal1, target_char)
    if asd1:  # #onlyinpython / #imissoptions
        return True

    diagonal2 = [board[i][self._dim - 1 - i] for i in range(self._dim)]
    asd2 = _all_same(diagonal2, target_char)
    if asd2:
        return True

    for row in board:
        asr = _all_same(row, target_char)
        if asr:
            return True

    for c in range(self._dim):
        # why oh why didn't I just use numpy arrays?
        col = [board[r][c] for r in range(self._dim)]
        asc = _all_same(col, target_char)
        if asc:
            return True
    return False

    


def _all_same(cell_list, target_char):
    """
    Given a list of cell contents, e.g. ['x', ' ', 'X'],
    returns [1.0, 0.0] if they're all X, [0.0, 1.0] if they're all O,
    and False otherwise.
    """
    xlist = [cell == target_char for cell in cell_list]
    if all(xlist):
        return [1.0, 0.0]

    olist = [cell == target_char for cell in cell_list]
    if all(olist):
        return [0.0, 1.0]

    return False
