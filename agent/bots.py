#!/usr/bin/python
import math
import numpy as np
from GoT_problem import *
from GoT_types import CellType
import random
#import getch
from reinforce import Reinforce
from reinforce_with_baseline import ReinforceWithBaseline
import tensorflow as tf
import matplotlib.pyplot as plt
# import msvcrt


#model1 = ReinforceWithBaseline(STATE_SIZE2, ACTIONS)

ACTIONS = 4

model = Reinforce(13, 13, ACTIONS)
#model2 = Reinforce(13, ACTIONS)
model.load_weights("d5")

#python gamerunner.py -map maps/empty_room.txt -bots rl safe -runner training -no_image -no_msg -multi_test 10000
#python gamerunner.py -map maps/empty_room.txt -bots rl attack -runner training -no_image -no_msg -multi_test 10000
#python gamerunner.py -map maps/empty_room.txt -bots rl random -runner training -no_image -no_msg -multi_test 10000
#python gamerunner.py -map maps/empty_room.txt -bots rl rl2 -runner training -no_image -no_msg -multi_test 10000
#python gamerunner.py -map maps/empty_room.txt -bots rl student -runner training -no_image -no_msg -multi_test 10000

class StudentBot:
    
    def __init__(self):
        
        self.p1 = 0

        self.rewards = []
        self.states = []
        self.actions = []
        
        self.graph = []
        
        self.games = 0
        self.wins = 0 

    def discount(self, rewards, discount_factor=.95):
        """
        Takes in a list of rewards for each timestep in an episode, and
        returns a list of the discounted rewards for each timestep, which
        are calculated by summing the rewards for each future timestep, discounted
        by how far in the future it is.
        For example, in the simple case where the episode rewards are [1, 3, 5]
        and discount_factor = .99 we would calculate:
        dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
        dr_2 = 3 + 0.99 * 5 = 7.95
        dr_3 = 5
        and thus return [8.8705, 7.95 , 5].
        Refer to the slides for more details about how/why this is done.

        :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
        :param discount_factor: Gamma discounting factor to use, defaults to .99
        :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
        """
        # TODO: Compute discounted rewards
        
        if len(rewards) == 1:
            return rewards

        indices = np.arange(len(rewards))
        total = rewards[0]

        total = total + np.sum(rewards[1:] * np.power(discount_factor, indices[1:]))

        return np.concatenate((np.array([total]), self.discount(rewards[1:], discount_factor)))

    def decide(self, asp: GoTProblem):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        To get started, you can get the current
        state by calling asp.get_start_state()
        """

        #print("check-1")
        #print("check-2")
        
        cstate = asp.get_start_state()
        
        cells = 8
        
        board = cstate.board

        state = np.zeros((len(board),len(board[0]), cells))

        p1 = 0
        p2 = 0 
        
        for row in range(len(board)):
            for col in range(len(board[row])):
                if board[row][col] == '1':
                    state[row][col][0] = 1
                elif board[row][col] == '2':
                    state[row][col][1] = 1
                elif board[row][col] == CellType.ONE_PERM:
                    p1 += 1
                    state[row][col][2] = 1
                elif board[row][col] == CellType.TWO_PERM:
                    state[row][col][3] = 1
                elif board[row][col] == CellType.ONE_TEMP:
                    state[row][col][4] = 1
                elif board[row][col] == CellType.TWO_TEMP:
                    state[row][col][5] = 1
                elif board[row][col] == CellType.WALL:
                    state[row][col][6] = 1
                else:
                    state[row][col][7] = 1

        state /= cells
        
        pred = model(tf.expand_dims(state, 0))[0]   
        safe = asp.get_safe_actions(cstate.board, cstate.player_locs[cstate.ptm], cstate.ptm)

        moves = ['D', 'U', 'L', 'R']
        choices = []
        probs = []
        
        if 'D' in safe:
            choices.append(0)
            probs.append(pred[0])
        if 'U' in safe:
            choices.append(1)
            probs.append(pred[1])
        if 'L' in safe:
            choices.append(2)
            probs.append(pred[2])
        if 'R' in safe:
            choices.append(3)
            probs.append(pred[3])
            
        probs = np.array(probs)
        
        if probs.sum() == 0:
            probs += 0.1
            choices = [0,1,2,3]
            
        #if self.games > 0 and self.games < 10000:
            
            #probs = probs + 0.1 - (0.1*self.games/10000)
            
        if self.games % 100 == 0:
            if len(self.actions)%5 == 0:
                print(pred)
        
        probs /= probs.sum()

        output = np.random.choice(choices, 1, p=probs)[0]
        #print(output)
        #print(moves[output])
        
        self.rewards.append((p1 - self.p1)/1073)
        #self.rewards.append(0)
        self.actions.append(output)
        self.states.append(state)
        
        self.p1 = p1 
        
        return moves[output]
    
    def cleanup(self):
        
        '''self.games += 1
        
        if len(self.rewards) > 0:
            
            if self.p1 > 1:
                print(self.p1)
            #else:
            self.rewards = self.rewards[1:]
            
            if w == 0:
                self.rewards.append((1073/2-self.p1)/1073)
                #self.rewards.append(1)
                self.wins += 1
            else:
                self.rewards.append(0)
                #-0.01 * (self.games/10000))
            with tf.GradientTape() as tape:
                discounted_rewards = self.discount(self.rewards)
                
                loss2 = model.loss(self.states, self.actions, discounted_rewards)
        
            gradients = tape.gradient(loss2, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            print("NO REWARDS!")
        
        if self.games%100 == 0:
            print("Wins/100: " + str(self.wins))
            print("Games: " + str(self.games))
            print("Score: " + str(self.p1))
            self.graph.append(self.wins)
            self.wins = 0 
        
        if self.games%10000 == 0 and len(self.graph) > 1:
            plt.plot(np.arange(len(self.graph)),self.graph)
            plt.show()
            #model.save_weights("e1")
            
        self.rewards = []
        self.states = []
        self.actions = []
        self.p1 = 0'''
        
        pass
        
class RLBot2:
    
    def __init__(self):
        
        self.p1 = 0

        self.rewards = []
        self.states = []
        self.actions = []
        
        self.graph = []
        
        self.games = 0
        self.wins = 0 

    def discount(self, rewards, discount_factor=.95):
        """
        Takes in a list of rewards for each timestep in an episode, and
        returns a list of the discounted rewards for each timestep, which
        are calculated by summing the rewards for each future timestep, discounted
        by how far in the future it is.
        For example, in the simple case where the episode rewards are [1, 3, 5]
        and discount_factor = .99 we would calculate:
        dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
        dr_2 = 3 + 0.99 * 5 = 7.95
        dr_3 = 5
        and thus return [8.8705, 7.95 , 5].
        Refer to the slides for more details about how/why this is done.

        :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
        :param discount_factor: Gamma discounting factor to use, defaults to .99
        :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
        """
        # TODO: Compute discounted rewards
        
        if len(rewards) == 1:
            return rewards

        indices = np.arange(len(rewards))
        total = rewards[0]

        total = total + np.sum(rewards[1:] * np.power(discount_factor, indices[1:]))

        return np.concatenate((np.array([total]), self.discount(rewards[1:], discount_factor)))

    def decide(self, asp: GoTProblem):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        To get started, you can get the current
        state by calling asp.get_start_state()
        """

        #print("check-1")
        #print("check-2")
        
        #print("CHECKED")
        
        cstate = asp.get_start_state()
        
        cells = 8

        state = np.zeros((STATE_SIZE,STATE_SIZE, cells))
        
        #print("2")
        
        board = cstate.board
        
        p1 = 0
        p2 = 0 
        
        
        for row in range(len(board)):
            for col in range(len(board[row])):
                if board[row][col] == '1':
                    state[row][col][0] = 1
                elif board[row][col] == '2':
                    state[row][col][1] = 1
                elif board[row][col] == CellType.ONE_PERM:
                    state[row][col][2] = 1
                elif board[row][col] == CellType.TWO_PERM:
                    p1 += 1
                    state[row][col][3] = 1
                elif board[row][col] == CellType.ONE_TEMP:
                    state[row][col][4] = 1
                elif board[row][col] == CellType.TWO_TEMP:
                    state[row][col][5] = 1
                elif board[row][col] == CellType.WALL:
                    state[row][col][6] = 1
                else:
                    state[row][col][7] = 1

        state /= cells
        
        pred = model2(tf.expand_dims(state, 0))[0]   
        safe = asp.get_safe_actions(cstate.board, cstate.player_locs[cstate.ptm], cstate.ptm)
        
        #print(safe)
        
        moves = ['D', 'U', 'L', 'R']
        choices = []
        probs = []
        
        if 'D' in safe:
            choices.append(0)
            probs.append(pred[0])
        if 'U' in safe:
            choices.append(1)
            probs.append(pred[1])
        if 'L' in safe:
            choices.append(2)
            probs.append(pred[2])
        if 'R' in safe:
            choices.append(3)
            probs.append(pred[3])
            
        probs = np.array(probs)
        
        if probs.sum() == 0:
            probs += 0.1
            choices = [0,1,2,3]
            
        if self.games > 0 and self.games < 10000:
            
            probs = probs + 0.1 - (0.1 * self.games/10000)
        
        probs /= probs.sum()

        output = np.random.choice(choices, 1, p=probs)[0]
        #print(output)
        #print(moves[output])
        
        self.rewards.append((p1 - self.p1)/121)
        #self.rewards.append(0)
        self.actions.append(output)
        self.states.append(state)
        
        self.p1 = p1
        
        return moves[output]
    
    def cleanup(self, w):
        
        self.games += 1
        
        if len(self.rewards) > 0:
            
            self.rewards = self.rewards[1:]
            
            if w == 0:
                self.rewards.append(0)
            else:
                self.rewards.append((61 - self.p1)/121)
                self.wins += 1

            with tf.GradientTape() as tape:
                discounted_rewards = self.discount(self.rewards)
                
                loss2 = model2.loss(self.states, self.actions, discounted_rewards)
        
            gradients = tape.gradient(loss2, model2.trainable_variables)
            model2.optimizer.apply_gradients(zip(gradients, model2.trainable_variables))
        else:
            print("NO REWARDS!")
        
        if self.games%100 == 0:
            print("Wins/100: " + str(self.wins))
            print("Games: " + str(self.games))
            print("Score: " + str(self.p1))
            self.graph.append(self.wins)
            self.wins = 0 
            
        self.rewards = []
        self.states = []
        self.actions = []
        self.p1 = 0


# Throughout this file, ASP means adversarial search problem.
class StudentBot2:
    """ Write your student bot here """

    """alpha beta pruning cutoff bot"""

    def __init__(self):
        self.wwloc = np.zeros(2).astype(int)

    def heuristic_func(cstate, s):

        c1 = 1
        c2 = 1

        board = cstate.board

        for r in range(len(board)):
            for c in range(len(board[r])):
                if board[r][c] == CellType.ONE_PERM:
                    c1 += 2
                    c2 -= 0.01

                if board[r][c] == CellType.ONE_TEMP:
                    c1 += 0.01

                elif board[r][c] == CellType.TWO_PERM:
                    c2 += 2
                    c1 -= 0.01

                if board[r][c] == CellType.TWO_TEMP:
                    c2 += 0.01

        if s == 0:

            return c1

        else:

            return c2

    def getMin3(self, asp: GoTProblem, cstate, limit, s, c, c2, a2, ww):

        #whitewalker

        size = len(cstate.board) * len(cstate.board[0])

        if asp.is_terminal_state(cstate):

            p1, p2 = asp.evaluate_state(cstate)

            loc = cstate.player_locs[cstate.ptm]

            if s == 0:

                #if cstate.board[loc[0]][loc[1]] == CellType.WALL:
                    #p1 = 0

                return a2, p1 * size

            else:

                #if cstate.board[loc[0]][loc[1]] == CellType.WALL:
                    #p2 = 0

                return a2, p2 * size

        if c2 >= c:

            #print("m9")

            return 0, StudentBot.heuristic_func(cstate, s)

        maxs = []

        actions = asp.get_available_actions(cstate)

        a2 = []

        for action in actions:

            if len(maxs) == 0:
                nlimit = None
            else:
                nlimit = min(maxs)

            a, m = StudentBot.getMax3(self, asp, asp.transition(cstate,action), nlimit, s, c, c2+1, action, ww)

            a2.append(action)
            maxs.append(m)

            if not limit is None and m < limit:
                break

        element = min(maxs)
        index = np.argmin(maxs)

        return a2[index], element


    def getMax3(self, asp: GoTProblem, cstate, limit, s, c, c2, a2, ww):

        tww = np.zeros(2).astype(int)
        tww2 = np.zeros(2).astype(int)

        size = len(cstate.board) * len(cstate.board[0])

        if self.wwloc[0] > 0 and self.wwloc[1] > 0:

            tww[0] = cstate.ww_locs[0][0]
            tww[1] = cstate.ww_locs[0][1]
            tww2[0] = ww[0]
            tww2[1] = ww[1]

            for i in range(int(c2/2)+1):
                tww[0] = tww[0] + tww2[0]
                tww[1] = tww[1] + tww2[1]

                if cstate.board[tww[0]][tww[1]] in CellType.STOPS_WHITE_WALKERS:

                    t = False
                    if cstate.board[tww[0]-tww2[0]][tww[1]] in CellType.STOPS_WHITE_WALKERS:
                        tww2[1] *= -1
                        tww[1] += tww2[1]*2
                        t = True

                    if cstate.board[tww[0]][tww[1]-tww2[1]] in CellType.STOPS_WHITE_WALKERS:
                        tww2[0] *= -1
                        tww[0] += tww2[0]*2
                        t = True

                    if not t:
                        tww2[0] *= -1
                        tww2[1] *= -1
                        tww[0] += tww2[0]*2
                        tww[1] += tww2[1]*2

        if asp.is_terminal_state(cstate):
            p1, p2 = asp.evaluate_state(cstate)
            loc = cstate.player_locs[cstate.ptm]

            if s == 0:

                return a2, p1 * size

            else:

                return a2, p2 * size

        if c2 >= c:

            return 0, StudentBot.heuristic_func(cstate, s)

        mins = []

        actions = asp.get_available_actions(cstate)

        a2 = []

        for action in actions:

            if len(mins) == 0:
                nlimit = None
            else:
                nlimit = max(mins)

            cstate2 = asp.transition(cstate,action)

            if not cstate2.player_locs[0] is None and not cstate2.player_locs[1] is None:

                if s == 0 and cstate2.board[tww[0]][tww[1]] == CellType.ONE_TEMP:
                    a2.append(action)
                    mins.append(0)
                    continue
                if s == 1 and cstate2.board[tww[0]][tww[1]] == CellType.TWO_TEMP:
                    a2.append(action)
                    mins.append(0)
                    continue
                if s == 0 and tww[0] == cstate2.player_locs[0][0] and tww[1] == cstate2.player_locs[0][1]:
                    a2.append(action)
                    mins.append(0)
                    continue
                if s == 1 and tww[0] == cstate2.player_locs[1][0] and tww[1] == cstate2.player_locs[1][1]:
                    a2.append(action)
                    mins.append(0)
                    continue

            a, m = StudentBot.getMin3(self, asp,asp.transition(cstate,action), nlimit, s, c, c2+1, action, ww)

            a2.append(action)
            mins.append(m)

            if not limit is None and m > limit:
                break

        element = max(mins)
        index = np.argmax(mins)

        if np.sum(mins) == 0:

            sa = list(asp.get_safe_actions(cstate.board, cstate.player_locs[cstate.ptm], cstate.ptm))

            if len(sa) > 0:
                return sa[0], 0


        return a2[index], element


    def alpha_beta_cutoff(self,
        asp: GoTProblem,
        cutoff_ply: int):
        """
        This function should:
        - search through the asp using alpha-beta pruning
        - cut off the search after cutoff_ply moves have been made.

        Input:
            asp - an AdversarialSearchProblem
            cutoff_ply - an Integer that determines when to cutoff the search and
                use heuristic_func. For example, when cutoff_ply = 1, use
                heuristic_func to evaluate states that result from your first move.
                When cutoff_ply = 2, use heuristic_func to evaluate states that
                result from your opponent's first move. When cutoff_ply = 3 use
                heuristic_func to evaluate the states that result from your second
                move. You may assume that cutoff_ply > 0.
            heuristic_func - a function that takes in a GameState and outputs a
                real number indicating how good that state is for the player who is
                using alpha_beta_cutoff to choose their action. You do not need to
                implement this function, as it should be provided by whomever is
                calling alpha_beta_cutoff, however you are welcome to write
                evaluation functions to test your implemention. The heuristic_func
                we provide does not handle terminal states, so evaluate terminal
                states the same way you evaluated them in the previous algorithms.
        Output:
            an action(an element of asp.get_available_actions(asp.get_start_state()))
        """

        dir = np.zeros(2).astype(int)
        loc = np.zeros(2).astype(int)

        if len(asp.get_start_state().ww_locs) > 0:
            loc[0] = asp.get_start_state().ww_locs[0][0]
            loc[1] = asp.get_start_state().ww_locs[0][1]

        if not (self.wwloc[0] == 0 and self.wwloc[1] == 0):

            dir[0] = loc[0]-self.wwloc[0]
            dir[1] = loc[1]-self.wwloc[1]

        self.wwloc = loc

        s = asp.get_start_state().player_to_move()

        action, m = StudentBot.getMax3(self, asp, asp.get_start_state(), None, s, cutoff_ply, 0, "U", dir)

        return action

    def decide(self, asp: GoTProblem):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        To get started, you can get the current
        state by calling asp.get_start_state()
        """

        #print("check-1")
        #print("check-2")
        print("TEST")

        #print(asp.get_start_state().get_safe_actions())
        
        cstate = asp.get_start_state()

        print(asp.get_safe_actions(cstate.board, cstate.player_locs[cstate.ptm], cstate.ptm))
        action = StudentBot.alpha_beta_cutoff(self, asp, 2)

        return action

    def cleanup(self, w):
        """
        Input: None
        Output: None
        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """

        print("WINNER: " + str(w))
        pass


class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(GoTProblem.get_safe_actions(board, loc, ptm))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self, w):
        pass


class ManualBot:
    """Bot which can be manually controlled using W, A, S, D"""

    def decide(self, asp: GoTProblem):
        # maps keyboard input to {U, D, L, R}
        dir_map = {'A': 'L', 'W': 'U',
                   'a': 'L', 'w': 'U',
                   'S': 'D', 'D': 'R',
                   's': 'D', 'd': 'R'}
        # Command for mac/unix:
        direction = getch.getch()
        # Command for Windows:
        # direction = msvcrt.getch().decode('ASCII')
        return dir_map[direction]

    def cleanup(self):
        pass

class AttackBot:
    """Aggressive bot which attacks opposing player when possible"""

    def __init__(self):
        self.prev_cell_type = None
        self.last_move = None
        self.ptm = None
        self.perm_cell = None
        self.temp_cell = None
        self.opp_temp_cell = None

    def cleanup(self, w):
        self.prev_cell_type = None
        self.last_move = None
        self.ptm = None
        self.perm_cell = None
        self.temp_cell = None
        self.opp_temp_cell = None

    def dist_from_opp(self, opp_loc, ptm_loc):
        dist = 0
        for i in range(len(opp_loc)):
            dist += abs(opp_loc[i] - ptm_loc[i])
        return dist

    def min_dist_to_temp(self, board, ptm_loc):
        locs = self.temp_barrier_locs_from_board(board)
        min_dist = math.inf
        for loc in locs:
            this_dist = self.dist_from_opp(loc, ptm_loc)
            if this_dist < min_dist:
                min_dist = this_dist
        return min_dist

    def temp_barrier_locs_from_board(self, board):
        if self.opp_temp_cell is None:
            return []
        loc_dict = {}
        num_temp = 0
        for r in range(len(board)):
            for c in range(len(board[r])):
                char = board[r][c]
                if char == self.opp_temp_cell:
                    loc_dict[num_temp] = (r, c)
                    num_temp += 1
        loc_list = []
        for index in range(num_temp):
            loc_list.append(loc_dict[index])
        return loc_list

    def decide(self, asp):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(GoTProblem.get_safe_actions(board, loc, ptm))
        opp_loc = locs[(ptm + 1) % 2]
        if self.ptm is None:
            self.ptm = ptm
            self.perm_cell = CellType.TWO_PERM
            self.temp_cell = CellType.TWO_TEMP
            self.opp_temp_cell = CellType.ONE_TEMP
            if ptm == 0:
                self.perm_cell = CellType.ONE_PERM
                self.temp_cell = CellType.ONE_TEMP
                self.opp_temp_cell = CellType.TWO_TEMP

        if not possibilities:
            return "U"

        if self.prev_cell_type is None:
            "Attack bot starting"
            self.prev_cell_type = self.temp_cell
            this_move = possibilities[0]
            self.last_move = this_move
            return this_move

        # if player needs to return to perm area
        must_return_to_perm = False
        if self.prev_cell_type == self.temp_cell:
            this_move = None
            if self.last_move == "U":
                this_move = "D"
            elif self.last_move == "D":
                this_move = "U"
            elif self.last_move == "R":
                this_move = "L"
            elif self.last_move == "L":
                this_move = "R"
            else:
                raise Exception

            self.prev_cell_type = self.perm_cell
            self.last_move = this_move
            must_return_to_perm = True

        # else, player is (potentially) leaving perm area
        min_dist = math.inf
        min_dist_to_temp = math.inf
        go_for_temp = False
        decision = possibilities[0]
        min_next_loc = [None] * 2
        for move in possibilities:
            next_loc = GoTProblem.move(loc, move)
            dist_from_opponent = self.dist_from_opp(next_loc, opp_loc)
            this_dist_to_temp = self.min_dist_to_temp(board, next_loc)
            if this_dist_to_temp == 0:
                return move

            if not must_return_to_perm:
                # If we are close to temp barrier
                if this_dist_to_temp <= 5 or go_for_temp:
                    go_for_temp = True
                    if this_dist_to_temp < min_dist_to_temp:
                        min_dist_to_temp = this_dist_to_temp
                        decision = move
                        min_next_loc = next_loc

                elif dist_from_opponent < min_dist:
                    min_dist = dist_from_opponent
                    decision = move
                    min_next_loc = next_loc
                    min_dist_to_temp = this_dist_to_temp

                elif dist_from_opponent == min_dist:
                    if this_dist_to_temp < min_dist_to_temp:
                        min_dist_to_temp = this_dist_to_temp
                        decision = move
                        min_next_loc = next_loc

        if not must_return_to_perm:
            if board[min_next_loc[0]][min_next_loc[1]] == self.perm_cell:
                self.prev_cell_type = self.perm_cell
            else:
                self.prev_cell_type = self.temp_cell
            self.last_move = decision
        return self.last_move


class SafeBot:
    """Bot that plays safe and takes area"""

    def __init__(self):
        self.prev_move = None
        self.to_empty = []
        self.algo_path = []
        self.path = []
        self.calc_empty = False
        self.order = {"U": ("L", "R"),
                    "D": ("L", "R"),
                    "L": ("U", "D"),
                    "R": ("U", "D")}

    def cleanup(self, w):
        self.prev_move = None
        self.to_empty = []
        self.algo_path = []
        self.path = []
        self.calc_empty = False
        self.order = {"U": ("L", "R"),
                    "D": ("L", "R"),
                    "L": ("U", "D"),
                    "R": ("U", "D")}

    def get_safe_neighbors_wall(self, board, loc):
        neighbors = [
                ((loc[0] + 1, loc[1]), D),
                ((loc[0] - 1, loc[1]), U),
                ((loc[0], loc[1] + 1), R),
                ((loc[0], loc[1] - 1), L),
            ]
        return list(filter(lambda m: board[m[0][0]][m[0][1]] != CellType.WALL, neighbors))

    def get_safe_neighbors_no_wall(self, board, loc, wall):
        neighbors = [
                ((loc[0] + 1, loc[1]), D),
                ((loc[0] - 1, loc[1]), U),
                ((loc[0], loc[1] + 1), R),
                ((loc[0], loc[1] - 1), L),
            ]
        return list(filter(lambda m: board[m[0][0]][m[0][1]] != CellType.WALL and board[m[0][0]][m[0][1]] != wall, neighbors))

    def decide(self, asp: GoTProblem):
        state = asp.get_start_state()
        if not self.path:
            if self.calc_empty:
                self.gen_path_to_empty(state)
                self.path += self.to_empty
                self.to_empty = []
                self.calc_empty = False
            else:
                self.gen_space_grab(state)
                self.path += self.algo_path
                self.algo_path = []
                self.calc_empty = True
        move = self.path.pop(0)
        self.prev_move = move
        return move

    def gen_space_grab(self, state : GoTState):
        board = state.board
        loc = state.player_locs[state.ptm]
        if state.ptm == 0:
            player_wall = CellType.ONE_PERM
        else:
            player_wall = CellType.TWO_PERM
        avail_actions = {U, D, L, R}
        prev = self.prev_move
        if prev:
            avail_actions.remove(prev)
        else:
            safe_actions = self.get_safe_neighbors_wall(board, loc)
            random.shuffle(safe_actions)
            loc, move = safe_actions[0]
            self.algo_path.append(move)
            avail_actions.remove(move)
            prev = move
        while avail_actions:
            safe_moves = self.get_safe_neighbors_no_wall(board, loc, player_wall)
            safe_moves_wall = self.get_safe_neighbors_wall(board,loc)
            if not safe_moves and not safe_moves_wall:
                self.algo_path.append(U)
                return
            random.shuffle(safe_moves)
            random.shuffle(safe_moves_wall)
            use_wall = True
            for loc, move in safe_moves:
                board_val = board[loc[0]][loc[1]]
                if move in self.order[prev] and move in avail_actions and board_val != player_wall:
                    self.algo_path.append(move)
                    avail_actions.remove(move)
                    prev = move
                    use_wall = False
                    break
            if use_wall:
               for loc, move in safe_moves_wall:
                    board_val = board[loc[0]][loc[1]]
                    if move in self.order[prev] and move in avail_actions:
                        self.algo_path.append(move)
                        avail_actions.remove(move)
                        prev = move
                        use_wall = False
                        break
        return

    def gen_path_to_empty(self, state: GoTState):
        board = state.board
        player_loc = state.player_locs[state.ptm]
        to_check = [(player_loc, None)]
        checked = {(player_loc, None): None}
        while to_check:
            loc, m = to_check.pop(0)
            neighbors = [
                ((loc[0] + 1, loc[1]), D),
                ((loc[0] - 1, loc[1]), U),
                ((loc[0], loc[1] + 1), R),
                ((loc[0], loc[1] - 1), L),
            ]
            random.shuffle(neighbors)
            for move in neighbors:
                x, y = move[0][0], move[0][1]
                board_val = board[x][y]
                if move not in checked and board_val != CellType.WALL:
                    checked[move] = (loc, m)
                    if board_val == ' ':
                        path = []
                        while move[1] is not None:
                            path.append(move[1])
                            move = checked[move]
                        self.to_empty += path
                        return
                    else:
                        to_check.append(move)
        self.to_empty += [U]
        return
