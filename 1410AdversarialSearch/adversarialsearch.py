import sys
from typing import Callable
import time

sys.path.insert(0, "../../to_distribute")

from adversarialsearchproblem import (
    Action,
    AdversarialSearchProblem,
    State as GameState,
)
from asps.gamedag import DAGState, GameDAG


def minimax(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
    """
    print("minimax playing.....")
    state = asp.get_start_state()
    bestMove = None
    bestVal = float("-inf")
    player = state.player_to_move()
    print("player to move is ", player, flush=True)
    for action in asp.get_available_actions(state):
        next_state = asp.transition(state, action)
        val = min_value(asp, next_state, player)
        if val > bestVal:
            bestMove = action
            bestVal = val
        num_it += 1
    print("num it: ", num_it)
    print(f"This took: {time.time() - begin} seconds")

    assert bestMove is not None
    return bestMove


def max_value(asp, state, player):
    if asp.is_terminal_state(state):
        payoffs = asp.evaluate_terminal(state)
        return payoffs[player]
    v = float("-inf")
    print("getting available actions in max... ")
    for a in asp.get_available_actions(state):
        next_state = asp.transition(state, a)
        v = max(v, min_value(asp, next_state, player))
    return v


def min_value(asp, state, player):
    if asp.is_terminal_state(state):
        payoffs = asp.evaluate_terminal(state)
        return payoffs[player]
    v = float("inf")
    print("getting available actions in min... ")
    for a in asp.get_available_actions(state):
        next_state = asp.transition(state, a)
        v = min(v, max_value(asp, next_state, player))
    return v


def alpha_beta(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """

    state = asp.get_start_state()
    bestMove = None
    bestVal = float("-inf")
    player = state.player_to_move()
    alpha = float("-inf")
    beta = float("inf")
    for action in asp.get_available_actions(state):
        next_state = asp.transition(state, action)
        val = ab_min_value(asp, next_state, player, alpha, beta)
        if val > bestVal:
            bestMove = action
            bestVal = val
        if val >= beta:
            break
        alpha = max(alpha, val)
    assert bestMove is not None
    return bestMove


def ab_max_value(asp, state, player, alpha, beta):
    if asp.is_terminal_state(state):
        payoffs = asp.evaluate_terminal(state)
        return payoffs[player]
    v = float("-inf")
    for a in asp.get_available_actions(state):
        next_state = asp.transition(state, a)
        v = max(v, ab_min_value(asp, next_state, player, alpha, beta))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v


def ab_min_value(asp, state, player, alpha, beta):
    if asp.is_terminal_state(state):
        payoffs = asp.evaluate_terminal(state)
        return payoffs[player]
    v = float("inf")
    for a in asp.get_available_actions(state):
        next_state = asp.transition(state, a)
        v = min(v, ab_max_value(asp, next_state, player, alpha, beta))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v


def alpha_beta_cutoff(
    asp: AdversarialSearchProblem[GameState, Action],
    cutoff_ply: int,
    # See AdversarialSearchProblem:heuristic_func
    heuristic_func: Callable[[GameState], float],
) -> Action:
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
    state = asp.get_start_state()
    bestMove = None
    bestVal = float("-inf")
    alpha = float("-inf")
    beta = float("inf")
    player = state.player_to_move()
    for action in asp.get_available_actions(state):
        next_state = asp.transition(state, action)
        val = abc_min_value(
            asp, next_state, player, alpha, beta, cutoff_ply - 1, heuristic_func
        )
        if val > bestVal:
            bestMove = action
            bestVal = val
        if bestVal >= beta:
            break
        alpha = max(alpha, val)
    assert bestMove is not None
    return bestMove


def abc_max_value(asp, state, player, alpha, beta, cutoff_ply, heuristic_func):
    if asp.is_terminal_state(state):
        payoffs = asp.evaluate_terminal(state)
        return payoffs[player]
    if cutoff_ply == 0:
        return heuristic_func(state)
    v = float("-inf")
    for a in asp.get_available_actions(state):
        next_state = asp.transition(state, a)
        v = max(
            v,
            abc_min_value(
                asp, next_state, player, alpha, beta, cutoff_ply - 1, heuristic_func
            ),
        )
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v


def abc_min_value(asp, state, player, alpha, beta, cutoff_ply, heuristic_func):
    if asp.is_terminal_state(state):
        payoffs = asp.evaluate_terminal(state)
        return payoffs[player]
    if cutoff_ply == 0:
        return heuristic_func(state)
    v = float("inf")
    for a in asp.get_available_actions(state):
        next_state = asp.transition(state, a)
        v = min(
            v,
            abc_max_value(
                asp, next_state, player, alpha, beta, cutoff_ply - 1, heuristic_func
            ),
        )
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v


def general_minimax(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the generalization of the minimax algorithm that was
    discussed in the handout, making no assumptions about the
    number of players or reward structure of the given game.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
    """
    print("general_minimax")
    state = asp.get_start_state()
    bestMove = None
    bestVal = float("-inf")
    player = state.player_to_move()
    for action in asp.get_available_actions(state):
        next_state = asp.transition(state, action)
        val = float("-inf")
        if next_state.player_to_move() == player:
            val = multi_max_value(asp, next_state, player)
        else:
            val = multi_min_value(asp, next_state, player)
        if val > bestVal:
            bestMove = action
            bestVal = val
    assert bestMove is not None
    return bestMove


def multi_max_value(asp, state, player):
    if asp.is_terminal_state(state):
        payoffs = asp.evaluate_terminal(state)
        return payoffs[player]
    v = float("-inf")
    for a in asp.get_available_actions(state):
        next_state = asp.transition(state, a)
        if next_state.player_to_move() == player:
            v = max(v, multi_max_value(asp, next_state, player))
        else:
            v = max(v, multi_min_value(asp, next_state, player))
    return v


def multi_min_value(asp, state, player):
    if asp.is_terminal_state(state):
        payoffs = asp.evaluate_terminal(state)
        return payoffs[player]
    v = float("inf")
    for a in asp.get_available_actions(state):
        next_state = asp.transition(state, a)
        if next_state.player_to_move() == player:
            v = min(v, multi_max_value(asp, next_state, player))
        else:
            v = min(v, multi_min_value(asp, next_state, player))
    return v


def example_dag():
    """
    An example of an implemented GameDAG from the gamedag class.
    Look at handout in section 3.3 to see visualization of the tree.
    """

    X = True
    _ = False
    matrix = [
        [_, X, X, _, _, _, _],
        [_, _, _, X, X, _, _],
        [_, _, _, _, _, X, X],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
        [_, _, _, _, _, _, _],
    ]
    start_state = DAGState(0, 0)
    terminal_evaluations = {3: (-1, 1), 4: (-2, 2), 5: (-3, 3), 6: (-4, 4)}
    dag = GameDAG(matrix, start_state, terminal_evaluations)
    return dag


# print(general_minimax(example_dag()))
