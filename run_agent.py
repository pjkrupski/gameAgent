from adversarialsearchproblem import AdversarialSearchProblem
from bots import StudentBot, StudentBot2, RandomBot, MinmaxBot
from tttproblem import TTTProblem, TTTUI
from tttproblem import TTTProblem, TTTUI
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt 
import argparse
import numpy as np 


def run_game(asp: AdversarialSearchProblem, bots, game_ui=None):
    """
    Inputs:
            - asp: a game to play, represented as an adversarial search problem
            - bots: a list in which the i'th element is adversarial search
                    algorithm that player i will use.
                    The algorithm must take in an ASP only and output an action.
            - game_ui (optional): a GameUI that visualizes ASPs and allows for
                    direct input in place of a bot that is None. If no argument is
                    passed, run_game() will not be interactive.
    Output:
            - the evaluation of the terminal state.
    """

    # Ensure game_ui is present if a bot is None:
    if not game_ui and any(bot is None for bot in bots):
        raise ValueError("A GameUI instance must be provided if any bot is None.")

    state = asp.get_start_state()
    if game_ui:
        game_ui.update_state(state)
        game_ui.render()

    while not (asp.is_terminal_state(state)):
        curr_bot = bots[state.player_to_move()]

        # Obtain decision from the bot itself, or from GameUI if bot is None:
        if curr_bot:
            decision = curr_bot.decide(asp)

            # If the bot tries to make an invalid action,
            # returns any valid action:
            available_actions = asp.get_available_actions(state)
            if decision not in available_actions:
                decision = available_actions.pop()
        else:
            decision = game_ui.get_user_input_action()

        result_state = asp.transition(state, decision)
        asp.set_start_state(result_state)
        state = result_state

        if game_ui:
            game_ui.update_state(state)
            game_ui.render()

        if asp.is_terminal_state(state): # and bots[0].games%100 == 0:
            print(asp.board_to_pretty_string(state.board))

    tup = asp.evaluate_terminal(state)

    bots[0].cleanup(tup[0] == 1.0)

    return asp.evaluate_terminal(asp.get_start_state())

def main():
    """
    Provides an example of the TTTProblem class being used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=["ttt", "custom"], default="ttt")
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument(
        "--player1", choices=["self", "minimax", "bot", "random"], default="bot"
    )
    parser.add_argument(
        "--player2", choices=["self", "minimax", "bot", "random"], default="minimax"
    )
    parser.add_argument("--pattern", choices=["l", "line", "t", "v"], default="line")
    parser.add_argument("--gameNum", type=int, default=1000)
    args = parser.parse_args()
    player_args = [args.player1, args.player2]


    # Assign players:
    players = [None, None]
    algorithm_dict = {
        "self": None,
        "minimax": MinmaxBot(),
        "bot": StudentBot(),
        "random": RandomBot()
    } 
    for i, player in enumerate(player_args):
        players[i] = algorithm_dict.get(player)

    ### Game: Tic-Tac-Toe
    if args.game == "ttt":
        if args.dimension is not None:
            if args.dimension < 3:
                parser.error("--dimension must be at least 3 for Tic-Tac-Toe")
            #TODO
            #Pass custom arg in game instantiation 
            game = TTTProblem(args.pattern, dim=args.dimension)
        else:
            game = TTTProblem()
        game_ui = TTTUI(game)

    ### Game: Custom
    # if args.game == "custom":
    #     game, game_ui = get_custom_asp(args)

    #random_trained = load_model("1000_line_vs_random")
  
    games = args.gameNum
    for i in range(games):
      t = TTTProblem()
      run_game(t, players)

    #players[0].model.save_weights("1000_line_vs_random_weights")
    print("saved model")

    

    
   # for player in players:
   #     if player is StudentBot:
   #         print("saving model")
   #         player.model.save("1000_line_vs_random")

    #print(s.graph)

if __name__ == "__main__":
    main()