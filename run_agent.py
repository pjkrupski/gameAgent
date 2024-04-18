from adversarialsearchproblem import AdversarialSearchProblem
from bots import StudentBot, RandomBot
from tttproblem import TTTProblem

import matplotlib.pyplot as plt 
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

    tup = asp.evaluate_terminal(state)

    bots[0].cleanup(tup[0] == 1.0)

    return asp.evaluate_terminal(asp.get_start_state())

def main():
    """
    Provides an example of the TTTProblem class being used.
    """
    s = StudentBot()
    r = RandomBot()

    games = 10000
    for i in range(games):
      t = TTTProblem()
      run_game(t, [s, r])

      if i%100 == 0: 
        print(t.board_to_pretty_string(t.get_start_state().board))

    print(s.graph)
    plt.plot(np.arange(int(games/100)), s.graph)
    plt.savefig("plot.png")

if __name__ == "__main__":
    main()