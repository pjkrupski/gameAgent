To run a game, run one of these commands inside gameAgent/1410AdversarialSearch

python gamerunner.py --game=connect4 --dimension=8 --player1=self --player2=self --x 5

python gamerunner.py --game=ttt --dimension=8 --player1=self --player2=self



python gamerunner.py --player1=self --player2=minimax --game=ttt --dimension=8

For connect4

Dimensions can be adjusted and --x is a custom arg for connect 4 to make it instead connect x in a row

The board dim is asserted to be >= x thats given 


For ttt
Dimension arg determines size as well as how many needed in a row to win




python gamerunner.py --player1=self --player2=minimax --game=ttt --dimension=8 --pattern=line


Use .split() instead of passing param?????