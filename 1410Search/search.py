# NOTE TO STUDENT: Please read the handout before continuing.

from queue import LifoQueue, PriorityQueue, Queue
from typing import Callable, List

from dgraph import DGraph
from searchproblem import SearchProblem, State
from tilegameproblem import TileGame, TileGameState


### GENERAL SEARCH IMPLEMENTATIONS - NOT SPECIFIC TO THE TILEGAME PROBLEM ###


def bfs(problem: SearchProblem[State]) -> List[State]:
    """
    Implement breadth-first search.

    Input:
        problem - the problem on which the search is conducted, a SearchProblem

    Output: a list of states representing the path of the solution

    """
    start = problem.get_start_state()
    visited = {start: start}
    q = Queue()
    q.put(start)
    while not q.empty():
        node = q.get()
        # if goal state is found, retrace path
        if problem.is_goal_state(node):
            path = [node]
            while node != start:
                node = visited[node]
                path.insert(0, node)
            return path
        # add children to queue
        else:
            successors = problem.get_successors(node)
            for child in successors:
                if child not in visited:
                    visited[child] = node
                    q.put(child)
    return []


def dfs(problem: SearchProblem[State]) -> List[State]:
    """
    Implement depth-first search.

    Input:
        problem - the problem on which the search is conducted, a SearchProblem

    Output: a list of states representing the path of the solution

    """
    start = problem.get_start_state()
    visited = {start: start}
    stack = LifoQueue()
    stack.put(start)
    while not stack.empty():
        node = stack.get()
        # if goal state is found, retrace path
        if problem.is_goal_state(node):
            path = [node]
            while node != start:
                node = visited[node]
                path.insert(0, node)
            return path
        # add children to stack
        else:
            successors = problem.get_successors(node)
            for child in successors:
                if child not in visited:
                    visited[child] = node
                    stack.put(child)
    return []


def ids(problem: SearchProblem[State]) -> List[State]:
    """
    Implement iterative deepening search.

    Input:
        problem - the problem on which the search is conducted, a SearchProblem

    Output: a list of states representing the path of the solution

    """
    start = problem.get_start_state()
    depth = 0

    while True:
        depth_visited = {start: 0}
        frontier = [(start, int(), tuple())]

        while frontier:
            popped_val = frontier.pop()
            node, d, _ = popped_val
            if problem.is_goal_state(node):
                return back_track(popped_val)
            if d < depth:
                for child in problem.get_successors(node):
                    if depth_visited.get(child, float("inf")) > d + 1:
                        depth_visited[child] = d + 1
                        frontier.append((child, d + 1, popped_val))
        depth += 1


def back_track(node):
    path = []
    while node is not tuple():
        child, _, parent = node
        path.insert(0, child)
        node = parent

    return path


def bds(problem: SearchProblem[State], goal: State) -> List[State]:
    """
    Implement bi-directional search.

    The input 'goal' is a goal state (not a search problem, just a state)
    from which to begin the search toward the start state.

    Assume that the input search problem can be thought of as
    an undirected graph. That is, all actions in the search problem
    are reversible.

    Input:
        problem - the problem on which the search is conducted, a SearchProblem
        goal - the goal state, a state

    Output: a list of states representing the path of the solution

    """
    start = problem.get_start_state()
    start_queue: Queue[State] = Queue()
    goal_queue: Queue[State] = Queue()
    start_queue.put(start)
    goal_queue.put(goal)
    start_visited = {start: start}
    goal_visited = {goal: goal}
    start_depths = {start: 0}
    goal_depths = {goal: 0}
    start_last_depth = 0
    goal_last_depth = 0
    shared_node = start

    while not start_queue.empty() and not goal_queue.empty():
        # BFS from start state to goal state
        if not start_queue.empty():
            node = start_queue.get()
            start_last_depth = start_depths[node]
            # stop when goal state is found or shared node is found
            if problem.is_goal_state(node) or node in goal_visited:
                shared_node = node
                break
            for child in problem.get_successors(node):
                if child not in start_visited:
                    start_visited[child] = node
                    start_queue.put(child)
                    start_depths[child] = start_depths[node] + 1

        # BFS from goal state to start state
        if not goal_queue.empty():
            node = goal_queue.get()
            goal_last_depth = goal_depths[node]
            # stop when start state is found or shared node is found
            if start == node or node in start_visited:
                shared_node = node
                break
            for child in problem.get_successors(node):
                if child not in goal_visited:
                    goal_visited[child] = node
                    goal_queue.put(child)
                    goal_depths[child] = goal_depths[node] + 1

    # exhaust start_queue and goal_queue at the depth up to the shared_node
    # this ensures we find the optimal path
    shortest_path_length = start_depths[shared_node] + goal_depths[shared_node]
    while not start_queue.empty():
        node = start_queue.get()
        if start_depths[node] > start_last_depth:
            break
        if node in goal_depths:
            path_length = start_depths[node] + goal_depths[node]
            if path_length < shortest_path_length:
                shared_node = node
                shortest_path_length = path_length
    while not goal_queue.empty():
        node = goal_queue.get()
        if goal_depths[node] > goal_last_depth:
            break
        if node in start_depths:
            path_length = start_depths[node] + goal_depths[node]
            if path_length < shortest_path_length:
                shared_node = node
                shortest_path_length = path_length

    path = [shared_node]
    # retrace path from shared node to start
    node_a = shared_node
    while node_a != start:
        node_a = start_visited[node_a]
        path.insert(0, node_a)
    # retrace path from shared node to goal state
    node_b = shared_node
    while node_b != goal:
        node_b = goal_visited[node_b]
        path.append(node_b)
    return path


def astar(problem: SearchProblem[State], heur: Callable[[State], float]) -> List[State]:
    """
    Implement A* search.

    The given heuristic function will take in a state of the search problem
    and produce a real number.

    Your implementation should be able to work with any heuristic
    that is for the given search problem (but, of course, without a
    guarantee of optimality if the heuristic is not admissible).

    Input:
        problem - the problem on which the search is conducted, a SearchProblem
        heur - a heuristic function that takes in a state as input and outputs a number
    Output: a list of states representing the path of the solution
    """
    start = problem.get_start_state()
    pq = PriorityQueue()
    pq.put((0, start))
    parent = {start: None}
    cost_to_node = {start: 0.0}

    while not pq.empty():
        node = pq.get()[1]
        if problem.is_goal_state(node):
            path = []
            while node is not None:
                path.insert(0, node)
                node = parent[node]
            return path

        for (child, hop_cost) in problem.get_successors(node).items():
            cost = cost_to_node[node] + hop_cost
            if child not in cost_to_node or cost < cost_to_node[child]:
                cost_to_node[child] = cost_to_node[node] + hop_cost
                parent[child] = node
                pq.put((cost + heur(child), child))
    return []


### SPECIFIC TO THE TILEGAME PROBLEM ###


def tilegame_heuristic(state: TileGameState) -> float:
    """
    Produces a number for the given tile game state representing
    an estimate of the cost to get to the goal state. Remember that this heuristic must be
    admissible, that is it should never overestimate the cost to reach the goal.
    Input:
        state - the tilegame state to evaluate. Consult handout for how the tilegame state is represented

    Output: a float.

    """
    total = 0
    for row in range(3):
        for col in range(3):
            num = state[row][col] - 1
            real_row = num // 3
            real_col = num % 3
            total += abs(real_row - row) + abs(real_col - col)
    return total // 2


### YOUR SANDBOX ###


def main():
    """
    Do whatever you want in here; this is for you.
    The examples below shows how your functions might be used.
    """

    # initialize a random 3x3 TileGame problem
    tg = TileGame(3)
    # print(TileGame.board_to_pretty_string(tg.get_start_state()))
    # compute path using dfs
    path = dfs(tg)
    # display path
    TileGame.print_pretty_path(path)

    # initialize a small DGraph
    small_dgraph = DGraph([[None, 1], [1, None]], {1})
    # print the path using ids
    print(ids(small_dgraph))


if __name__ == "__main__":
    main()
