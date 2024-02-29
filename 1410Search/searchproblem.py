from abc import ABC, abstractmethod
from typing import Dict, Generic, Hashable, TypeVar

# In SearchProblem, we require that all states are hashable so that we can
# represent successive states as a dictionary.
#
# ints and strings are hashable, as well as most of Python's immutable
# built-in data structures (e.g. tuples, but not lists or dictionaries)

State = TypeVar("State", bound=Hashable)


class SearchProblem(ABC, Generic[State]):
    @abstractmethod
    def get_start_state(self) -> State:
        """
        Produces the state from which to search.
        """
        pass

    @abstractmethod
    def is_goal_state(self, state: State) -> bool:
        """
        Produces the boolean that the given state, state, is a goal state.
        """
        pass

    @abstractmethod
    def get_successors(self, state: State) -> Dict[State, float]:
        """
        Produces a dictionary, whose keys are the states that can be reached
        from the given state and whose values are the costs of reaching
        each associated key.
        """
        pass
