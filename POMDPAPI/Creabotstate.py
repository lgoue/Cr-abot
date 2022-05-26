import numpy as np
from action import Action, ActionType
from moods import N_MOOD, Mood
from observation import Idea, N_IDEA_QUALITY


class CreabotState:
    """
    Enumerated state for the Creabot POMDP

    Consists of  :
        - A int "last agent action" wich is *fully* observable
        - The "user mood" wich is "obscured"
        - The "user idea quality" wich is "obscured"


         A single CreabotState represents a
    "guess" of the true belief state - which is the probability distribution over all states

    """

    def __init__(self, mood, last_strat, idea_score):

        self.mood = mood  # object of type mood
        self.last_strat = last_strat
        self.idea_score = idea_score  # Object of type Idea


    def as_tuple(self):
        return self.last_strat

    def copy(self):

        return CreabotState(
            self.mood,
            self.last_strat,
            self.idea_score,
        )

    def to_string(self):
        state_string = ""
        state_string += self.mood.to_string()
        state_string += " - "
        state_string += str(Action(self.last_strat).to_string())
        state_string += " - "
        state_string += self.idea_score.to_string()
        state_string += " - "

        return state_string

    def print(self):
        """
        Pretty printing
        :return:
        """
        print(self.to_string())
