import json
import matplotlib.pyplot as plt
import numpy as np
from action import N_ACTION, Action, ActionType
from agent import Agent
from Creabotstate import CreabotState
from moods import N_MOOD, Mood, MoodType
from observation import MoodObservation,Idea,N_IDEA_QUALITY,IdeaQuality
import pandas as pd
from utils import entropy, softmax
import requests

class InteractionModel:
    def __init__(self,cond,config,load=None):
        # logging utility
        self.creabot_config = config

        # -------- Model configurations -------- #
        self.cond = cond
        print("condition :",cond)

        self.reward_crea = self.creabot_config["reward_crea"]
        self.action = Action(ActionType.NEUTRAL)
        self.state = self.get_start_state()
        # -------------Initialize agent------------------#

        self.agent = Agent(
            gamma=self.creabot_config["gamma"],
            wp=self.creabot_config["wp"],
            eps_alpha=0.0001,
            epsilon=self.creabot_config["epsilon"],
            reward_crea=self.reward_crea
        )
        if load != "None":
            self.agent.load("save_models/"+load)

    def new_interaction(self):
        self.time = 0
        self.state = self.get_start_state()

    def get_start_state(self):
        return CreabotState(
            Mood(MoodType.HAPPY),
            ActionType.NEUTRAL,
            Idea(IdeaQuality.NA),
        )

    def update_action(self, state):
        if self.cond == "random":
            self.action = self.agent.get_action_random(state)
        elif self.cond == "adaptive":
            self.action = self.agent.get_action(state)
        elif self.cond == "neutral":
            self.action = self.agent.get_action_neutral(state)
        else:
            raise "Condition unknown"


    def update_state(self,mood_observation,idea_observation):
        self.state = self.agent.update(mood_observation,idea_observation, self.action, self.state)

    def get_action(self):
        return self.action
    def get_state(self):
        return self.state
    def get_belief_state(self):
        return self.agent.belief.belief_proba
