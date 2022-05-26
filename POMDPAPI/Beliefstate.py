import numpy as np
from utils import softmax
from moods import N_MOOD,Mood
from observation import N_IDEA_QUALITY, Idea

SHAPE_BELIEF_STATE = (N_MOOD,N_IDEA_QUALITY)
LEN_BELIEF_STATE = N_IDEA_QUALITY*N_MOOD

class BeliefState:
    """
    Implement the belief state on mood and idea quality

    """

    def __init__(self, belief_proba=None):
        if belief_proba is None:
            self.belief_proba = np.ones(SHAPE_BELIEF_STATE) / LEN_BELIEF_STATE
        else:
            self.belief_proba = belief_proba

        self.idea_scores = [Idea(i) for i in range(N_IDEA_QUALITY)]
        self.moods = [Mood(m) for m in range(N_MOOD)]

    def distance_to(self, other_state):
        distance = np.sum(np.abs(self.belief_proba - other_state.belief_proba))
        return distance

    def copy(self):
        return BeliefState(belief_proba=self.belief_proba.copy())

    def equals(self, other_state):
        if self.belief_proba == other_state.belief_proba:
            return 1
        else:
            return 0


    def to_string(self):
        max = 0
        m_max = 0
        i_max = 0
        for m in range(N_MOOD):
            for i in range(N_IDEA_QUALITY):
                if self.belief_proba[m,i]> max:
                    max = self.belief_proba[m,i]
                    m_max = m
                    i_max = i
        state_string = "The most probable idea score is "
        state_string += self.idea_scores[i_max].to_string()
        state_string = "The most probable mood is "
        state_string += self.moods[m_max].to_string()
        return state_string

    def print_state(self):
        """
        Pretty printing
        :return:
        """
        print(self.to_string())

    def next_belief_state(
        self, observation_mood,observation_idea, action, current_state, next_state, transition
    ):
        Om = softmax([-m.distance_to_observation(observation_mood) for m in self.moods])
        Oi = softmax([-e.distance_to_observation(observation_idea) for e in self.idea_scores])

        B = np.zeros(SHAPE_BELIEF_STATE)
        for m, om in enumerate(Om):
            for i, oi in enumerate(Oi):
                temp = 0
                for ei in self.idea_scores:
                    for em in self.moods:
                        temp += (
                            transition[em.bin_number][ei.bin_number][current_state.as_tuple()][
                            action.bin_number, m,i
                            ][next_state.as_tuple()]
                            * self.belief_proba[em.bin_number,ei.bin_number]
                )
                B[m,i]=(temp*om*oi )

        B = B / (np.sum(B)+0.0001)
        
        return BeliefState(belief_proba=B)
