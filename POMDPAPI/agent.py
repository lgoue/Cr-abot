import numpy as np
from action import N_ACTION, Action
from Creabotstate import CreabotState
from moods import N_MOOD, Mood
from utils import softmax,entropy
from Beliefstate import BeliefState
from observation import N_IDEA_QUALITY, Idea

SHAPE_TRANSITION = (N_MOOD,N_IDEA_QUALITY, N_ACTION, N_ACTION, N_MOOD,N_IDEA_QUALITY, N_ACTION )
LEN_TRANSITION = (N_MOOD *N_IDEA_QUALITY* N_ACTION )
SHAPE_SAMPLES = (N_MOOD,N_IDEA_QUALITY, N_ACTION, N_ACTION )
EPS = 0.001
class Agent:
    def __init__(self, gamma=0.2, wp=0.3, eps_alpha=0.0001, epsilon=0,reward_crea=6):

        self.mean_transitions = np.ones(SHAPE_TRANSITION) / LEN_TRANSITION
        # Number of passed interaction
        self.ia = np.ones(SHAPE_SAMPLES)*EPS
        # Coefficient for the importance of creativity reward againt entropy reward
        self.reward_crea = reward_crea
        # Epsilon greedy action selection parameter
        self.epsilon = epsilon

        states = []
        for m in range(N_MOOD):
            for i in range(N_IDEA_QUALITY):
                for a in range(N_ACTION):
                    states.append(CreabotState(Mood(m), a, Idea(i)))
        self.all_states = states

        self.current_user_transitions = np.ones(SHAPE_TRANSITION) / LEN_TRANSITION
        self.current_user_sample = np.ones(SHAPE_SAMPLES)*EPS

        # alpha vector for the QMDP algorithm
        self.alpha = np.zeros(SHAPE_SAMPLES)
        self.previous_alpha = self.alpha.copy()
        # Importance of the mean transition against current_user_transitions
        self.wp = wp
        # Discount factor for the reward
        self.gamma = gamma
        # Difference beetween two iteraction of alpha where they are considered the same
        self.eps_alpha = eps_alpha

        self.actions = [Action(a) for a in range(N_ACTION)]
        self.idea_scores = [Idea(i) for i in range(N_IDEA_QUALITY)]
        self.moods = [Mood(m) for m in range(N_MOOD)]
        self.belief = BeliefState()

        self.update_transition()
        for a in range(N_ACTION):
            for la in range(N_ACTION):
                self.update_alpha(CreabotState(Mood(0),la,Idea(0)),Action(a))

    def load(self,path):
        self.mean_transitions = np.load(path+"_transition.npy" )
        self.ia = np.load(path+"_ia.npy")

    def save(self,path):
        np.save(path+"transition.npy",self.mean_transitions )
        np.save(path+"current_user_transition.npy",self.current_user_transitions )
        np.save(path+"ia.npy",self.ia)
        np.save(path+"current_user_sample.npy",self.current_user_sample)

    def new_user(self):
        self.current_user_transitions = self.mean_transitions.copy()
        self.current_user_sample = np.ones(SHAPE_SAMPLES)*EPS

        self.alpha = np.zeros(SHAPE_SAMPLES)
        self.previous_alpha = self.alpha.copy()

        self.belief = BeliefState()
        self.update_transition()
        for a in range(N_ACTION):
            for la in range(N_ACTION):
                self.update_alpha(CreabotState(Mood(0),la,Idea(0)),Action(a))



    def update(self, mood_observation,idea_observation, action, state):

        next_state = CreabotState(Mood(0),action.bin_number,Idea(0))
        belief = self.belief.next_belief_state(
            mood_observation,idea_observation, action, state, next_state, self.transitions
        )
        belief_proba = belief.belief_proba
        self.update_current_user_transition(
                state, next_state, action, belief_proba
        )

        self.update_transition()
        self.update_alpha(state, action)
        self.belief.belief_proba = belief_proba

        max = belief_proba[0,0]
        m_max,i_max = 0,0
        for m in range(N_MOOD):
            for i in range(N_IDEA_QUALITY):
                if belief_proba[m,i]> max:
                    max = belief_proba[m,i]
                    m_max = m
                    i_max = i

        mood_state = Mood(m_max)
        idea_state = Idea(i_max)

        next_state = CreabotState(
                mood_state,
                next_state.last_strat,
                idea_state,
            )

        return next_state

    def update_current_user_transition(
        self, state, next_state, action, belief_proba
    ):

        state_tuple = state.as_tuple()
        next_state_tuple = next_state.as_tuple()
        max = self.belief.belief_proba[0,0]
        m_max,i_max=0,0
        for previous_m in range(N_MOOD):
            for previous_i, previous_b in enumerate(self.belief.belief_proba[previous_m]):
                if previous_b>max:
                    max = previous_b
                    m_max = previous_m
                    i_max=previous_i
        previous_m = m_max
        previous_i = i_max
        previous_b = max
        self.current_user_transitions[previous_m][previous_i][state_tuple][
                            action.bin_number
                            ] *= self.current_user_sample[previous_m][previous_i][state_tuple][
                            action.bin_number
                            ]
        for previous_m in range(N_MOOD):
            for previous_i, previous_b in enumerate(belief_proba[previous_m]):
                if previous_b>max:
                    max = previous_b
                    m_max = previous_m
                    i_max=previous_i
        m = m_max
        i = i_max
        self.current_user_transitions[previous_m][previous_i][state_tuple][
                                action.bin_number, m,i
                                ][next_state_tuple] = (
        self.current_user_transitions[previous_m][previous_i][state_tuple][
                                action.bin_number, m,i
                                ][next_state_tuple]
                                + previous_b
                                )
        self.current_user_transitions[previous_m][previous_i][state_tuple][action.bin_number] /= (self.current_user_sample[previous_m][previous_i][state_tuple][action.bin_number]
                                + previous_b
                                )
        self.current_user_sample[previous_m][previous_i][state_tuple][
                                action.bin_number
                                ] += previous_b



    def update_mean_transition(self):

        for m in range(N_MOOD):
            for i in range(N_IDEA_QUALITY):
                for a in range(N_ACTION):
                    for at in range(N_ACTION):
                        self.mean_transitions[m,i,a,at] = (
            self.ia[m,i,a,at] * self.mean_transitions[m,i,a,at].copy()
            + self.current_user_transitions[m,i,a,at].copy()*self.current_user_sample[m,i,a,at]
        ) / (self.ia[m,i,a,at] + self.current_user_sample[m,i,a,at])

    def update_transition(self):

        self.transitions = (
            self.wp * self.current_user_transitions.copy()
            + (1 - self.wp) * self.mean_transitions.copy()
        )


    def get_action_random(self,state):
        return self.actions[np.random.randint(len(self.actions))]
    def get_action_neutral(self,state):
        return self.actions[3]
    def get_action(self, state):

        self.Q = np.zeros(N_ACTION)
        for a in self.actions:
            for m in range(N_MOOD):
                for i in range(N_IDEA_QUALITY):
                    self.Q[a.bin_number] += (
                            self.belief.belief_proba[m,i]
                            * self.alpha[m][i][state.as_tuple()][a.bin_number]
                            )
        print(self.Q)
        r = np.random.rand()
        p=softmax(0.3*self.Q)
        for i in range(len(self.actions)):
            if r < np.sum(p[:i+1]):
                return self.actions[i]

        if r < self.epsilon:
            return self.actions[np.random.randint(N_ACTION)]
        else:
            return self.actions[np.argmax(softmax(0.3*self.Q))]

    def update_alpha(self, state, action):

        diff_alpha = 100
        while diff_alpha > self.eps_alpha:

            self.previous_alpha = self.alpha.copy()
            for mood in self.moods:
                for idea in self.idea_scores:
                    a = 0
                    s = state.copy()
                    s.mood = mood
                    s.idea_score = idea

                    for m in range(N_MOOD):
                        for i in range(N_IDEA_QUALITY):
                            sp =CreabotState(Mood(m), action.bin_number, Idea(i))

                            a += self.transitions[s.mood.bin_number][s.idea_score.bin_number][s.as_tuple()][
                                    action.bin_number, sp.mood.bin_number,sp.idea_score.bin_number
                                    ][sp.as_tuple()] * (
                                    (sp.idea_score.quality*self.reward_crea - self.reward_crea*100*((sp.last_strat == s.last_strat) & (s.last_strat!=0)))
                                            + self.gamma
                                            * np.max(self.alpha[sp.mood.bin_number][sp.idea_score.bin_number][sp.as_tuple()])
                                            )


                        self.alpha[s.mood.bin_number][s.idea_score.bin_number][state.as_tuple()][
                                action.bin_number
                            ] = a#- np.sum(
                                #entropy(
                                #    self.transitions[s.mood.bin_number][s.idea_score.bin_number][s.as_tuple()][
                                #        action.bin_number
                                #        ]
                                #        )
                                #        )
            diff_alpha = np.linalg.norm(self.alpha - self.previous_alpha)
