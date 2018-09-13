import numpy as np
from collections import defaultdict

def epsilon_greedy_probs(nA, Q_s, i_episode, eps=None):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    epsilon = 1.0 / i_episode
    if eps is not None:
        epsilon = eps
    policy_s = np.ones(nA) * epsilon / nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / nA)
    return policy_s

class Agent:



    def __init__(self, nA=6, gamma=1.0, alpha=0.01):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - Q: state-value function
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        
    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
            
        policy_s = epsilon_greedy_probs(self.nA, self.Q[state], i_episode)
        next_action = np.random.choice(np.arange(self.nA), p=policy_s)
        
        return next_action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        Qsa_next = self.Q[next_state][action]
        Qsa = self.Q[state][action]
        self.Q[state][action] += self.alpha * (reward + (self.gamma * Qsa_next) - Qsa)