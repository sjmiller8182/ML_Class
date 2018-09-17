import gym
import numpy as np
import matplotlib.pyplot as plt

class Agent_Tuner:

    def __init__(self, gym_env):
        """ Initialize agent.

        Params
        ======
        - gym_env: the AI gym environment to act on.
        """
        self.gym_env = gym_env
		self.tuning_results = list()
		self.best_parameters = {'gamma':-1, 'alpha':-1}
		self.has_been_run = False

	def tune(self, agent, interact, trials, gammas = [0.5], alphas = [0.01]):
		"""
		Run the agent over number of trials with alphas and gammas

        Params
        ======
        - agent: agent class definition
		- interact: episode interaction definition
		- trials: the number of trials to run
		- gammas: list of gammas for tuning
		- alphas: list of alphas for tuning
		"""
		self.has_been_run = True
		
		# init vars
		results = [dict(), dict()]
		
		# run all gammas and alphas for the number of trials
		for g in gammas:
			for a in alphas:
				key = str(g) + '_' + str(a)
				env = gym.make(self.gym_env)
				agent = Agent(gamma = g, alpha = a)
				avg_rewards, best_avg_reward = interact(env, agent, num_episodes=trials)
				results[0][key] = avg_rewards
				results[1][key] = best_avg_reward
		
		# saving tuning results
		self.tuning_results = results
		
	def plot_learning_curves():
		"""
		plot agents reward over episodes
		"""
		if not self.has_been_run:
			print("Tuning needs to be run first. Results are invalid.")
			
		d = self.turning_results[0]
		fig, ax = plt.subplots();
		for key in d.keys():
			td = d[key]
			label = 'g: ' + key.split('_')[0] + ' a: ' + key.split('_')[1]
			ax.plot(np.arange(len(td)), td, label = label);
		ax.set_title("Learning Over Iterations for Gammas");
		ax.set_xlabel("Agent Steps");
		ax.set_ylabel("100 Step Average Reward");
		ax.legend();
			
	
	def plot_results():
		"""
		plot bar chart of gamma/alpha combinations
		"""
		pass
	
	def get_best_params():
		"""
		returns the alpha and gamma from the best run in a dict
		"""
		if not self.has_been_run:
			print("Tuning needs to be run first. Results are invalid.")
		return self.best_parameters
	
	