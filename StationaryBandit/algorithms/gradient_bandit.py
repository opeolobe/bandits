import numpy as np


# Create a testbed for gradient bandit 
class GradientBandit:
    def __init__(self, num_arms, alpha):

        # number of arms or actions
        self.num_arms = num_arms

        # learning rate
        self.alpha = alpha

        # initial preference
        self.preference = np.zeros(self.num_arms)

        # initial action count
        self.action_counts = 0

        # average reward
        self.average_reward = 0

        # means of the reward distribution == true reward of each arm
        self.means = np.random.normal(0, 1, self.num_arms) 

    
    def select_action(self):

        self.probabilities = np.exp(self.preference) / np.sum(np.exp(self.preference))

        action = np.random.choice(self.num_arms, p= self.probabilities)

        return action
    

    def get_reward(self, action):

        reward = np.random.normal(self.means[action], 1)

        return reward
    
    
    def update_preference_value(self, action, reward):

        # update action count
        self.action_counts += 1

        # update average reward using incremental
        self.average_reward += (reward - self.average_reward) / self.action_counts

        # update preference for each action
        for arm in range(self.num_arms):
            if arm == action:
                self.preference[arm] += self.alpha * (reward - self.average_reward) * (1 - self.probabilities[arm])
            else:
                self.preference[arm] -= self.alpha * (reward - self.average_reward) * self.probabilities[arm]



    def run_experiment(self, num_steps):
             
        optimal_action_per_step = []

        optimal_action_counts_per_step = np.zeros(self.num_arms)
    
        for step in range(num_steps):

            # select an action
            action = self.select_action()

            # obtain reward
            reward = self.get_reward(action= action)
        
            # if action is optimal
            if action == np.argmax(self.means):
                optimal_action_counts_per_step[action] += 1

            # update the preference value
            self.update_preference_value(action= action, reward= reward)

            # percentage optimal action
            optimal_action_percentage = np.sum(optimal_action_counts_per_step)/self.action_counts

            optimal_action_per_step.append(optimal_action_percentage)

        return optimal_action_per_step