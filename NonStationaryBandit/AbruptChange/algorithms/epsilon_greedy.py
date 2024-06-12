import numpy as np


# Build the TestBed Environment using greedy search 
class EpsilonGreedy:
    def __init__(self, num_arms, epsilon, alpha= None):

        # number of arms or actions
        self.num_arms = num_arms  

        # probability of exploration
        self.epsilon = epsilon  

        # fixed step size
        self.alpha = alpha

        # initial action values
        self.q_values = np.zeros(self.num_arms)

        # initial action count
        self.action_counts = np.zeros(num_arms)

        # same starting means for the arms
        self.means = np.random.normal(0, 1, self.num_arms)


    def select_action(self):
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_arms)      # exploration

        else:
            action = np.argmax(self.q_values)    # exploitation

        return action
    
    

    def get_reward(self, action):

        reward = np.random.normal(self.means[action], 1)

        return reward
    

    
    def update_action_value(self, action, reward):

        # update action counts
        self.action_counts[action] += 1

        if self.alpha == None:

            # update action value using incremental sample average
            self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]

        else:

            # update action value using fixed step size
            self.q_values[action] += self.alpha * (reward - self.q_values[action])



    def update_mean(self, probability):

        # simulate non-stationarity by changing the means
        if np.random.random() < probability:
            for idx in range(self.num_arms):
                swap_idx = np.random.randint(self.num_arms)
                self.means[idx], self.means[swap_idx] = self.means[swap_idx], self.means[idx]
        


    
    def run_experiment(self, num_steps):

        average_reward_per_step = []
        cumulative_reward = 0

        for step in range(1, num_steps+1):

            # select an action
            action = self.select_action()

            # obtain reward
            reward = self.get_reward(action= action)
        
            cumulative_reward += reward

            # average reward per step
            average_reward_per_step.append(cumulative_reward / step)

            # update the action value estimate
            self.update_action_value(action= action, reward= reward)

            # update means
            self.update_mean(probability= 0.005)


        # return the terminal average reward

        return average_reward_per_step[-1]