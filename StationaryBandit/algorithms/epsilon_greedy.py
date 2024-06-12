import numpy as np


# Build the TestBed Environment using epsilon-greedy search 
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

        # means of the reward distribution == true reward of each arm
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



    
    def run_experiment(self, num_steps):

        average_reward_per_step = []
        optimal_action_per_step = []
        optimal_action_counts_per_step = np.zeros(self.num_arms)
        cumulative_reward = 0

        for step in range(1, num_steps+1):

            # select an action
            action = self.select_action()

            # obtain reward
            reward = self.get_reward(action= action)
        
            # if action is optimal
            if action == np.argmax(self.means):
                optimal_action_counts_per_step[action] += 1

            cumulative_reward += reward

            # average reward per step
            average_reward_per_step.append(cumulative_reward / step)

            # update the action value estimate
            self.update_action_value(action= action, reward= reward)

            # percentage optimal action
            optimal_action_percentage = np.sum(optimal_action_counts_per_step)/np.sum(self.action_counts)
            optimal_action_per_step.append(optimal_action_percentage)
            

        return average_reward_per_step, optimal_action_per_step