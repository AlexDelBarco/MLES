import numpy as np

class MaintenanceProblem:
    def __init__(self,gamma=1, P_sa=None,REWARDS=None,VALID_ACTIONS=None):
        self.gamma = gamma
        self.P_sa = P_sa
        self.REWARDS = REWARDS
        self.VALID_ACTIONS = VALID_ACTIONS

    # Define the actions and states
    ACTIONS = np.array(['Ignore', 'Maintain', 'Repair'])
    STATES = np.array(['Good', 'Deteriorating', 'Broken'])
    
    # Value Iteration

    def value_iteration(self, epsilon=1e-6):
        # Initialize the value function with V(s) = 0
        V = np.zeros(len(self.STATES))
        while True:
            # Initialize the new value function
            V_new = np.zeros(len(self.STATES))
            for s in range(len(self.STATES)):
                # Get the valid actions for the current state
                valid_actions = self.VALID_ACTIONS[s]
                # Extract the rewards and transitions for the current state
                rewards = self.REWARDS[s]
                transitions = self.P_sa[s * len(self.ACTIONS):(s + 1) * len(self.ACTIONS)]
                # Update the value function
                V_new[s] = np.max([rewards[a] + self.gamma * np.sum(transitions[a] * V) for a in valid_actions])
            # Check for convergence
            if np.max(np.abs(V - V_new)) < epsilon:
                break
            V = V_new
        return V

    def policy_from_value(self, V):
        # Initialize the policy with a random action for each state
        policy = np.zeros(len(self.STATES), dtype=int)
        for s in range(len(self.STATES)):
            # Get the valid actions for the current state
            valid_actions = self.VALID_ACTIONS[s]
            # Initialize the action values with -inf
            action_values = np.ones(len(self.ACTIONS)) * (-np.inf)
            # Extract the rewards and transitions for the current state
            rewards = self.REWARDS[s]
            transitions = self.P_sa[s * len(self.ACTIONS):(s + 1) * len(self.ACTIONS)]
            # Compute the value of each action
            for a in valid_actions:
                action_values[a] = rewards[a] + self.gamma * np.sum(transitions[a] * V)
            # Select the action with the highest value
            policy[s] = np.argmax(action_values)
        return policy

    # Policy Iteration

    def _evaluate_policy(self, policy):
        # Initialize rewards and transitions for each state
        transitions = np.zeros((len(self.STATES), len(self.STATES)))
        rewards = np.zeros(len(self.STATES))    
        for s in range(len(self.STATES)):
            # Extract the rewards and transitions for the current state
            rewards[s] = self.REWARDS[s,policy[s]]
            transitions[s] = self.P_sa[s*len(self.STATES)+policy[s]]
        # Update the value function
        V = np.linalg.inv(np.eye(len(self.STATES)) - self.gamma * transitions).dot(rewards)
        return V

    def policy_iteration(self):
        # Initialize the policy with a random action for each state
        policy = np.random.randint(0, len(self.ACTIONS), len(self.STATES))
        while True:
            # Evaluate the current policy
            V = self._evaluate_policy(policy)
            new_policy = np.zeros(len(self.STATES), dtype=int)
            for s in range(len(self.STATES)):
                valid_actions = self.VALID_ACTIONS[s]
                # Initialize the action values with -inf
                action_values = np.ones(len(self.ACTIONS)) * (-np.inf)
                # Extract the rewards and transitions for the current state
                rewards = self.REWARDS[s]
                transitions = self.P_sa[s * len(self.ACTIONS):(s + 1) * len(self.ACTIONS)]
                # Compute the value of each action
                for a in valid_actions:
                    action_values[a] = rewards[a] + self.gamma * np.sum(transitions[a] * V)
                # Select the action with the highest value and update the policy
                new_policy[s] = np.argmax(action_values) 
            # Check for convergence   
            if np.array_equal(policy, new_policy):
                break
            policy = new_policy
        return policy, V

def main():
    # Define the rewards, where we just use 0 for impossible transitions, as they will be ignored anyway
    rewards = np.array([[0, -1, 0],     # Good
                        [0, -1, 0],     # Deteriorating
                        [-2, 0, -10]])  # Broken
    
    # Define valid actions for each state (e.g. maintain in the broken state is not valid)
    valid_actions = np.array([[0,1], # Good
                              [0,1],   # Deteriorating
                              [0,2]])  # Broken

    # Define the transition probabilities
    P_sa = np.array([[0.96, 0.04, 0.0], # Good, Ignore
                     [1.0, 0.0, 0.0],    # Good, Maintain
                     [0.0, 0.0, 0.0],    # Good, Repair
                     [0.0, 0.9, 0.1],    # Deteriorating, Ignore
                     [0.95, 0.04, 0.01], # Deteriorating, Maintain
                     [0.0, 0.0, 0.0],    # Deteriorating, Repair
                     [0.0, 0.0, 1.0],    # Broken, Ignore
                     [0.0, 0.0, 0.0],    # Broken, Maintain
                     [1.0, 0.0, 0.0]])   # Broken, Repair 

    # Create the maintenance problem
    example = MaintenanceProblem(0.9, P_sa, rewards, valid_actions)

    # Value Iteration
    print("Value Iteration:")
    optimal_values = example.value_iteration()
    optimal_policy = example.policy_from_value(optimal_values)
    for i, state in enumerate(example.STATES):
        print(f"{state:<15} {optimal_values[i]:<10.2f} {example.ACTIONS[optimal_policy[i]]}")

    # Policy Iteration
    print("\nPolicy Iteration:")
    optimal_policy, optimal_values = example.policy_iteration()
    for i, state in enumerate(example.STATES):
        print(f"{state:<15} {optimal_values[i]:<10.2f} {example.ACTIONS[optimal_policy[i]]}")

if __name__ == "__main__":
    main()