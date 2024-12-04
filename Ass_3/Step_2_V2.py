#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:11:31 2024

@author: salomeaubri
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gb
from gurobipy import GRB

pd.set_option('mode.chained_assignment', None)

nb_state_price = 5
nb_state_SOC = 6 # from 0 to 500 MWh with a 100MWh step

SOC_states = range(nb_state_SOC)  # Iterable range of SOC states
price_states = range(nb_state_price)  # Iterable range of price states

########### Step 2 & 3: Code ##############
def load_prices(nb_price_ranges, market_zone):
    """
    Load and discretize prices into quantile-based ranges, storing the mean price for each range.

    Parameters:
        nb_price_ranges (int): The number of price ranges to create.
        market_zone (string): The area of interest.

    Returns:
        prices_area (pd.DataFrame): The DataFrame with a 'price_range' column.
        quantiles (dict): A dictionary where keys are range labels and values are mean prices for each range.
    """

    # Load the Excel file
    prices = pd.read_excel("Price.xlsx")

    # Filter by area
    prices_area = prices[prices['PriceArea'] == market_zone].copy()  # Use .copy() to avoid warnings
    prices_area.reset_index(drop=True, inplace=True)
    
    prices_area['Month'] = prices['HourDK'].dt.month
    prices_area['Year'] = prices['HourDK'].dt.year
    
    # Generate quantiles based on the number of ranges
    prices_ranges = np.linspace(0, 1, num=nb_price_ranges + 1)

    # Dynamically generate labels
    labels = [f"Range {i+1}" for i in range(nb_price_ranges)]

    # Discretize prices into ranges
    prices_area.loc[:, 'price_range'] = pd.qcut(
        prices_area['PriceEUR'], 
        q=prices_ranges, 
        labels=labels, 
        duplicates='drop'
    )

    # Calculate the mean price for each range
    quantiles = (
        prices_area.groupby('price_range')['PriceEUR']
        .mean()
        .to_dict()
    )

    return prices_area, quantiles


def get_price(state, quantiles):

    category_price = state[1]

    # Return the quantile range for the given category
    return quantiles[category_price]


class BatteryProblem:
    def __init__(self,gamma=1, P_sa=None,REWARDS=None,VALID_ACTIONS=None):
        self.gamma = gamma
        self.P_sa = P_sa
        self.REWARDS = REWARDS
        self.VALID_ACTIONS = VALID_ACTIONS

    # Define the actions and states
    ACTIONS = np.array(['Inactive', 'Charge', 'Discharge'])
    STATES = np.array([(s, "Range {0}".format(p+1)) for s in SOC_states for p in price_states])
    
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
                transitions = self.P_sa[s + np.array(range(len(self.ACTIONS))) * len(self.STATES)]
                # Update the value function
                action_index_map = {'Inactive': 0, 'Charge': 1, 'Discharge': 2}  # Map actions to correct indexes

                # Calculate V_new[s]
                V_new[s] = np.max([
                    rewards[action_index_map[action]] + self.gamma * np.sum(transitions[action_index_map[action]] * V)
                    for action in valid_actions
                ])
    
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
            transitions = self.P_sa[s + np.array(range(len(self.ACTIONS))) * len(self.STATES)]
            # Compute the value of each action
            action_index_map = {'Inactive': 0, 'Charge': 1, 'Discharge': 2}  # Map actions to correct indexes

            for action in valid_actions :
                action_values[action_index_map[action]] = rewards[action_index_map[action]] + self.gamma * np.sum(transitions[action_index_map[action]] * V)
            
            # Select the action with the highest value
            policy[s] = np.argmax(action_values)
        return policy

    # Policy Iteration        
    def _evaluate_policy(self, policy):
        """
        Evaluate the current policy by solving the Bellman equation:
            V = (I - gamma * P_pi)^-1 * R_pi
        """
        num_states = len(self.STATES)
        transitions = np.zeros((num_states, num_states))  # Transition matrix under policy
        rewards = np.zeros(num_states)  # Reward vector under policy
        
        for s in range(num_states):
            action = policy[s]
            rewards[s] = self.REWARDS[s, action]
            transitions[s] = self.P_sa[action * num_states + s]
        
        # Solve Bellman equation to get V
        I = np.eye(num_states)
        V = np.linalg.solve(I - self.gamma * transitions, rewards)
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
                transitions = self.P_sa[s + np.array(range(len(self.ACTIONS))) * len(self.STATES)]
                # Compute the value of each action
                action_index_map = {'Inactive': 0, 'Charge': 1, 'Discharge': 2}  # Map actions to correct indexes

                for action in valid_actions :
                    action_values[action_index_map[action]] = rewards[action_index_map[action]] + self.gamma * np.sum(transitions[action_index_map[action]] * V)
                    
                # Select the action with the highest value and update the policy
                new_policy[s] = np.argmax(action_values) 
            # Check for convergence   
            if np.array_equal(policy, new_policy):
                break
            policy = new_policy
        return policy, V


def main():
    
    market_zone = 'DK2' # Choose between DK1, DK2, DE, SO3, SO4, NO2, SYSTEM
    prices, quantiles = load_prices(nb_state_price, market_zone)
    
    ACTIONS = np.array(['Inactive', 'Charge', 'Discharge'])
    STATES = np.array([(s, "Range {0}".format(p+1)) for s in SOC_states for p in price_states])

    
    def reward_matrix(quantiles, STATES, ACTIONS):
        rewards = np.zeros((len(STATES), len(ACTIONS)))
        delta_soc = 500 / (nb_state_SOC - 1)
        for s, state in enumerate(STATES):
            soc, price_range = state
            price = quantiles.get(price_range, 0)  # Default to 0 if range is invalid
            for a, action in enumerate(ACTIONS):
                if action == 'Inactive':
                    rewards[s][a] = 0
                elif action == 'Discharge':
                    rewards[s][a] = price * delta_soc  # Positive revenue for discharge
                elif action == 'Charge':
                    rewards[s][a] = -price * delta_soc  # Negative cost for charging
        return rewards

     
    def valid_action_matrix(STATES, ACTIONS):
        valid_actions = []
        for s in range(len(STATES)):
            list_action = []
            for a in ACTIONS:
                if s in [i for i in range(len(quantiles))]: # when the battery SOC is 0
                    if a != 'Discharge':
                        list_action.append(a)
                elif s in sorted([len(STATES) - i-1 for i in range(len(quantiles))]): # when the battery SOC is 500
                    if a!= 'Charge':
                        list_action.append(a)
                else:
                    list_action.append(a)
            valid_actions.append(list_action)
        return valid_actions
    
    
    def count_transitions(prices_area):

        unique_ranges = prices_area['price_range'].unique()
        unique_ranges = sorted(unique_ranges, key=lambda x: int(x.split()[-1]))  # Ensure proper range ordering
        transitions = {f"{src}-{dst}": 0 for src in unique_ranges for dst in unique_ranges}
    
        prev_range = None
        for current_range in prices_area['price_range']:
            if prev_range is not None:
                key = f"{prev_range}-{current_range}"
                transitions[key] += 1
            prev_range = current_range
    
        return transitions
    
    
    def probability_matrix(prices_area, transitions):
    
        # Get unique price ranges and sort them numerically
        unique_ranges = prices_area['price_range'].unique()
        unique_ranges = sorted(unique_ranges, key=lambda x: int(x.split()[-1]))  # Ensure proper range ordering
    
        # Count total occurrences of each range
        totals = prices_area['price_range'].value_counts().reindex(unique_ranges, fill_value=0)
    
        # Initialize probability matrix
        prob_matrix = pd.DataFrame(0, index=unique_ranges, columns=unique_ranges)
    
        # Fill the matrix with transition probabilities
        for src in unique_ranges:
            for dst in unique_ranges:
                key = f"{src}-{dst}"
                if totals[src] > 0:  # Avoid division by zero
                    prob_matrix.loc[src, dst] = transitions[key] / totals[src]
    
        return prob_matrix
    
    
    def create_combined_matrix(original_matrix, nb_state_SOC, nb_state_price):
        """
        Construct the full transition probability matrix for all states and actions.
        """
        original_matrix = np.array(original_matrix)
        len_states = nb_state_SOC * nb_state_price
        matrices = {
            'Inactive': np.zeros((len_states, len_states)),
            'Charge': np.zeros((len_states, len_states)),
            'Discharge': np.zeros((len_states, len_states))
        }
        
        for action in matrices:
            for soc in range(nb_state_SOC):
                start = soc * nb_state_price
                end = start + nb_state_price
                if action == 'Inactive':
                    matrices[action][start:end, start:end] = original_matrix
                elif action == 'Charge' and soc > 0:
                    matrices[action][start - nb_state_price:end - nb_state_price, start:end] = original_matrix
                elif action == 'Discharge' and soc < nb_state_SOC - 1:
                    matrices[action][start + nb_state_price:end + nb_state_price, start:end] = original_matrix
        
        # Stack vertically for combined matrix
        combined_matrix = np.vstack(list(matrices.values()))
        return combined_matrix

    
    def get_new_soc(policy_value, soc):
        """Calulcates the new SOC after applying a policy
        """

        # Calculate the change in SOC based on the action
        if policy_value == 1:        
            delta_soc = 1
        elif policy_value == 2:      
            delta_soc = -1
        else:
            delta_soc = 0  # No change in SOC for inaction

        # Get the new state of charge
        new_soc = soc + delta_soc

        return new_soc
    
    
    def get_policy_value(price_range, soc_index, policy_vector, STATES):
        """Gets the policy value for a given state
        """
        
        index = int(np.where((STATES[:, 0] == str(int(soc_index))) & (STATES[:, 1] == price_range))[0])
        policy_value = policy_vector[index]
        
        return policy_value


    def get_revenue(policy_value, price):
        """Calculates the revenue for a SOC and policy
        """
        # Constants for the problem
        changing_rate = 500/(nb_state_SOC - 1)
        
        # Calculate the change in SOC based on the action
        if policy_value == 1:       
            delta_soc = 1
        elif policy_value == 2:       
            delta_soc = -1
        else:
            delta_soc = 0  # No change in SOC for inaction
        
        # Calculate the revenue  
        revenue = price * (-delta_soc) * changing_rate

        return revenue
    
    
    def calculate_revenue(prices, quantiles, policy_vector):
        # Initialize vectors
        revenue = pd.DataFrame(index=prices["HourDK"])
        soc = pd.DataFrame(index=prices["HourDK"])
                    
        revenue["Revenue"] = 0
        soc["SOC"] = 0
    
        # Dictionary to store monthly revenue
        monthly_revenue = {}
    
        # Hourly Revenue calculation
        for i in range(len(prices)):
            price = prices.iloc[i]["PriceEUR"]
            price_range = prices.iloc[i]["price_range"]
            soc_index = soc["SOC"][i]
            
            # Calculate policy value and revenue
            policy_value = get_policy_value(price_range, soc_index, policy_vector, STATES)
            hourly_revenue = get_revenue(policy_value, price)
            revenue["Revenue"][i] = hourly_revenue
    
            # Extract year and month
            year = prices.iloc[i]["HourDK"].year
            month = prices.iloc[i]["HourDK"].month
    
            # Update monthly revenue
            if (year, month) not in monthly_revenue:
                monthly_revenue[(year, month)] = 0
            monthly_revenue[(year, month)] += hourly_revenue
    
            # State of charge for next hour calculation
            if i < (len(prices) - 1):
                soc["SOC"][i + 1] = get_new_soc(policy_value, soc["SOC"][i])
        
        return monthly_revenue, revenue, soc
    
    # Get the rewards matrix
    rewards = reward_matrix(quantiles, STATES, ACTIONS)
    
    # Get a matrix with the valid_actions for each state
    valid_actions = valid_action_matrix(STATES, ACTIONS)
    
    # Compute number of transitions
    transitions = count_transitions(prices)
    
    # Get transition matrix between price states
    pb_matrix = probability_matrix(prices, transitions)
    
    # Compute and combine final transition probalbility matrix
    P_sa = create_combined_matrix(pb_matrix, nb_state_SOC, nb_state_price)
    
    # Create the maintenance problem
    example = BatteryProblem(0.9, P_sa, rewards, valid_actions)

    # Value Iteration
    print("Value Iteration:")
    optimal_values = example.value_iteration()
    optimal_policy = example.policy_from_value(optimal_values)
    for i, state in enumerate(example.STATES):
        state_str = str(state)  # Convert the state (numpy array) to a string
        print(f"{state_str:<15} {optimal_values[i]:<10.2f} {example.ACTIONS[optimal_policy[i]]}")
    
    monthly_revenue_VI, revenue_VI, soc_VI = calculate_revenue(prices, quantiles, optimal_policy)
    
    # Policy Iteration
    print("\nPolicy Iteration:")
    optimal_policy, optimal_values = example.policy_iteration()
    for i, state in enumerate(example.STATES):
        state_str = str(state)  # Convert the state (numpy array) to a string
        print(f"{state_str:<15} {optimal_values[i]:<10.2f} {example.ACTIONS[optimal_policy[i]]}")
        
    monthly_revenue_PI, revenue_PI, soc_PI = calculate_revenue(prices, quantiles, optimal_policy)
    
    
    ###### Plot revenue ######
    plt.figure(figsize=(12, 8))
    
    # Revenue plot
    plt.subplot(2, 1, 1)
    plt.plot(revenue_VI, label="Revenue VI", color='blue')
    plt.plot(revenue_PI, label="Revenue PI", color='red')
    plt.title("Revenue Over Time")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid()
    
    # Hide x-axis labels for the top plot, but keep the ticks
    plt.gca().tick_params(labelbottom=False)

    
    # SOC plot
    plt.subplot(2, 1, 2)
    plt.plot(soc_VI, label="State of Charge (SOC) VI", color='green')
    plt.plot(soc_PI, label="State of Charge (SOC) PI", color='orange')
    plt.title("State of Charge Over Time")
    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.legend()
    plt.grid()


    ######## Plot SOC and prices versus time ############
    fig, ax1 = plt.subplots(figsize=(10, 6))
    interval = 100
    time_range = prices['HourDK'][:interval]
    
    # Plot SOC for VI and PI on the primary y-axis
    ax1.step(time_range, soc_VI['SOC'][:interval], label="State of Charge (SOC) VI", color='green', linestyle='-')
    ax1.step(time_range, soc_PI['SOC'][:interval], label="State of Charge (SOC) PI", color='orange', linestyle='-')
    ax1.set_title("State of Charge and Prices Over Time", fontsize=14)
    ax1.set_xlabel("Time", fontsize=14)
    ax1.set_ylabel("SOC", fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.legend(loc="upper left", fontsize = 12)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Create a secondary y-axis for prices
    ax2 = ax1.twinx()
    ax2.plot(time_range,prices["PriceEUR"][:interval], label="Prices (EUR)", color='blue', linestyle='--', alpha=0.7)
    ax2.set_ylabel("Prices (EUR)", fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.legend(loc="upper right", fontsize=12)

    # Adjust layout and show the plot
    fig.tight_layout()
    plt.show()
    
    
    ######### Plot monthly revenue ############
    # Generate x-axis labels from the dictionary keys
    x_labels_VI = [f"{year}-{month:02d}" for year, month in monthly_revenue_VI.keys()]
    x_labels_PI = [f"{year}-{month:02d}" for year, month in monthly_revenue_PI.keys()]
    
    # Ensure both VI and PI have the same x-axis labels for comparison
    x_labels = sorted(set(x_labels_VI) | set(x_labels_PI))  # Union of both sets, sorted
    
    # Align monthly revenues with x_labels
    monthly_rev_VI = [monthly_revenue_VI.get(tuple(map(int, label.split('-'))), 0) for label in x_labels]
    monthly_rev_PI = [monthly_revenue_PI.get(tuple(map(int, label.split('-'))), 0) for label in x_labels]

    # Calculate limits and ticks for y-axis
    max_revenue = max(max(monthly_rev_VI), max(monthly_rev_PI))
    min_revenue = min(min(monthly_rev_VI), min(monthly_rev_PI))
    y_limit = (min_revenue, max_revenue)
    tick_gap = 50000
    y_ticks = np.arange(-2 * tick_gap, max_revenue + 1.05 * tick_gap, tick_gap)

    plt.figure(figsize=(12, 8))

    # Revenue plot for Value Iteration (VI)
    plt.subplot(2, 1, 1)
    plt.bar(x_labels, monthly_rev_VI, color='blue', alpha=0.7)
    plt.title("Revenue Over Time (Value Iteration)", fontsize=14)
    plt.ylabel("Revenue", fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    plt.ylim(y_limit)
    plt.yticks(y_ticks)
    
    # Hide x-axis labels for the top plot, but keep the ticks
    plt.gca().tick_params(labelbottom=False)

    # Revenue plot for Policy Iteration (PI)
    plt.subplot(2, 1, 2)
    plt.bar(x_labels, monthly_rev_PI, color='blue', alpha=0.7)
    plt.title("Revenue Over Time (Policy Iteration)", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Revenue", fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels to avoid overlap
    plt.grid(axis='y', alpha=0.7)
    plt.ylim(y_limit)
    plt.yticks(y_ticks)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    main()


print('End')
    