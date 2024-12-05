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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd.set_option('mode.chained_assignment', None)

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
    
    # Split dataset
    split_index_train = int(0.6 * len(prices_area))
    split_index_val = int(0.8 * len(prices_area))
    
    X_train = prices_area[:split_index_train]
    X_val = prices_area[split_index_train:split_index_val]
    X_test = prices_area[split_index_val:]
    
    X_train['Month'] = X_train['HourDK'].dt.month
    X_train['Year'] = X_train['HourDK'].dt.year
    
    # Generate quantiles based on the number of ranges
    prices_ranges = np.linspace(0, 1, num=nb_price_ranges + 1)

    # Dynamically generate labels
    labels = [f"Range {i+1}" for i in range(nb_price_ranges)]

    # Discretize prices into ranges
    X_train.loc[:, 'price_range'] = pd.qcut(
        X_train['PriceEUR'], 
        q=prices_ranges, 
        labels=labels, 
        duplicates='drop'
    )

    # K-Means Clustering
    features = ['PriceEUR']
    clustering_data = X_train[features].dropna()

    # Standardize the features (important for K-Means)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=nb_price_ranges, random_state=42)
    clustering_data['Cluster'] = kmeans.fit_predict(scaled_data)

    # Add the cluster labels back to the ornb_state_priceiginal dataset
    X_train['Cluster'] = None
    X_train.loc[clustering_data.index, 'Cluster'] = clustering_data['Cluster']

    cluster_means = X_train.groupby('Cluster')['PriceEUR'].mean()
    X_train['ClusterMean'] = X_train['Cluster'].map(cluster_means)

    cluster_centers = kmeans.cluster_centers_

    cluster_means = cluster_means.sort_values(ascending=True).values

    # Calculate the mean price for each range
    cluster_means = (
        X_train.groupby('price_range')['PriceEUR']
        .mean()
        .to_dict()
    )

    # Example centroids (you will use kmeans.cluster_centers_)
    centroids = kmeans.cluster_centers_

    # Sort centroids for proper boundary calculation
    sorted_centroids = np.sort(centroids, axis=0)

    # Compute boundaries (midpoints between centroids)
    boundaries = []
    for i in range(len(sorted_centroids) - 1):
        midpoint = (sorted_centroids[i] + sorted_centroids[i + 1]) / 2
        boundaries.append(midpoint)

    # Add min and max values to define the complete range
    min_value = sorted_centroids[0]
    max_value = sorted_centroids[-1]

    # Combine min, midpoints, and max into a single array
    price_ranges = np.vstack([min_value] + boundaries + [max_value])

    # Get quantile bins from the computed price ranges in X_train
    bins = pd.qcut(X_train['PriceEUR'], q=prices_ranges, retbins=True, duplicates='drop')[1]
    
    # Extend the bins slightly to handle edge cases
    bins[0] = bins[0] - 100  # Extend lower bound slightly
    bins[-1] = bins[-1] + 200  # Extend upper bound slightly

    # Assign the 'price_range' column to X_val and X_test using the same quantile bins
    X_val.loc[:, 'price_range'] = pd.cut(
        X_val['PriceEUR'], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )
    X_test.loc[:, 'price_range'] = pd.cut(
        X_test['PriceEUR'], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )    

    return prices_area, cluster_means, X_train, X_val, X_test, price_ranges
  
    
def get_price(state, quantiles):

    category_price = state[1]

    # Return the quantile range for the given category
    return quantiles[category_price]

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

 
def valid_action_matrix(quantiles, STATES, ACTIONS):
    valid_actions = []
    for s in range(len(STATES)):
        list_action = []
        for a in ACTIONS:
            if s in [i for i in range(len(quantiles))]: # when the battery SOC is 0
                print(s)
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


def calculate_revenue(prices, quantiles, policy_vector, STATES):
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


#nb_state_price = 5
nb_state_SOC = 6 # from 0 to 500 MWh with a 100MWh step

SOC_states = range(nb_state_SOC)  # Iterable range of SOC
#price_states = range(nb_state_price)  # Iterable range of price states

validation_revenue = {}

for nb_state_price in range(2,12):
    price_states = range(nb_state_price)  # Iterable range of price states
    
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
    
    
    #### Run Step 2 and 3 ####
    market_zone = 'DK2' # Choose between DK1, DK2, DE, SO3, SO4, NO2, SYSTEM
    prices, quantiles, X_train, X_val, X_test, price_ranges = load_prices(nb_state_price, market_zone)
    
    ACTIONS = np.array(['Inactive', 'Charge', 'Discharge'])
    STATES = np.array([(s, "Range {0}".format(p+1)) for s in SOC_states for p in price_states])
    action_index_map = {'Inactive': 0, 'Charge': 1, 'Discharge': 2}  # Map actions to correct indexes
    
    # Get the rewards matrix
    rewards = reward_matrix(quantiles, STATES, ACTIONS)
    
    # Get a matrix with the valid_actions for each state
    valid_actions = valid_action_matrix(quantiles, STATES, ACTIONS)
    
    # Compute number of transitions
    transitions = count_transitions(X_train)
    
    # Get transition matrix between price states
    pb_matrix = probability_matrix(X_train, transitions)
    
    # Compute and combine final transition probalbility matrix
    P_sa = create_combined_matrix(pb_matrix, nb_state_SOC, nb_state_price)
    
    # Create the maintenance problem
    example = BatteryProblem(0.9, P_sa, rewards, valid_actions)

    # Value Iteration
    print("Value Iteration:")
    VI_optimal_values = example.value_iteration()
    VI_optimal_policy = example.policy_from_value(VI_optimal_values)
    for i, state in enumerate(example.STATES):
        state_str = str(state)  # Convert the state (numpy array) to a string
        print(f"{state_str:<15} {VI_optimal_values[i]:<10.2f} {example.ACTIONS[VI_optimal_policy[i]]}")
    
    monthly_revenue_VI, revenue_VI, soc_VI = calculate_revenue(X_train, quantiles, VI_optimal_policy, STATES)
    
    # Policy Iteration
    print("\nPolicy Iteration:")
    PI_optimal_policy, PI_optimal_values = example.policy_iteration()
    for i, state in enumerate(example.STATES):
        state_str = str(state)  # Convert the state (numpy array) to a string
        print(f"{state_str:<15} {PI_optimal_values[i]:<10.2f} {example.ACTIONS[PI_optimal_policy[i]]}")
        
    monthly_revenue_PI, revenue_PI, soc_PI = calculate_revenue(X_train, quantiles, PI_optimal_policy, STATES)
    
    ### Validation set ####
    val_monthly_revenue_VI, val_revenue_VI, val_soc_VI = calculate_revenue(X_val, quantiles, VI_optimal_policy, STATES)
    val_monthly_revenue_PI, val_revenue_PI, val_soc_PI = calculate_revenue(X_val, quantiles, PI_optimal_policy, STATES)
    
    validation_revenue[nb_state_price] = val_revenue_VI['Revenue'].sum()
    
    ###### Plot revenue ######
    plt.figure(figsize=(12, 8))
    
    # Revenue plot
    plt.subplot(2, 1, 1)
    plt.plot(revenue_VI[:100], label="Revenue VI", color='blue')
    plt.plot(revenue_PI[:100], label="Revenue PI", color='red')
    plt.title(f"Revenue Over Time - {nb_state_price} state prices")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid()
    
    # Hide x-axis labels for the top plot, but keep the ticks
    plt.gca().tick_params(labelbottom=False)

    
    # SOC plot
    plt.subplot(2, 1, 2)
    plt.plot(soc_VI[:100], label="State of Charge (SOC) VI", color='green')
    plt.plot(soc_PI[:100], label="State of Charge (SOC) PI", color='orange')
    plt.title(f"State of Charge Over Time - {nb_state_price} state prices")
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
    ax1.set_title(f"State of Charge and Prices Over Time - {nb_state_price} state prices", fontsize=14)
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
    
    
    ######### Plot monthly revenue for training ############
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
    plt.title(f"Revenue Over Time (Value Iteration) - {nb_state_price} state prices", fontsize=14)
    plt.ylabel("Revenue", fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    plt.ylim(y_limit)
    plt.yticks(y_ticks)
    
    # Hide x-axis labels for the top plot, but keep the ticks
    plt.gca().tick_params(labelbottom=False)

    # Revenue plot for Policy Iteration (PI)
    plt.subplot(2, 1, 2)
    plt.bar(x_labels, monthly_rev_PI, color='blue', alpha=0.7)
    plt.title("fRevenue Over Time (Policy Iteration) - {nb_state_price} state prices", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Revenue", fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels to avoid overlap
    plt.grid(axis='y', alpha=0.7)
    plt.ylim(y_limit)
    plt.yticks(y_ticks)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    
    ######### Plot monthly revenue for validation ############
    # Generate x-axis labels from the dictionary keys
    x_labels_VI = [f"{year}-{month:02d}" for year, month in val_monthly_revenue_VI.keys()]
    x_labels_PI = [f"{year}-{month:02d}" for year, month in val_monthly_revenue_PI.keys()]
    
    # Ensure both VI and PI have the same x-axis labels for comparison
    x_labels = sorted(set(x_labels_VI) | set(x_labels_PI))  # Union of both sets, sorted
    
    # Align monthly revenues with x_labels
    monthly_rev_VI = [val_monthly_revenue_VI.get(tuple(map(int, label.split('-'))), 0) for label in x_labels]
    monthly_rev_PI = [val_monthly_revenue_PI.get(tuple(map(int, label.split('-'))), 0) for label in x_labels]

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
    plt.title(f"Revenue Over Time (Value Iteration - Validation) - {nb_state_price} state prices", fontsize=14)
    plt.ylabel("Revenue", fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    plt.ylim(y_limit)
    plt.yticks(y_ticks)
    
    # Hide x-axis labels for the top plot, but keep the ticks
    plt.gca().tick_params(labelbottom=False)

    # Revenue plot for Policy Iteration (PI)
    plt.subplot(2, 1, 2)
    plt.bar(x_labels, monthly_rev_PI, color='blue', alpha=0.7)
    plt.title(f"Revenue Over Time (Policy Iteration - Validation) - {nb_state_price} state prices", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Revenue", fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels to avoid overlap
    plt.grid(axis='y', alpha=0.7)
    plt.ylim(y_limit)
    plt.yticks(y_ticks)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()



print(validation_revenue)

# Extract keys and values
keys = list(validation_revenue.keys())
values = list(validation_revenue.values())

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(keys, values, marker='o', linestyle='-', color='b')

# Add labels, title, and legend
plt.xlabel('Number of discretized price states')
plt.ylabel('Cumulative revenue')
plt.title('Cumulative revenue against the number of price state (Valdiation)')
plt.legend()
plt.grid()

# Show the graph
plt.show() 
    


######## Step 4 : Code ##############

def get_prices(market_zone):
    """
    Load the prices from the chosen market zone.

    Parameters:
        market_zone (string): The area of interest.

    Returns:
        prices_area (pd.DataFrame): The DataFrame with a 'price_range' column.
    """

    # Load the Excel file
    prices = pd.read_excel("Price.xlsx")

    # Filter by area
    prices = prices[prices['PriceArea'] == market_zone]
    prices.reset_index(drop=True, inplace=True)

    return prices


def run_deterministic_model(prices, power_capacity, energy_capacity):
    # Calculate the number of intervals per day based on the interval_hours
    
    intervals_per_day = 24 # hourly actions 
    # The optimisation model is ran one day at a time. 
    # Indeed, it is not realistic to look at a bigger time range as the prices will not be forecasted well enough too long in advance
    daily_index = [_ for _ in range(0, len(prices), intervals_per_day)]
    daily_index.pop()
    
    # Create dictionnaries to save data
    hourly_revenue = {}
    hourly_operations = {}
    hourly_operations['charge'] = {}
    hourly_operations['discharge'] = {}
    hourly_operations['SOC'] = {-1: 0} # We assume that the battery is empty when starting in the first hour of the time period studied (ie 01/01/2021)
    
    count_unfeasible = 0
    
    #initial_SOC = 0.5 * usable_capacity

    # Run the optimization model for each day of the time period studied
    for day_index in daily_index:
        TIME = [_ for _ in range(day_index, day_index + intervals_per_day, 1)]
        
        #if day_index == 23760: # to deal with the fact that one day in 2023 has only 23 hours 
        #    TIME.pop()

        # Create a Gurobi model for the energy arbitrage problem
        EA_model = gb.Model("Energy Arbitrage for day d")

        # Set time limit
        EA_model.Params.TimeLimit = 100

        # Add variables to the Gurobi model
        # Charging rate at hour h in MW
        charged_power = {
            h: EA_model.addVar(
                lb=0,
                ub=1, # the battery cannot charge more than its power capacity in an hour
                vtype = GRB.INTEGER,
                name="Charging rate at hour {0}".format(h),
            ) for h in TIME}

        # Discharging rate at hour h in MW
        discharged_power = {
            h: EA_model.addVar(
                lb=0,
                ub=1, # the battery cannot charge more than its power capacity in an hour
                vtype = GRB.INTEGER,
                name="Discharging rate at hour {0}".format(h),
            ) for h in TIME}

        # State of charge at hour h in MWh
        SOC = {
            h: EA_model.addVar(
                lb=0,
                ub=energy_capacity/100, # the battery cannot have a SOC exceeding its energy capacity
                name="State of charge at hour {0}".format(h),
            ) for h in TIME}

        # Initial state of charge of the battery (MWh) at TIME[0]
        SOC_init = EA_model.addVar(
            lb=0,
            ub=gb.GRB.INFINITY,
            name="Initial battery SOC"
        )
        
        # Binary variable to ensure charging and discharging don`t happen at the same time
        Y_bess = {
            h: EA_model.addVar(
                vtype = GRB.BINARY,
                name="Binary variable for hour {0}".format(h),
            ) for h in TIME}

        # Objective function
        obj = gb.quicksum(prices['PriceEUR'][h] * discharged_power[h] - prices['PriceEUR'][h] * charged_power[h] for h in TIME)
        EA_model.setObjective(obj, GRB.MAXIMIZE)

        # Add constraints to Guroby model
        # SOC constraint
        SOC_constraint = ({})
        for t in TIME:
            if t == TIME[0]:
                SOC_constraint[t] = EA_model.addConstr(
                    SOC[t],
                    gb.GRB.EQUAL,
                    SOC_init + charged_power[t] - discharged_power[t],
                    name="Constraint on SOC at time 1",
                )
            else:
                SOC_constraint[t] = EA_model.addConstr(
                    SOC[t],
                    gb.GRB.EQUAL,
                    SOC[t-1] + charged_power[t] - discharged_power[t],
                    name="Constraint on SOC at time {0}".format(t),
                )

        # Initial SOC constraint
        Init_SOC_constraint = {
            EA_model.addConstr(
                SOC_init,
                gb.GRB.EQUAL,
                hourly_operations['SOC'][TIME[0]-1],
                name="Initial SOC equation",
            )}
        
        # Charging rate
        max_charge_constraint = {
            t: EA_model.addConstr(
                charged_power[t],
                gb.GRB.LESS_EQUAL,
                1 * Y_bess[t],
                name="Maximum power charged at time {0}".format(t)
                )
            for t in TIME
        }
        
        # Discharging rate
        max_discharge_constraint = {
            t: EA_model.addConstr(
                discharged_power[t],
                gb.GRB.LESS_EQUAL,
                1 * (1- Y_bess[t]),
                name="Maximum power discharged at time {0}".format(t)
                )
            for t in TIME
        }
        
        """
        # end of the day SOC constraint
        # Since we don't want the battery to discharge completely at the end of the day
        end_SOC_constraint = EA_model.addConstr(
            SOC[TIME[intervals_per_day-1]],
            gb.GRB.GREATER_EQUAL,
            0.4*energy_capacity/100,
            name="End of the day SOC constraint")
        """
        EA_model.optimize()
        
        
        if EA_model.status == GRB.OPTIMAL: 
            for t in TIME:
                hourly_operations['charge'][t] = charged_power[t].x * power_capacity
                hourly_operations['discharge'][t] = discharged_power[t].x * power_capacity
                hourly_operations['SOC'][t] = SOC[t].x * power_capacity
                hourly_revenue[t] = power_capacity * (prices['PriceEUR'][t] * discharged_power[t].x - prices['PriceEUR'][t] * charged_power[t].x)
        else: # unfeasible for August 8th - only negative prices
            count_unfeasible+=1
            for t in TIME:
                hourly_operations['charge'][t] = 0
                hourly_operations['discharge'][t] = 0
                hourly_operations['SOC'][t] = 0
                hourly_revenue[t] = 0
                
    return hourly_operations, hourly_revenue, count_unfeasible


def compute_monthly_revenues(hourly_revenue, prices):
    # Create DataFrame from hourly_revenue
    revenue_df = pd.DataFrame(list(hourly_revenue.items()), columns=['Hour', 'Revenue'])
    revenue_df['HourDK'] = prices['HourDK'][:len(revenue_df)]  # Match the datetime column from prices
    
    # Add columns for day and month
    revenue_df['Day'] = revenue_df['HourDK'].dt.date  # Extract the day
    revenue_df['Month'] = revenue_df['HourDK'].dt.to_period('M')  # Extract the month (as Period)

    # Compute daily revenue (sum per day)
    daily_revenue = revenue_df.groupby('Day')['Revenue'].sum().to_dict()

    # Compute monthly revenue (sum per month), with keys as (year, month)
    monthly_revenue = revenue_df.groupby(revenue_df['HourDK'].dt.to_period('M'))['Revenue'].sum()
    monthly_revenue = monthly_revenue.rename(lambda x: (x.year, x.month), axis='index').to_dict()

    return monthly_revenue


def compare_revenues(monthly_revenue_VI, monthly_revenue_PI, opti_monthly_revenue, prices):
    # Extract keys (which are (year, month) tuples) and values
    months = list(monthly_revenue_VI.keys())
    revenue_VI = list(monthly_revenue_VI.values())
    revenue_PI = list(monthly_revenue_PI.values())
    revenue_opti = list(opti_monthly_revenue.values())
    
    # Monthly prices
    monthly_prices = prices.groupby(prices['HourDK'].dt.to_period('M'))['PriceEUR'].mean()
    
    # Prepare the labels for the x-axis (Year-Month)
    month_labels = [f"{year}-{month:02d}" for year, month in months]
    # Create the figure and the first y-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot revenues on the first y-axis
    ax1.plot(month_labels, revenue_VI, label="Revenue VI", marker='o', color='green', linestyle='-', alpha=0.7)
    ax1.plot(month_labels, revenue_PI, label="Revenue PI", marker='o', color='orange', linestyle='-', alpha=0.7)
    ax1.plot(month_labels, revenue_opti, label="Optimized Revenue", marker='o', color='blue', linestyle='-', alpha=0.7)

    # Labeling the first y-axis
    ax1.set_ylabel("Revenue", fontsize=12)
    ax1.set_xlabel("Month", fontsize=12)
    ax1.tick_params(axis='y')
    ax1.set_xticks(range(len(month_labels)))
    ax1.set_xticklabels(month_labels, rotation=45, ha='right')
    
    # Adding a legend for the first y-axis
    ax1.legend(loc="upper left")

    # Create the second y-axis
    ax2 = ax1.twinx()
    
    # Plot monthly prices on the second y-axis
    ax2.plot(month_labels, monthly_prices, label="Average Monthly Price (EUR)", color='red', linestyle='--', alpha=0.7)
    
    # Labeling the second y-axis
    ax2.set_ylabel("Average Monthly Price (EUR)", fontsize=12)
    ax2.tick_params(axis='y')

    # Adding a legend for the second y-axis
    ax2.legend(loc="upper right")

    # Adding the title and rotating x-axis labels
    plt.title("Monthly Revenue Comparison", fontsize=14)
    
    # Adding grid lines for the primary y-axis
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    

############## Run step 4 #############
# Battery specifications
power_capacity = 100 # MW
energy_capacity = 500 # MWh

#opti_hourly_operations, opti_hourly_revenue, count_unfeasible = run_deterministic_model(X_train, power_capacity, energy_capacity)

#opti_monthly_revenue = compute_monthly_revenues(opti_hourly_revenue, X_train)

#compare_revenues(monthly_revenue_VI, monthly_revenue_PI, opti_monthly_revenue, X_train)


    
#if __name__ == "__main__":
#    main()

print('End')