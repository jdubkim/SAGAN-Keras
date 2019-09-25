import numpy as np

n_iter = 2000
k = 10
timesteps = 1000

counts = np.zeros(k) 
estimate_qs = np.zeros(k) 
reward_log = []

real_qs = np.random.normal(0, 1, size=k)

eps = [0, 0.01, 0.1]

def main():
    result = [] # 2000 X 3 X 1000
    for n in range(n_iter):
        result.append([])
        for e in eps:
            seed = np.random.randint(1000, size=1)
            reward_log = experiment(seed, e) # dim: 1 X 1000
            result[-1].append(reward_log) 
    print(result)
    return result

def experiment(seed, e):
    np.random.seed(1000)
    for t in range(timesteps):
   
        action = choose_action()
        reward = get_reward(action)
        update_q_function(action, reward)
        reward_log.append(reward)

    return reward_log

def choose_action():
    max_value = np.amax(estimate_qs) 
    index = np.where(estimate_qs == max_value)

    return action

def get_reward(action):
    return np.random.normal(estimate_qs[action], 1, size=1)

def update_q_function(action, reward):
    assert action >=0 and action < k
    
    estimate_qs[action] = (estimate_qs[action] * counts[action] + reward) / (counts[action] + 1)
    counts[action] = counts[action] + 1 
    

