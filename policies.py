import numpy as np
def fixedPolicy(state):
    playerSum = state[0]
    if(playerSum<25):
        return 1
    else:
        return 0

def epsilon_greedy(current_state, Q_dic, n_actions,epsilon):

    if np.random.rand() >= epsilon:
        return np.argmax([Q_dic[(current_state,1)],Q_dic[(current_state,0)]])
    else:
        a = np.random.randint(0,n_actions)
        return a