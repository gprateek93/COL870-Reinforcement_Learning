import numpy as np 
from player import Player
from simulator import environment

def fixedPolicy(state):
    playerSum = state[0]
    if(playerSum<25):
        return 1
    else:
        return 0

def getAllStates():
    states = []
    for i in range (0,32):
        playerSum = i
        for j in  range (1,11):
            dealerCard = (j,0)
            for k in range (0,3):
                ind_1 = k
                for l in range (0,3):
                    ind_2 = l
                    for m in range(0,3):
                        ind_3 = m
                        for a in range (0,2):
                            action = a
                            states.append(((playerSum,dealerCard,(ind_1,ind_2,ind_3)),action))
    return states


def monte_carlo_FV(episodes,discount = 1):
    Q = {}
    total_states = getAllStates()
    for s in total_states:
        Q[s] = 0
    total_return = {}
    env = environment()
    for i in range(episodes):
        print("episode number is")
        print(i)
        history = []
        current_state = env.reset()
        while True:
            action = fixedPolicy(current_state)
            next_state, reward, terminate = env.step(action)
            if current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0:
                hashstate = (current_state[0],current_state[1],tuple(current_state[2]))
                history.append((hashstate,action,reward))
            current_state = next_state
            if terminate:
                break
        expected_return = 0

        for i in range(len(history)-2,-1,-1):
            expected_return = discount * expected_return + history[i+1][2]
            flag = 0
            for j in range(0,i):
                if history[j][0] == history[i][0] and history[j][1] == history[j][0]:
                    flag = 1
                    break
            s = history[i][0]
            a = history[i][1]
            if flag == 0:
                if total_return.get((s,a)) is None:
                    total_return[(s,a)] = []
                total_return[(s,a)].append(expected_return)
                if Q.get((s,a)) is None:
                    Q[(s,a)] = 0
                Q[(s,a)] = sum(total_return[(s,a)])/len(total_return[(s,a)])
    m = 0
    for k,v in Q.items():
        m = max(k[0][0],m)
    print(m)
    print(len(Q.keys()))
    return Q

monte_carlo_FV(100000)             