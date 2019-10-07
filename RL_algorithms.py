from player import Player
from simulator import environment
from policies import *

def getAllStates():
    states = []
    for i in range (-11,42):
        playerSum = i
        for j in  range (0,11):
            for card in range(0,2):
                dealerCard = (j,card)
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


def monte_carlo(episodes,mode = "fv", discount = 1):
    Q = {}
    total_states = getAllStates()
    for s in total_states:
        Q[s] = 0
    total_return = {}
    env = environment()
    for i in range(episodes):
        # print("episode number is")
        # print(i)
        history = []
        current_state = env.reset()
        while True:
            action = fixedPolicy(current_state)
            next_state, reward, terminate = env.step(action)
            # if current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0:
            hashstate = (current_state[0],current_state[1],tuple(current_state[2]))
            history.append((hashstate,action,reward))
            current_state = next_state
            if terminate:
                break
        expected_return = 0

        for i in range(len(history)-2,-1,-1):
            expected_return = discount * expected_return + history[i+1][2]
            flag = 0
            if mode == "fv":
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
    # print(m)
    # print(len(Q.keys()))
    return Q


def td_learning(episodes,alpha = 0.1,discount_factor = 1, k = 1):
    Q = {}
    total_states = getAllStates()
    for s in total_states:
        Q[s] = 0
    env = environment()
    for i in range(episodes):
        history = []
        # print("Episode is ",i)
        current_state = env.reset()
        while not (current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0) : 
            current_state = env.reset()
        T = 1e10
        t = 0
        while True:
            if t<T:
                action =  fixedPolicy(current_state)
                s_t, r_t, terminate = env.step(action)
                # if current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0:
                hashstate = (current_state[0],current_state[1],tuple(current_state[2]))
                history.append((hashstate,action,r_t))
                current_state = s_t
                if terminate:
                    T  = t+1
            tau = t-k+1
            if tau >= 0:
                G = 0
                for i in range(tau+1,min(tau+k,T)+1):
                    G += pow(discount_factor, i-(tau+1)) * history[i-1][2]
                if tau+k < T:
                    G+=pow(discount_factor,k)*Q[(history[tau+k-1][0],history[tau+k-1][1])]
                Q[(history[tau][0],history[tau][1])] += alpha * (G - Q[(history[tau][0],history[tau][1])])
            t+=1
            if tau == T-1:
                break

    m = 0
    for n,v in Q.items():
        m = max(v,m)
    # print(m)
    return Q

def n_step_sarsa(episodes,alpha = 0.1, discount_factor = 1, k = 1, epsilon = 0.1, decay = False):
    Q = {}
    pi = {}
    total_states = getAllStates()
    for s in total_states:
        Q[s] = 0
        pi[s[0]] = 0
    env = environment()
    episodic_rewards = []
    for i in range(episodes):
        history = []
        total_episodic_reward = 0
        # print("Episode is ",i)
        current_state = env.reset()
        # while not (current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0) : 
        #     current_state = env.reset()
        if decay:
            epsilon = epsilon / (i+1)
        hashstate = (current_state[0],current_state[1],tuple(current_state[2]))
        # current_action = fixedPolicy(current_state)
        current_action = epsilon_greedy(hashstate,Q,2,epsilon)
        history.append((hashstate,0))
        T = 1e10
        t = 0
        while True:
            print(t)
            if t<T:
                s_t, r_t, terminate = env.step(current_action)
                print(terminate)
                print(r_t)
                total_episodic_reward+=r_t
                current_state = s_t
                # if current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0:
                hashstate = (current_state[0],current_state[1],tuple(current_state[2]))
                if terminate:
                    T  = t+1
                else:
                    # current_action = fixedPolicy(current_state)
                    current_action = epsilon_greedy(hashstate,Q,2,epsilon)
                history.append((hashstate,current_action,r_t))
            tau = t-k+1
            if tau >= 0:
                G = 0
                for i in range(tau+1,min(tau+k,T)+1):
                    G += pow(discount_factor, i-(tau+1)) * history[i][2]
                if tau+k < T:
                    G+=pow(discount_factor,k)*Q[(history[tau+k-1][0],history[tau+k-1][1])]
                Q[(history[tau][0],history[tau][1])] += alpha * (G - Q[(history[tau][0],history[tau][1])])
            t+=1
            if tau == T-1:
                episodic_rewards.append(total_episodic_reward)
                break

    m = 0
    for k,v in Q.items():
        m = max(v,m)
    # print(m)
    # # print(Q)
    return Q,episodic_rewards

 
def q_learning(episodes, alpha = 0.1, discount_factor = 1, epsilon = 0.1,k=1):
    Q = {}
    pi = {}
    total_states = getAllStates()
    for s in total_states:
        Q[s] = 0
        pi[s[0]] = 0
    env = environment()
    episodic_rewards = []
    for i in range(episodes):
        history = []
        total_episodic_reward = 0
        # print("Episode is ",i)
        current_state = env.reset()
        while not (current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0) : 
            current_state = env.reset()
        T = 1e10
        t = 0
        while True:
            if t<T:
                hashcurrentstate = (current_state[0],current_state[1],tuple(current_state[2]))
                action =  epsilon_greedy(hashcurrentstate,Q,2,epsilon)
                pi[hashcurrentstate] = action
                s_t, r_t, terminate = env.step(action)
                total_episodic_reward+=r_t
                # if current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0:
                hashstate = (current_state[0],current_state[1],tuple(current_state[2]))
                history.append((hashstate,action,r_t))
                current_state = s_t
                if terminate:
                    T  = t+1
            tau = t-k+1
            if tau >= 0:
                G = 0
                for i in range(tau+1,min(tau+k,T)+1):
                    G += pow(discount_factor, i-(tau+1)) * history[i-1][2]
                if tau+k < T:
                    G+=pow(discount_factor,k)*max(Q[(history[tau+k-1][0],0)],Q[(history[tau+k-1][0],1)]) #update by 1 step ahead greedy policy
                Q[(history[tau][0],history[tau][1])] += alpha * (G - Q[(history[tau][0],history[tau][1])])
            t+=1
            if tau == T-1:
                episodic_rewards.append(total_episodic_reward)
                break

    m = 0
    for k,v in Q.items():
        m = max(v,m)
    # print(m)
    # print(Q)
    return Q,episodic_rewards

def sarsa_lambda(episodes,alpha=0.1,discount_factor=1,epsilon=0.1,lmbda=0.5,decay=False):
    Q = {}
    total_states = getAllStates()
    for s in total_states:
        Q[s] = 0
    env = environment()
    episodic_rewards = []
    for i in range(episodes):
        total_episodic_reward = 0
        E = {}
        for s in total_states:
            E[s] = 0
        # print("Episode ",i)
        current_state = env.reset()
        while not (current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0) : 
            current_state = env.reset()
        current_action = 1
        while True:
            # if current_state[0] >= 0 and current_state[0] <=31 and current_state[1][1] == 0:
            hashcurrentstate = (current_state[0],current_state[1],tuple(current_state[2]))
            next_state, reward, terminate=  env.step(current_action)
            total_episodic_reward+=reward
            hashnextstate = (next_state[0],next_state[1],tuple(next_state[2]))
            if decay:
                epsilon = epsilon / (i+1)
            next_action = epsilon_greedy(hashnextstate,Q,2,epsilon)
            delta = reward + discount_factor * (Q[(hashnextstate,next_action)]) - Q[(hashcurrentstate,current_action)]
            E[(hashcurrentstate,current_action)] += 1
            for s in Q.keys():
                Q[s]+=alpha*delta*E[s]
                # if s == (hashcurrentstate,current_action) and delta:
                #     print("True")
                #     print(Q[s])
                E[s] = discount_factor*lmbda*E[s]
            current_state = next_state
            current_action = next_action
            if terminate:
                episodic_rewards.append(total_episodic_reward)
                break

    m = 0
    for k,v in Q.items():
        m = max(v,m)
    # print(m)
    # print(Q)
    return Q,episodic_rewards            
#sarsa_lambda(1000)
_,t = n_step_sarsa(100000,k=100)
print(t)