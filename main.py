from player import Player
from simulator import environment

a = environment()
terminate = False
counter = 0
flag  = 0
while(not terminate):
    if flag == 0:
        state,reward,terminate = a.step(1)
        print(state[0])
    else:
        state,reward,terminate = a.step(0)
    if(state[0] >= 25):
        flag = 1
print(state)
print(reward)