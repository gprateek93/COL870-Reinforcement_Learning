from player import Player
from simulator import environment

a = environment()
terminate = False
counter = 0
while(not terminate):
    state,reward,terminate = a.step(1)
print(state)
print(reward)