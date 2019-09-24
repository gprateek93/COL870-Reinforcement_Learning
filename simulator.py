import numpy as np
class environment:
    def __init__(self):
        '''defines the environment of the game.'''
        self.minCardValue,self.maxCardValue = 1,10
        self.gameLowerBound,self.gameUpperBound = 0,31
        self.dealerThreshold = 25 #above threshold the dealer always sticks otherwise hits always.
        self.actionSpace = (0,1) # 0 = stick and 1 = hit.
        self.dealerCardSum = self.drawNewCard()
        self.agentCardSum = self.drawNewCard()
        self.firstCardDrawn = True

    def drawNewCard(self):
        '''returns the value of the new card drawn from the deck. If the card is red then a negative value is returned otherwise a positive value is returned'''
        newCardValue = np.random.randint(self.minCardValue,self.maxCardValue+1)
        if np.random.random() <= 1/3: #red card 
            return -newCardValue
        else: # black card
            return newCardValue

    def step(self,action):
        '''Defines what will be the next state, reward given a current state and action'''

        assert action in [0,1], "Expection action in [0, 1] but got %i"%action

        if(self.firstCardDrawn):
            self.firstCardDrawn = False
            if(self.agentCardSum < 0 and self.dealerCardSum<0):
                reward = 0
                done = True
            else if(self.agentCardSum<0):
                reward = -1
                done = True
            else if(self.dealerCardSum<0):
                reward = 1
                done = True
        
        if action == 0:
            #agent sticks
            if self.agentCardSum>=self.minCardValue and self.agentCardSum<=self.maxCardValue:
                reward = self.evaluateDealer()
            else:
                reward = -1
            done = True

        
         

