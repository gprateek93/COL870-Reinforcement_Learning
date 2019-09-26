import numpy as np
from player import Player

class environment:
    def __init__(self):
        '''defines the environment of the game.'''
        self.minCardValue,self.maxCardValue = 1,10
        self.gameLowerBound,self.gameUpperBound = 0,31
        self.dealerThreshold = 25 #above threshold the dealer always sticks otherwise hits always.
        self.actionSpace = (0,1) # 0 = stick and 1 = hit.
        self.dealerIndicators = [0,0,0]
        self.reset()

    def reset(self):
        '''This function is used to reset the environment to the initial state where one card is drawn'''
        playerCard = self.drawNewCard()
        dealerCard = self.drawNewCard()
        if dealerCard == (1,0):
            self.dealerIndicators[0] = 1
        if dealerCard == (2,0):
            self.dealerIndicators[1] = 1
        if dealerCard == (3,0):
            self.dealerIndicators[2] = 1    
        self.dealerCardSum = dealerCard[0]
        self.firstCardDrawn  = True
        self.player = Player(playerCard,dealerCard)
        return self.player.getState()
    
    def drawNewCard(self):
        '''returns the value of the new card drawn from the deck. If the card is red then a negative value is returned otherwise a positive value is returned'''
        newCardValue = np.random.randint(self.minCardValue,self.maxCardValue+1)
        if np.random.random() <= 1/3: #red card 
            return (newCardValue,1)
        else: # black card
            return (newCardValue,0)

    def isBust(self,sum):
        return 1 if (sum <self.gameLowerBound or sum >self.gameUpperBound) else 0
    
    def evaluateDealerPolicy(self):
        while(self.dealerCardSum < self.dealerThreshold and self.dealerCardSum>=0):
            newCard = self.drawNewCard()
            self.dealerCardSum  = self.dealerCardSum + newCard[0] if newCard[1] == 0 else self.dealerCardSum - newCard[0]
            if(newCard == (1,0) and self.dealerIndicators[0]!=1):
                if self.dealerCardSum <= 21:
                    self.dealerCardSum+=10
                    self.dealerIndicators[0] = 2
                else:
                    self.dealerIndicators[0] = 1
            if(newCard == (2,0) and self.dealerIndicators[1]!=2):
                if self.dealerCardSum <= 21:
                    self.dealerCardSum+=10
                    self.dealerIndicators[1] = 2
                else:
                    self.dealerIndicators[1] = 1
            if(newCard == (3,0) and self.dealerIndicators[0]!=2):
                if self.dealerCardSum <= 21:
                    self.dealerCardSum+=10
                    self.dealerIndicators[2] = 2
                else:
                    self.dealerIndicators[2] = 1
            if self.dealerCardSum > self.gameUpperBound:
                self.dealerCardSum,self.dealerIndicators = self.modifySum(self.dealerCardSum,self.dealerIndicators,1)
            if self.dealerCardSum < self.gameLowerBound:
                self.dealerCardSum,self.dealerIndicators = self.modifySum(self.dealerCardSum,self.dealerIndicators,0)

    def modifySum(self,playerSum,indicators,mode):
        if(mode == 0):
            #add 10
            if indicators[0] == 1 or indicators[1] == 1 or indicators[2] == 1:
                playerSum += 10
                if(indicators[0] == 1):
                    indicators[0] = 2
                elif(indicators[1] == 1):
                    indicators[1] = 2
                elif(indicators[2] == 1):
                    indicators[2] = 2
                return playerSum,indicators
            else:
                return playerSum,indicators
        elif mode == 1:
            #subtract 10
            if indicators[0] == 2 or indicators[1] == 2 or indicators[2] == 2:
                playerSum -= 10
                if(indicators[0] == 2):
                    indicators[0] = 1
                elif(indicators[1] == 2):
                    indicators[1] = 1
                elif(indicators[2] == 2):
                    indicators[2] = 1
                return playerSum,indicators
            else:
                return playerSum,indicators

    def step(self,action):
        '''Defines what will be the next state, reward given a current state and action'''

        assert action in [0,1], "Expection action in [0, 1] but got %i"%action

        if(self.firstCardDrawn):
            #initial check
            self.firstCardDrawn = False
            if self.isBust(self.dealerCardSum) and self.isBust(self.player.playerSum):
                done = True
                return (self.player.getState(),0,done)
            if self.isBust(self.dealerCardSum):
                done = True
                return (self.player.getState(),1,done)
            if self.isBust(self.player.playerSum):
                done = True
                return (self.player.getState(),-1,done)
        
        if action == 0:
            #agent sticks
            self.evaluateDealerPolicy()
            #print(self.dealerCardSum)
            if (self.isBust(self.dealerCardSum) or self.player.playerSum > self.dealerCardSum):
                done = True
                return (self.player.getState(),1,done)
            elif (self.player.playerSum == self.dealerCardSum):
                done = True
                return (self.player.getState(),0,done)
            else:
                done = True
                return (self.player.getState(),-1,done)
        
        else:
            #agent hits
            newCard = self.drawNewCard()
            ##print(self.player.state)
            #print("newCard is")
            #print(newCard)
            self.player.evaluate(newCard)
            if self.player.playerSum > self.gameUpperBound:
                playerSum,indicators =self.modifySum(self.player.playerSum,self.player.indicators,1)
                if(playerSum != self.player.playerSum):
                    self.player.playerSum = playerSum
                    self.player.indicators = indicators
                else:
                    return (self.player.getState(),-1,True)
                            
            elif self.player.playerSum < self.gameLowerBound:
                playerSum,indicators =self.modifySum(self.player.playerSum,self.player.indicators,0)
                if(playerSum!=self.player.playerSum):
                    self.player.playerSum = playerSum
                    self.player.indicators = indicators
                else:
                    return (self.player.getState(),-1,True)
            
            else:
                if self.player.indicators != [0,0,0] and self.player.indicators != [2,2,2]:
                    if(newCard == (1,0) and self.player.indicators[0] == 1 and (self.player.playerSum + 10) <=self.gameUpperBound):
                        self.player.indicators[0] = 2
                        self.player.playerSum+=10
                    elif(newCard == (2,0) and self.player.indicators[1] == 1 and (self.player.playerSum + 10) <=self.gameUpperBound):
                        self.player.indicators[1] = 2
                        self.player.playerSum+=10
                    if(newCard == (3,0) and self.player.indicators[2] == 1 and (self.player.playerSum + 10) <=self.gameUpperBound):
                        self.player.indicators[2] = 2
                        self.player.playerSum+=10
            self.player.state = (self.player.playerSum,self.player.dealerFirstCard,self.player.indicators)  
            done = False
            return self.player.getState(),0,done