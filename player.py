import numpy as np

class Player:
    def __init__(self,playerCard,dealerCard):
        self.playerSum = playerCard[0] if playerCard[1] == 0 else -playerCard[0]
        self.dealerFirstCard = dealerCard
        self.indicators = [0,0,0]
        if playerCard == (1,0):
            self.indicators[0] = 2
            self.playerSum+=10
        if playerCard == (2,0):
            self.indicators[1] = 2
            self.playerSum+=10
        if playerCard == (3,0):
            self.indicators[2] = 2
            self.playerSum+=10
        self.state = (self.playerSum,self.dealerFirstCard,self.indicators)
    
    def getState(self):
        return self.state

    def evaluate(self,card):
        self.playerSum= self.playerSum + card[0] if card[1] == 0 else self.playerSum - card[0]
        if card == (1,0):
            if self.indicators[0] == 0:
                self.indicators[0] = 1
        if card == (2,0):
            if self.indicators[1] == 0:
                self.indicators[1] = 1
        if card == (3,0):
            if self.indicators[2] == 0:
                self.indicators[2] = 1  
        self.state = (self.playerSum,self.dealerFirstCard,self.indicators)         
