#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:04:18 2019

@author: juho

Create lookup table for all hole card combinations.
"""

import itertools
import numpy as np
import scipy.special
from numba import jit, prange
import time

from equity_calculator.params import LUT_nChooseK_2cards
from equity_calculator.equity_calculator import computeEquity
from hand_eval.evaluator_numba import evaluate_numba2 as evaluator
from hand_eval.params import ranks_7cards, LUT_nChooseK_7cards, LUT_nChooseK_5cards, \
    cardToInt, intToCard


@jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def computeFlopEquities(cardCombs, nIters):
    LUT_equities = np.zeros(len(cardCombs), dtype=np.float32)
    
    for i in prange(len(cardCombs)):
        if(i%10000==0): print(i/1000000,len(cardCombs)/1000000)
        
        curCardComb = cardCombs[i]

        eq = computeEquity(curCardComb[:2], curCardComb[2:], nIters)

#        print('hole ' + str([intToCard[card] for card in curCardComb[:2] ]))
#        print('board ' + str([intToCard[card] for card in curCardComb[2:] ]))
        
        # Remove current cards from the deck
#        mask = np.ones(52, dtype=np.bool_)
#        mask[curCardComb] = 0
#        deck = np.arange(52)[mask]
#        nCardsDeck = len(deck)
#        
#        # Run simulations
#        wins = 0
#        curCards, opponentCards = np.zeros(7, dtype=np.uint8), np.zeros(7, dtype=np.uint8)
#        for k in range(nIters):
#            
#            # Draw random cards from the deck
#            for t in range(4):
#                rndIdx = np.random.randint(t,nCardsDeck)
#                opponentCards[t] = deck[rndIdx]
#                deck[t],deck[rndIdx] = deck[rndIdx],deck[t]
#            opponentCards[4:] = curCardComb[2:]
#            
#            curCards[:2] = curCardComb[:2]      # Hole cards
#            curCards[2:] = opponentCards[2:]    # Board cards
#
#            rankCur = evaluator(curCards, ranks_7cards, LUT_nChooseK_7cards)
#            rankOpponent = evaluator(opponentCards, ranks_7cards, 
#                                     LUT_nChooseK_7cards)
#    
#            isWin = int(rankCur < rankOpponent)
#            wins += isWin
            
        # Get index to equity LUT
        tmp0 = LUT_nChooseK_5cards[curCardComb[0],0]
        tmp1 = LUT_nChooseK_5cards[curCardComb[1],1]
        tmp2 = LUT_nChooseK_5cards[curCardComb[2],2]
        tmp3 = LUT_nChooseK_5cards[curCardComb[3],3]
        tmp4 = LUT_nChooseK_5cards[curCardComb[4],4]
        ind = tmp0 + tmp1 + tmp2 + tmp3 + tmp4
        
        LUT_equities[ind] = eq
#        LUT_equities[ind] = wins / nIters
                                
    return LUT_equities


nIters = 20000

flopCardCombs = np.array(list(itertools.combinations(np.arange(52), 5)))

st = time.time()
LUT = computeFlopEquities(flopCardCombs, nIters)
print(time.time()-st)


np.save('LUT_equity_flop', LUT)

# %%    

                                    
    
    

    
    
    
    





