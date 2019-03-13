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
from hand_eval.evaluator_numba import evaluate_numba2 as evaluator
from hand_eval.params import ranks_7cards, LUT_nChooseK_7cards, cardToInt, intToCard


@jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def computeHolePreflopEquities(holeCardCombs, LUT_nChooseK_2cards, nIters):
    LUT_equityHoleCards = np.zeros(len(holeCardCombs), dtype=np.float32)
    
    for i in prange(len(holeCardCombs)):
        print(i,len(holeCardCombs))
        
        curHoleCards = holeCardCombs[i]

        # Remove current hole cards from the deck
        mask = np.ones(52, dtype=np.bool_)
        mask[curHoleCards] = 0
        deck = np.arange(52)[mask]
        nCardsDeck = len(deck)
        
        # Run simulations
        wins = 0
        curCards, opponentCards = np.zeros(7, dtype=np.uint8), np.zeros(7, dtype=np.uint8)
        for k in range(nIters):
            # Draw random cards from the deck
            for t in range(7):
                rndIdx = np.random.randint(t,nCardsDeck)
                opponentCards[t] = deck[rndIdx]
                deck[t],deck[rndIdx] = deck[rndIdx],deck[t]
            
            curCards[:2] = curHoleCards
            curCards[2:] = opponentCards[:5]    # Board cards

            rankCur = evaluator(curCards, ranks_7cards, LUT_nChooseK_7cards)
            rankOpponent = evaluator(opponentCards, ranks_7cards, 
                                     LUT_nChooseK_7cards)
    
            isWin = int(rankCur < rankOpponent)
            wins += isWin
            
        # Get index to equity LUT
        tmp0 = LUT_nChooseK_2cards[curHoleCards[0],0]
        tmp1 = LUT_nChooseK_2cards[curHoleCards[1],1]
        ind = tmp0 + tmp1
        
        LUT_equityHoleCards[ind] = wins / nIters
                                
    return LUT_equityHoleCards


nIters = 10000000

holeCardCombs = np.array(list(itertools.combinations(np.arange(52), 2)))

st = time.time()
LUT = computeHoleCardsEquities(holeCardCombs, LUT_nChooseK_2cards, nIters)
print(time.time()-st)


np.save('LUT_equity_hole_cards', LUT)

# %%    

                                    
    
    

    
    
    
    





