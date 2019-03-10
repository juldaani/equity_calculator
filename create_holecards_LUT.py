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
def computeHoleCardsEquities(holeCardCombs, LUT_nChooseK_2cards):
    LUT_equityHoleCards = np.zeros(len(holeCardCombs), dtype=np.float32)
    
    tmp2 = np.zeros(2, dtype=np.uint64)

    for i in prange(len(holeCardCombs)):
        print(i,len(holeCardCombs))
        
        curHoleCards = holeCardCombs[i]

        # Remove current hole cards from the deck
        mask = np.ones(52, dtype=np.bool_)
        mask[curHoleCards] = 0
        deck = np.arange(52)[mask]
        
        # Loop over all possible 7 card combinations (board cards + opponent hole cards)
        # from the deck (for current hole cards)
        counter, wins = 0, 0
        tmp = np.zeros(7, dtype=np.uint64)
        curCards, opponentCards = np.zeros(7, dtype=np.uint8), np.zeros(7, dtype=np.uint8)
        for i1 in range(len(deck)):
            c1 = deck[i1]
            
            for i2 in range(i1+1,len(deck)):
                c2 = deck[i2]
            
                for i3 in range(i2+1,len(deck)):
                    c3 = deck[i3]
                    
                    for i4 in range(i3+1,len(deck)):
                        c4 = deck[i4]
                        
                        for i5 in range(i4+1,len(deck)):
                            c5 = deck[i5]
                            
                            for i6 in range(i5+1,len(deck)):
                                c6 = deck[i6]
                                
                                for i7 in range(i6+1,len(deck)):
                                    c7 = deck[i7]
        
                                    counter += 1
#                                    if(counter % 10000000 == 0): print(counter/1000000)
        
                                    opponentCards[0], opponentCards[1], opponentCards[2], \
                                        opponentCards[3], opponentCards[4], opponentCards[5], \
                                        opponentCards[6] = c1,c2,c3,c4,c5,c6,c7
                                    
                                    curCards[0], curCards[1], curCards[2], \
                                        curCards[3], curCards[4], curCards[5], \
                                        curCards[6] = curHoleCards[0],curHoleCards[1],c1,c2,c3,c4,c5
                                    
                                    rankCur = evaluator(curCards, ranks_7cards, LUT_nChooseK_7cards, tmp)
                                    rankOpponent = evaluator(opponentCards, ranks_7cards, 
                                                             LUT_nChooseK_7cards, tmp)
                                    
                                    isWin = int(rankCur < rankOpponent)
                                    wins += isWin
        
        # Get index to equity LUT
        tmp2[0] = LUT_nChooseK_2cards[curHoleCards[0],0]
        tmp2[1] = LUT_nChooseK_2cards[curHoleCards[1],1]
        ind = np.sum(tmp2)
        
        LUT_equityHoleCards[ind] = wins / counter
                                
    return LUT_equityHoleCards


def convertIntToCard(ints):
    return [intToCard[intt] for intt in ints]



holeCardCombs = np.array(list(itertools.combinations(np.arange(52), 2)))

st = time.time()
LUT = computeHoleCardsEquities(holeCardCombs, LUT_nChooseK_2cards)
print(time.time()-st)

np.save('LUT_equity_hole_cards', LUT)

# %%    

                                    
    
    

    
    
    
    





