#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:50:09 2019

@author: juho

Texas holdem equity calculator
"""

import numpy as np
from numba import jit, prange

from hand_eval.evaluator_numba import evaluate_numba2 as evaluator
from hand_eval.params import ranks_7cards, LUT_nChooseK_7cards, cardToInt, intToCard
from equity_calculator.params import LUT_nChooseK_2cards, holeCardEquities





# %%


cards = np.random.choice(52, size=7, replace=0)
holeCards = cards[:2]
flopCards = cards[2:5]
turnCards = cards[2:6]
riverCards = cards[2:7]


# Sort hole cards because otherwise lookup will fail
if(holeCards[0] > holeCards[1]):
    holeCards[0], holeCards[1] = holeCards[1], holeCards[0]

# Preflop equity
tmp0 = LUT_nChooseK_2cards[holeCards[0],0]
tmp1 = LUT_nChooseK_2cards[holeCards[1],1]
ind = tmp0 + tmp1
holeCardEquity = holeCardEquities[ind]



# Remove hole and board cards from the deck
boardCards = flopCards
mask = np.ones(52, dtype=np.bool_)
mask[holeCards] = 0
mask[boardCards] = 0
deck = np.arange(52)[mask]
np.random.shuffle(deck)

# %%

curCards, opponentCards = np.zeros(7, dtype=np.uint8), np.zeros(7, dtype=np.uint8)

counter, wins = 0, 0
for i1 in range(len(deck)):
    c1 = deck[i1]
    
    for i2 in range(i1+1,len(deck)):
        c2 = deck[i2]
    
        for i3 in range(i2+1,len(deck)):
            c3 = deck[i3]
            
            for i4 in range(i3+1,len(deck)):
                c4 = deck[i4]
                
                opponentCards[0], opponentCards[1], opponentCards[2], \
                    opponentCards[3], opponentCards[4], opponentCards[5], opponentCards[6] = \
                    boardCards[0],boardCards[1],boardCards[2],c1,c2,c3,c4
                
                curCards[0], curCards[1], curCards[2], curCards[3], \
                    curCards[4], curCards[5], curCards[6] = \
                    holeCards[0],holeCards[1],boardCards[0],boardCards[1],boardCards[2],c1,c2
                
                rankCur = evaluator(curCards, ranks_7cards, LUT_nChooseK_7cards)
                rankOpponent = evaluator(opponentCards, ranks_7cards, 
                                         LUT_nChooseK_7cards)
                
                isWin = int(rankCur < rankOpponent)
                wins += isWin

                counter += 1
                
equity = wins / counter

print(equity)

# %%

pp = 10000

counter, wins = 0, 0
for i in range(pp):
    idx = np.random.choice(47, size=4, replace=0)
    
    c1,c2,c3,c4 = deck[idx[0]],deck[idx[1]],deck[idx[2]],deck[idx[3]]
    
    opponentCards[0], opponentCards[1], opponentCards[2], \
                    opponentCards[3], opponentCards[4], opponentCards[5], opponentCards[6] = \
                    boardCards[0],boardCards[1],boardCards[2],c1,c2,c3,c4
                
    curCards[0], curCards[1], curCards[2], curCards[3], \
        curCards[4], curCards[5], curCards[6] = \
        holeCards[0],holeCards[1],boardCards[0],boardCards[1],boardCards[2],c1,c2
    
    rankCur = evaluator(curCards, ranks_7cards, LUT_nChooseK_7cards)
    rankOpponent = evaluator(opponentCards, ranks_7cards, 
                             LUT_nChooseK_7cards)
    
    isWin = int(rankCur < rankOpponent)
    wins += isWin

    counter += 1

equity = wins / counter
print(equity)


# 0.7466


# %%

def convertIntToCard(ints):
    return [intToCard[intt] for intt in ints]

import itertools

holeCardCombs = np.array(list(itertools.combinations(np.arange(52), 2)))

# %%

ii = 645

cur = holeCardCombs[ii]
cur = [0+13*1,13*3]

tmp0 = LUT_nChooseK_2cards[cur[0],0]
tmp1 = LUT_nChooseK_2cards[cur[1],1]
ind = tmp0 + tmp1


print(holeCardEquities[ind])
print(convertIntToCard(cur))






