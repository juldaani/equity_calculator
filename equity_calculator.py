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
from hand_eval.evaluator_numba import sortCards
from hand_eval.params import ranks_7cards, LUT_nChooseK_7cards, LUT_nChooseK_5cards, cardToInt,\
    intToCard
from equity_calculator.params import LUT_nChooseK_2cards, preflopEquities_LUT, flopEquities_LUT


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def computeEquity(holeCards, boardCards, nIters):
    # Remove hole and board cards from the deck
    mask = np.ones(52, dtype=np.bool_)
    mask[holeCards] = 0
    mask[boardCards] = 0
    deck = np.arange(52)[mask]
    nCardsDeck = len(deck)
    
    # Run simulations
    wins = 0
    curCards, opponentCards = np.zeros(7, dtype=np.uint8), np.zeros(7, dtype=np.uint8)
    for k in range(nIters):
        # Draw random cards from the deck
        for t in range(7-len(boardCards)):
            rndIdx = np.random.randint(t,nCardsDeck)
            opponentCards[t] = deck[rndIdx]
            deck[t],deck[rndIdx] = deck[rndIdx],deck[t]
        
        opponentCards[-len(boardCards):] = boardCards
        
        curCards[:5] = opponentCards[2:]    # Board cards
        curCards[-2:] = holeCards
    
        rankCur = evaluator(curCards, ranks_7cards, LUT_nChooseK_7cards)
        rankOpponent = evaluator(opponentCards, ranks_7cards, LUT_nChooseK_7cards)
    
        isWin = int(rankCur < rankOpponent)
        wins += isWin
    
    return wins / nIters


@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def computeEquities(holeCards, boardCards):
    preflopEquities = np.zeros(len(holeCards), dtype=np.float32)
    flopEquities = np.zeros(len(holeCards), dtype=np.float32)
    turnEquities = np.zeros(len(holeCards), dtype=np.float32)
    riverEquities = np.zeros(len(holeCards), dtype=np.float32)
    
    for i in prange(len(holeCards)):
        curBoardCards = boardCards[i]
        
        holeCard1 = holeCards[i,0]
        holeCard2 = holeCards[i,1]
        
        # Sort hole cards because otherwise lookup will fail
        if(holeCard1 > holeCard2):
            holeCard1, holeCard2 = holeCard2, holeCard1
        
        # Preflop equity
        ind = LUT_nChooseK_2cards[holeCard1,0] + LUT_nChooseK_2cards[holeCard2,1]
        preflopEquities[i] = preflopEquities_LUT[ind]
    
        # Flop equity
        c1,c2,c3,c4,c5,_,_ = sortCards(holeCard1, holeCard2, curBoardCards[0], curBoardCards[1],
                                       curBoardCards[2], 1000, 1000)
        tmp0 = LUT_nChooseK_5cards[c1,0]
        tmp1 = LUT_nChooseK_5cards[c2,1]
        tmp2 = LUT_nChooseK_5cards[c3,2]
        tmp3 = LUT_nChooseK_5cards[c4,3]
        tmp4 = LUT_nChooseK_5cards[c5,4]
        ind = tmp0 + tmp1 + tmp2 + tmp3 + tmp4
        flopEquities[i] = flopEquities_LUT[ind]
    
        turnEquities[i] = computeEquity(holeCards[i], curBoardCards[:4], 1500)
        riverEquities[i] = computeEquity(holeCards[i], curBoardCards, 2000)
        
    return preflopEquities, flopEquities, turnEquities, riverEquities










