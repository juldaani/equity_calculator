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
        curHoleCards = holeCards[i]
        curBoardCards = boardCards[i]
        
        holeCard1 = curHoleCards[0]
        holeCard2 = curHoleCards[1]
        
        # Sort hole cards because otherwise lookup will fail
        if(holeCard1 > holeCard2):
            holeCard1, holeCard2 = holeCard2, holeCard1
        
        # Preflop equity
        ind = LUT_nChooseK_2cards[holeCard1,0] + LUT_nChooseK_2cards[holeCard2,1]
        preflopEquities[i] = holeCardEquities[ind]
    
        # Flop equity
        flopEquities[i] = computeEquity(curHoleCards, curBoardCards[:3], 10000)
        turnEquities[i] = computeEquity(curHoleCards, curBoardCards[:4], 1500)
        riverEquities[i] = computeEquity(curHoleCards, curBoardCards, 1000)
        
    return preflopEquities, flopEquities, turnEquities, riverEquities








