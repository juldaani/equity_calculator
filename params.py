#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:17:43 2019

@author: juho
"""

import numpy as np
import scipy.special
import sys

LUT_nChooseK_2cards = np.zeros((52,2), dtype=np.int64)
LUT_nChooseK_2cards[:,0] = scipy.special.comb(np.arange(52), np.full(52,1)).astype(np.int)
LUT_nChooseK_2cards[:,1] = scipy.special.comb(np.arange(52), np.full(52,2)).astype(np.int)

preflopEquities_LUT = None
try:
    preflopEquities_LUT = np.load('/home/juho/dev_folder/equity_calculator/LUT_equity_preflop.npy')
except FileNotFoundError as err:
    print('VIRHE! Cannot load lookup table for preflop equities. Run create_preflop_LUT.py. \
          before proceeding.\n')
    print(err)

flopEquities_LUT = None
try:
    flopEquities_LUT = np.load('/home/juho/dev_folder/equity_calculator/LUT_equity_flop.npy')
except FileNotFoundError as err:
    print('VIRHE! Cannot load lookup table for flop equities. Run create_flop_LUT.py. \
          before proceeding.\n')
    print(err)
