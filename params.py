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


try:
    holeCardEquities = np.load('/home/juho/dev_folder/equity_calculator/LUT_equity_hole_cards.npy')
except FileNotFoundError as err:
    print('VIRHE! Cannot load lookup table for hole card equities. Run create_holecards_LUT.py. \
          before proceeding.\n')
    print(err)
    sys.exit()
    
