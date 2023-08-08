#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:54:03 2023

@author: crw
"""

import numpy as np
def avg_min_max(r, window_size):
    r_smoothed = [np.mean(r[lv-window_size:lv+window_size]) for lv in range(window_size,len(r)-window_size)]
    r_min = [np.min(r[lv-window_size:lv+window_size]) for lv in range(window_size,len(r)-window_size)]
    r_max = [np.max(r[lv-window_size:lv+window_size]) for lv in range(window_size,len(r)-window_size)]
    
    return r_smoothed, r_min, r_max
    