#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:49:17 2019

@author: rajdua
"""


import pandas as pd
import numpy as np
import time as time

size = 1000


start = time.time()

a  = np.arange(1, size + 1)

b = np.arange(1,size + 1)

answer = np.dot(a[:,None],b[None,:])

end  = time.time()

print(end-start)



