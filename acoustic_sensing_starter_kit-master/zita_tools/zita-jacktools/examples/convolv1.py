#!/usr/bin/python

from jacktools.jackconvolv import *
from audiotools.audiofile import *
from time import sleep

C = JackConvolv (4, 4, 'Conv')

C.connect_input (0, 'afp-st:out_1')
C.connect_input (1, 'afp-st:out_2')
C.connect_input (2, 'Conv:out_0')
C.connect_input (3, 'Conv:out_1')

C.connect_output (0, 'zita-mu1:in_2.L')
C.connect_output (1, 'zita-mu1:in_2.R')
C.connect_output (2, 'zita-mu1:in_3.L')
C.connect_output (3, 'zita-mu1:in_3.R')

C.configure (250000, 0.25)

A, fs, ft = read_audio ("weird.wav");

C.impdata_create (A [:,0], 0, 1)
C.impdata_create (A [:,1], 1, 0)
C.impdata_link (0, 1, 2, 3)
C.impdata_link (1, 0, 3, 2)
    
C.process ()
sleep (10000)
C.silence ()
del C


