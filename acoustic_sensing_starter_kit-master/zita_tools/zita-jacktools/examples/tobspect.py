#!/usr/bin/python

import sys
import signal
from PyQt5 import QtGui, QtCore, QtWidgets
from utils.kmeters import *
from jacktools.jackiecfilt import *
from jacktools.jackkmeter import *

class Testwin(QtWidgets.QWidget):
    
    def __init__(self, metrics):
        super(Testwin, self).__init__()
        self.metrics = metrics
        self.disp = K20meter(self, metrics, (31,), 7, 6)
        self.disp.move(0, 0)
        self.disp.show()
        self.setGeometry (200, 200, self.disp.sx, self.disp.sy)
        self.setWindowTitle('Spectrum')
        self.show()        
        self.tim = QtCore.QBasicTimer ()
        self.tim.start (50, self)

    def timerEvent (self, ev):
        st, rms, dpk = K20.get_levels ()
        self.disp.rms = rms
        self.disp.pks = dpk
        self.disp.update ()

IEC = JackIECfilt (1, 31, 'Filters')
K20 = JackKmeter (31, 'Meters')
IEC.connect_input (0, 'system:capture_1')
IEC.connect_input (0, 'Signal:out')
for i in range (31):
    IEC.set_filter (0, i, JackIECfilt.OCT3, i)
    IEC.connect_output (i, 'Meters:in_%d' % (i,))

        
signal.signal (signal.SIGINT, signal.SIG_DFL)
app = QtWidgets.QApplication(sys.argv)
pal = app.palette()
bgc = QtGui.QColor (40, 40, 40)
pal.setColor(pal.Window, bgc)
app.setPalette(pal)
metrics = makek20metrics (bgc)
win = Testwin (metrics)
sys.exit(app.exec_())


