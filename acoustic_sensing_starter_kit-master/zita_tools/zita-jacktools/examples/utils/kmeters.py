#!/usr/bin/python

from math import log10
from PyQt5 import QtGui, QtCore, QtWidgets


def makek20pixmap0 (sx, sy, m):
    M = QtGui.QPixmap (sx, sy + 2 * m);
    P = QtGui.QPainter (M)
    G = QtGui.QLinearGradient (0, sy + m, 0, m)
    G.setColorAt (0.00, QtGui.QColor (50, 50, 120))
    G.setColorAt (0.20, QtGui.QColor (0, 70, 90))
    G.setColorAt (0.55, QtGui.QColor (0, 90, 0))
    G.setColorAt (0.63, QtGui.QColor (100, 100, 0))
    G.setColorAt (1.00, QtGui.QColor (100, 30, 0))
    P.setPen(QtCore.Qt.NoPen)
    P.setBrush(G)
    P.drawRect (0.0, 0.0, sx, sy + 2 * m)
    return M        


def makek20pixmap1 (sx, sy, m):
    M = QtGui.QPixmap (sx, sy + 2 * m);
    P = QtGui.QPainter (M) 
    G = QtGui.QLinearGradient (0, sy + m, 0, m)
    G.setColorAt (0.00, QtGui.QColor (90, 90, 255))
    G.setColorAt (0.20, QtGui.QColor (0, 190, 190))
    G.setColorAt (0.55, QtGui.QColor (0, 255, 0))
    G.setColorAt (0.63, QtGui.QColor (255, 255, 0))
    G.setColorAt (0.75, QtGui.QColor (255, 128, 50))
    G.setColorAt (1.00, QtGui.QColor (255, 50, 50))
    P.setPen(QtCore.Qt.NoPen)
    P.setBrush(G)
    P.drawRect (0.0, 0.0, sx, sy + 2 * m)
    return M        


def makek20pixmap2 (sx, sy, m, bg):
    M = QtGui.QPixmap (sx, sy + 2 * m);
    P = QtGui.QPainter (M)
    P.setPen(QtCore.Qt.NoPen)
    P.setBrush(bg)
    P.drawRect (0.0, 0.0, sx, sy + 2 * m)
    S = [((120, 120, 255), (0.0, 1.00e-3, 3.16e-3)),
         ((  0, 255,   0), (1.00e-2, 3.16e-2)),
         ((255, 255,   0), (1.00e-1,)),
         ((255,  50,  50), (3.16e-1, 1.00))]
    P.setBrush (QtCore.Qt.NoBrush)
    for X in S:
        C = X [0]
        L = X [1];
        P.setPen (QtGui.QColor (C [0], C [1], C [2]))   
        for v in L:
           d = sy + m - K20mapfunc (v)
           P.drawLine (0, d, sx, d)
    return M        


def makek20pixmap3 (sx, sy, m, bg):
    M = QtGui.QPixmap (sx, sy + 2 * m);
    P = QtGui.QPainter (M)
    P.setPen(QtCore.Qt.NoPen)
    P.setBrush(bg)
    P.drawRect (0.0, 0.0, sx, sy + 2 * m)
    F = QtGui.QFont ('Sans', 7, QtGui.QFont.Normal)
    P.setFont (F)
    dx = sx / 2
    dy = sy + m
    P.setPen (QtGui.QColor (140, 140, 255))
    drawctext (P, dx, dy - K20mapfunc (1.00e-3), "-40");
    drawctext (P, dx, dy - K20mapfunc (3.16e-3), "-30");
    P.setPen (QtGui.QColor (0, 255, 0))
    drawctext (P, dx, dy - K20mapfunc (1.00e-2), "-20");
    drawctext (P, dx, dy - K20mapfunc (3.16e-2), "-10");
    P.setPen (QtGui.QColor (255, 255, 0))
    drawctext (P, dx, dy - K20mapfunc (1.00e-1), "0");
    P.setPen (QtGui.QColor (255, 80, 80))
    drawctext (P, dx, dy - K20mapfunc (3.16e-1), "10");
    drawctext (P, dx, dy - K20mapfunc (1.00), "20");
    return M


def drawctext (P, x, y, s):
    M = P.fontMetrics ()
    w = M.width (s)
    P.drawText (x - w / 2, y + M.ascent () // 2, s)


def K20mapfunc (v):
    if v < 1e-3: return int (12e3 * v)
    v = log10 (v) + 3
    if v < 2.0: return  int (12.1 + v * (50 + v * 8))
    if v > 3.05: v = 3.05;
    return int (v * 80 - 16)


def makek20metrics (bg):
    return ((makek20pixmap0 (8, 224, 6),
             makek20pixmap1 (8, 224, 6),
             makek20pixmap2 (8, 224, 6, bg),
             makek20pixmap3 (18, 224, 6, bg)),
             K20mapfunc, 6)

    
class K20meter (QtWidgets.QWidget):

    def __init__(self, parent, metrics, nn, d1, d2, label = None):
        super (K20meter, self).__init__(parent)
        self.pixmaps = metrics [0]
        self.mapfunc = metrics [1]
        self.margin = metrics [2]
        self.label = label
        self.nn = nn
        self.d1 = d1
        self.d2 = d2
        self.d3 = self.pixmaps [3].width () 
        self.dy = self.pixmaps [0].height ()
        self.sx = sum (nn) * (d1 + d2) + len (nn) * (d2 + self.d3) + self.d3 + 20
        self.sy = self.dy + 40
        self.rms = [0 for i in range (sum (nn))]
        self.pks = [0 for i in range (sum (nn))]
        self.resize (self.sx, self.sy)
        

    def paintEvent (self, ev):
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setPen (QtCore.Qt.NoPen)
        qp.setBrush (QtGui.QColor (255, 255, 255))
        d1 = self.d1
        d2 = self.d2
        d3 = self.d3
        i = 0
        x = 10
        y = 20
        qp.drawPixmap (x, y, self.pixmaps [3], 0, 0, d3, self.dy)
        x += d3
        for n in self.nn:
            for j in range (n):
                qp.drawPixmap (x, y, self.pixmaps [2], 0, 0, d2, self.dy)
                x += d2
                self.drawmeter (x, y, qp, i + j)
                x += d1
            qp.drawPixmap (x, y, self.pixmaps [2], 0, 0, d2, self.dy)
            x += d2
            i += n
            qp.drawPixmap (x, y, self.pixmaps [3], 0, 0, d3, self.dy)
            x += d3
        qp.end()


    def drawmeter (self, x, y, qp, ind):
        dx = self.d1
        dy1 = self.mapfunc (self.rms [ind]) + self.margin
        dy2 = self.mapfunc (self.pks [ind]) + self.margin + 1
        qp.drawPixmap (x, y + self.dy - dy1, self.pixmaps [1], 0, self.dy - dy1, dx, dy1)
        qp.drawPixmap (x, y, self.pixmaps [0], 0, 0, dx, self.dy - dy1)
        if dy2 > max (40, dy1): qp.drawRect (x, y + self.dy - dy2, dx, 3)
                      

