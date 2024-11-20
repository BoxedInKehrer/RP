# ----------------------------------------------------------------------------
#
#  Copyright (C) 2008-2018 Fons Adriaensen <fons@linuxaudio.org>
#    
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------------


# Modify as required.
#
SUFFIX := $(shell uname -m | sed -e 's/^unknown/$//' -e 's/^i.86/$//' -e 's/^x86_64/$/64/')
PREFIX ?= /usr/local
INCDIR ?= $(PREFIX)/include
LIBDIR ?= $(PREFIX)/lib$(SUFFIX)

MAJVERS = 0
MINVERS = 4.2
VERSION = $(MAJVERS).$(MINVERS)


CPPFLAGS += -D_REENTRANT -D_POSIX_PTHREAD_SEMANTICS
CXXFLAGS += -Wall -fPIC -O2 -ffast-math
CXXFLAGS += -march=native
LDFLAGS += 
LDLIBS +=


ZITA-JCLIENT_SO = libzita-jclient.so
ZITA-JCLIENT_MAJ = $(ZITA-JCLIENT_SO).$(MAJVERS)
ZITA-JCLIENT_MIN = $(ZITA-JCLIENT_MAJ).$(MINVERS)
ZITA-JCLIENT_DEP = -ljack
ZITA-JCLIENT_O = zita-jclient.o 
ZITA-JCLIENT_H = zita-jclient.h


$(ZITA-JCLIENT_MIN): $(ZITA-JCLIENT_O)
	$(CXX) -shared $(LDFLAGS) -Wl,-soname,$(ZITA-JCLIENT_MAJ) -o $(ZITA-JCLIENT_MIN) $(ZITA-JCLIENT_O) $(ZITA-JCLIENT_DEP)

$(ZITA-JCLIENT_O):	$(ZITA-JCLIENT_H)


install:	$(ZITA-JCLIENT_MIN)
	install -d $(DESTDIR)$(INCDIR)
	install -d $(DESTDIR)$(LIBDIR)
	install -m 644 $(ZITA-JCLIENT_H) $(DESTDIR)$(INCDIR)
	install -m 755 $(ZITA-JCLIENT_MIN) $(DESTDIR)$(LIBDIR)
	ldconfig
	ln -sf $(ZITA-JCLIENT_MIN) $(DESTDIR)$(LIBDIR)/$(ZITA-JCLIENT_SO)
	ln -sf $(ZITA-JCLIENT_MIN) $(DESTDIR)$(LIBDIR)/$(ZITA-JCLIENT_MAJ)

uninstall:
	/bin/rm -rf $(DESTDIR)$(INCDIR)/zita-jclient.h
	/bin/rm -rf $(DESTDIR)$(LIBDIR)/libzita-jclient*

clean:
	/bin/rm -rf *~ *.o *.a *.d *.so.* zita-jclient/*~

