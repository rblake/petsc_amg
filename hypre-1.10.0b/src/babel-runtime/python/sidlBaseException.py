#
# File:  sidlBaseException.py
# Copyright (c) 2005 The Regents of the University of California
# $Revision: 1.4 $
# $Date: 2005/11/14 21:20:12 $
#

class sidlBaseException(Exception):
    """Base class for all SIDL Exception classes"""

    def __init__(self, exception):
        self.exception = exception
