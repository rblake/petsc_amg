class Node:
    """A graph node class"""

    def __init__(self, nodeID=-1, C_point=-1, neighbors=[]):
        self.nodeID = nodeID
        self.C_point = C_point
        self.neighbors = neighbors
        self.color = -1
    
    def __str__(self):
        return str(self.nodeID)

    def __repr__(self):
        return self.__str__()

    def __cmp__(self, other):
        # Allows the nodes to be sorted by nodeID.
        return cmp(self.nodeID, other.nodeID)

########################################################################

class NodeSet(list):
    """A class containing a set of nodes"""

    def __init__(self):
        self.colored = False
        self.cr_rates = False


########################################################################

class NodePosition:
    def __init__(self, nodeID=-1, pos=[]):
        self.nodeID = nodeID
        self.pos = pos

    def __cmp__(self, other):
        # Allows the nodes to be sorted by nodeID.
        return cmp(self.nodeID, other.nodeID)

    def __str__(self):
        return str(self.nodeID)

    def __repr__(self):
        return self.__str__()
