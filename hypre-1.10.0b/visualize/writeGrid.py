#!/usr/bin/python

import sys

def writeGrid(nx, ny, nz):
    """Write a file containing the nodal positions of a Cartisian grid."""
    position_data = open('node-positions', 'w')

    # Set the divisor used in the loop. This is necessary because if any of
    # the parameters are 1, then (n* - 1) is zero and causes a divide by zero.
    if nx == 1:
        divx = 1
    else:
        divx = nx-1
        
    if ny == 1:
        divy = 1
    else:
        divy = ny-1

    if nz == 1:
        divz = 1
    else:
        divz = nz-1

    for step_z in range(nz):
        for step_y in range(ny):
            for step_x in range(nx):
                
                line = str(float(step_x) / divx) + "\t" + str(float(step_y) / divy) + "\t" + str(float(step_z) / (divz)) + "\n"
                position_data.write(line)
    position_data.close()


writeGrid(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
