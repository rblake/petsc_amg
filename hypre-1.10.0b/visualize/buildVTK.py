#!/usr/bin/python

import re
import os
from Node import *

def getRunInfo():
    """Read the information from the run-info file."""
    info = open('run-info', 'r')

    # Remove header from info file.
    info.readline()

    # Get data from info file.
    [dofs, procs, levels] = re.compile("\d+").findall(info.readline())
    
    info.close()
    os.system('rm run-info')

    return [int(dofs), int(procs), int(levels)]


########################################################################

def getCoarsenData(dofs, procs, levels):
    num_C_points = 0
    numbers_regexp = re.compile("-?\d+")  # used to extract numbers from
                                        # input read from file
    C_map = range(dofs) # C_map gives a mapping from a C-point's value on
                        # the current level to its value on the fine level
    grid_hierarchy = []

    for level in range(levels-1):
        nodes = NodeSet()
        prev_C_map = C_map[:]
        C_map = []

        for proc in range(procs):
            filename = "coarsen.out." + str(proc) + "." + str(level)
            coarsen_data = open(filename, 'r')

            # Begin reading the information from this file. The format of the
            # file is:
            #
            #    nodeID C/F neighbor_list
            #
            # where nodeID is the ID on the local level. C_map is used to
            # determine the global nodeID. C/F is 1 for C-point or -1 for
            # F-point, and neighbor_list is the (local) nodeIDs of the
            # connected nodes.
            for line in coarsen_data:
                line_data = numbers_regexp.findall(line)
                nodeID = int(line_data[0])
                CF = int(line_data[1])
                neighbors = []
                for neighbor in line_data[2:]:
                    neighbors.append(int(neighbor))
                
                # Replace the local nodeIDs (the IDs of the node on the
                # current level) with global nodeIDs.
                if level > 0:
                    #nodeID = grid_hierarchy[0][nodeID].nodeID
                    nodeID = prev_C_map[nodeID]
                    #for index in range(len(neighbors)):
                    #    neighbors[index] = grid_hierarchy[0][neighbors[index]].nodeID

                neighbors.sort()

                # Change C/F to 1/0. In output from hypre it is 1/-1.
                if CF == -1:
                    CF = 0

                # Add an entry to the map used on the next level if this node
                # is a C-point.
                if CF == 1:
                    #C_map.append(prev_C_map[nodeID])
                    C_map.append(nodeID)
                        
                # Create object containing this node.
                node = Node(nodeID, CF, neighbors)

                # Append this node to the node list.
                nodes.append(node)
        
            coarsen_data.close()
            rmcomm = 'rm ' + filename
            os.system(rmcomm)

        # Append data from this level to grid_hierarchy.
        grid_hierarchy.append(nodes)

    return grid_hierarchy


########################################################################

def getColorData(grid_hierarchy, procs, levels):
    numbers_regexp = re.compile("\d+")  # used to extract numbers from
                                             # input read from file

    for level in range(levels-1):
        for proc in range(procs):
            filename = "color.out." + str(proc) + '.' + str(level)
            try:
                color_data = open(filename, 'r')
            except IOError:
                # The file does not exist, which means that the coarsening
                # algorithm used does not precolor the graph. Nothing to do.
                return

            grid_hierarchy[level].colored = True
            for line in color_data:
#                line_data = numbers_regexp.findall(line)
                line_data = line.split()
                node = grid_hierarchy[level][int(line_data[0])]

                # Mark the node's color.
                node.color = int(line_data[1])

                if int(line_data[0]) == len(grid_hierarchy[level])-1:
                    break;

            color_data.close()
            rmcomm = 'rm ' + filename
            os.system(rmcomm)



########################################################################

def getCRData(grid_hierarchy, procs, levels):
    for level in range(levels-1):
        for proc in range(procs):
            filename = "cr-rates.out." + str(proc) + '.' + str(level)
            try:
                cr_data = open(filename, 'r')
            except IOError:
                # The file does not exist, which means that CR was not
                # used. Nothing to do.
                return

            grid_hierarchy[level].cr_rates = True
            for line in cr_data:
                line_data = line.split()
                node = grid_hierarchy[level][int(line_data[0])]

                # Mark the node's CR rate.
                node.cr_rate = float(line_data[1])

                if int(line_data[0]) == len(grid_hierarchy[level])-1:
                    break;

            cr_data.close()
            rmcomm = 'rm ' + filename
            os.system(rmcomm)



########################################################################

def getPositionData():
    numbers_regexp = re.compile("-?\d+\S*")  # used to extract numbers from
                                             # input read from file
    position_data = open('node-positions', 'r')
    positions = []
    for line in position_data:
        coords = numbers_regexp.findall(line)
        pos = []
        for coord in coords:
            pos.append(float(coord))
        positions.append(pos)
        
    position_data.close()
    return positions


########################################################################

def getPartitionData(dofs):
    partitions = []
    try:
        partition_data = open('node-partitions', 'r')
    except IOError:
        # The file does not exist, which means that the coarsening
        # algorithm used does not precolor the graph. Nothing to do.
        partitions = [0] * dofs
        return partitions
    for line in partition_data:
        partitions.append(int(line))
    return partitions


########################################################################

def writeVTK(positions, partitions, levels, grid_hierarchy):
    for level in range(levels-1):
        filename = "cgrid" + str(level) + ".vtk"
        cg_vtk_file = open(filename, 'w')

        level_colored = grid_hierarchy[level].colored
        cr_rates = grid_hierarchy[level].cr_rates

        # Write VTK header information.
        cg_vtk_file.write("# vtk DataFile Version 2.0\n")
        cg_vtk_file.write("Coarse Grid Visualization\n")
        cg_vtk_file.write("ASCII\n")
        cg_vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        # Build the node position list, node partition list, cell list, and
        # gather information about C/F for each node.
        cells = []
        CF = ""
        if(level_colored):
            colors = ""
        if(cr_rates):
            crrates = ""
        coords = ""
        parts = ""
        index = -1  # used to determine node's local ID
        for node in grid_hierarchy[level]:
            index += 1
            CF += str(node.C_point) + "\n"
            if(level_colored):
                colors += str(node.color) + "\n"
            if(cr_rates):
                crrates += str(node.cr_rate) + "\n"
            coords += str(positions[node.nodeID][0]) + " " + str(positions[node.nodeID][1]) + " " + str(positions[node.nodeID][2]) + "\n"
            parts += str(partitions[node.nodeID]) + "\n"
            for neighbor in node.neighbors:
                if neighbor < node.nodeID:
                    cells.append([index, neighbor])
                else:
                    break

        # Write the node positions.
        cg_vtk_file.write("POINTS " + str(len(grid_hierarchy[level])) + " float\n")
        cg_vtk_file.write(coords)

        # Now write the cell list.
        cg_vtk_file.write("CELLS " + str(len(cells)) + " " + str(3*len(cells)) + "\n")
        for cell in cells:
            cg_vtk_file.write("2 " + str(cell[0]) + " " + str(cell[1]) + "\n")

        # Write the cell types list. This is just "3" for each cell. This
        # represents that each cell is a line.
        cg_vtk_file.write("CELL_TYPES " + str(len(cells)) + "\n")
        cg_vtk_file.write("3\n"*len(cells))

        # Write point data (and point color information, if applicable).
        cg_vtk_file.write("POINT_DATA " + str(len(grid_hierarchy[level])) + "\n")
        cg_vtk_file.write("SCALARS C/F int 1\n")
        cg_vtk_file.write("LOOKUP_TABLE default\n")
        cg_vtk_file.write(CF)
        if(level_colored):
            cg_vtk_file.write("SCALARS Colors int 1\n")
            cg_vtk_file.write("LOOKUP_TABLE default\n")
            cg_vtk_file.write(colors)
        if(cr_rates):
            cg_vtk_file.write("SCALARS F_Damping_Rates float 1\n")
            cg_vtk_file.write("LOOKUP_TABLE default\n")
            cg_vtk_file.write(crrates)

        # Write the processor partition information.
        cg_vtk_file.write("SCALARS Partitions int 1\n")
        cg_vtk_file.write("LOOKUP_TABLE default\n")
        cg_vtk_file.write(parts)

        cg_vtk_file.close()


########################################################################

# Knowing the number of dofs, processors, and levels allows us to know
# which files exist and contain the grid hierarchy information.
[dofs, procs, levels] = getRunInfo()

# Gather information about the positions of the nodes. These positions
# are the same on all grid levels.
positions = getPositionData()

# Get information about which partition each node is assigned to.
partitions = getPartitionData(dofs)

# Gather information about the nodes on each coarse level (and their
# global node IDs).
grid_hierarchy = getCoarsenData(dofs, procs, levels)

# Gather information about the colors of the nodes (if it exists).
getColorData(grid_hierarchy, procs, levels)

# Gather information about the CR rate of each node (if it exists).
getCRData(grid_hierarchy, procs, levels)

# Now write the VTK files (coarse grids and colors, if applicable).
writeVTK(positions, partitions, levels, grid_hierarchy)
