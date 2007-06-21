#!/usr/bin/python

import os
from Test import *

def extractData(testname):
    test = Test(testname)

    dirlist = os.listdir(testname)
    trials = []
    for file in dirlist:
        if file.find('.log', -4) > -1:
            trials.append(int(file.split('.')[0]))
    trials.sort()

    for trial in trials:
        filename = testname + '/' + str(trial) + '.log'

        datafile = open(filename, 'r')

        line = ' '
        while line != '':
            line = datafile.readline()
            if line.find('Num levels') > -1:
                num_levels = int(line.strip().split()[-1])
                operator_info = [None]*num_levels
                nonzeros = [None]*num_levels

            if line.find('Coarsening Type =') > -1:
                # Extract the name of the coarsening algorithm.
                cg_alg = line[19:].strip()
                if cg_alg == "Falgout-CLJP":
                    pretty_name = "Falgout"
                    cg_alg = "falgout"
                elif cg_alg == "CLJP-c":
                    pretty_name = cg_alg
                    cg_alg = "cljpc"
                elif cg_alg == "Compatible Relaxation":
                    pretty_name = cg_alg
                    cg_alg = "cr"
                elif cg_alg == "Compatible Relaxation (CLJP-c Indep. Set)":
                    pretty_name = "CR (CLJP-c)"
                    cg_alg = "cr_cljpc"
                elif cg_alg == "Compatible Relaxation (PMIS-c1 Indep. Set)":
                    pretty_name = "CR (PMIS-c1)"
                    cg_alg = "cr_pmisc1"
                elif cg_alg == "PMIS":
                    pretty_name = cg_alg
                    cg_alg = "pmis"
                elif cg_alg == "HMIS":
                    pretty_name = cg_alg
                    cg_alg = "hmis"
                elif cg_alg == "PMIS-c1":
                    pretty_name = cg_alg
                    cg_alg = "pmisc1"
                elif cg_alg == "PMIS-c2":
                    pretty_name = cg_alg
                    cg_alg = "pmisc2"
                elif cg_alg == "Cleary-Luby-Jones-Plassman":
                    pretty_name = "CLJP"
                    cg_alg = "cljp"
                else:
                    pretty_name = "Unknown"
                    cg_alg = "unknown: " + cg_alg

            if line.find('Operator Matrix Information:') > -1:
                # Extract the number of rows and number of nonzeros per level.
                for i in range(4):
                    datafile.readline()
                for i in range(num_levels):
                    line = datafile.readline()
                    # This line has the information for level i.
                    line_data = line.split()
                    operator_info[i] = int(line_data[1])
                    nonzeros[i] = int(line_data[2])

            if line.find('BoomerAMG Setup:') > -1:
                # Extract setup time.
                line = datafile.readline()
                line_data = line.split()
                setup_time = float(line_data[4])

            if line.find('AMG SOLUTION INFO:') > -1:
                # Extract convergence factors and complexities.
                convergence_factors = []
                for i in range(4):
                    datafile.readline()
                line = datafile.readline()
                while line != '\n':
                    # Extract factor.
                    convergence_factors.append(float(line.split()[3]))
                    line = datafile.readline()

                # Now extract complexities.
                for i in range(3):
                    datafile.readline()
                line = datafile.readline()
                if not line.find('Complexity') > -1:
                    for i in range(5):
                        line = datafile.readline()

                c_grid = float(line.split()[3])
                line = datafile.readline()
                c_op = float(line.split()[2])
                line = datafile.readline()
                c_cycle = float(line.split()[2])

            if line.find('BoomerAMG Solve:') > -1:
                # Extract solve time and print data.
                line = datafile.readline()
                line_data = line.split()
                solve_time = float(line_data[4])

                trial_data = Trial(cg_alg, pretty_name, trial, operator_info[0], convergence_factors,
                                   c_op, c_grid, c_cycle, setup_time, solve_time, operator_info,
                                   nonzeros)

                test.addTrial(trial_data)

        datafile.close()

    return test
