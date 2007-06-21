#!/usr/bin/python

import sys
from extractData import *

testname = sys.argv[1]

#picklefile = open(testname + '.pcl', 'w')
#pickle.dump(test, picklefile)
#picklefile.close()

test = extractData(testname)
test.printSummary()
print
#test.matlabPlot(['cljpc', 'cr_cljpc', 'cr_pmisc1', 'falgout', 'hmis', 'pmisc1'])
#test.matlabPlot(['cljpc', 'cr_cljpc', 'falgout'])
#test.matlabPlot(['cljp', 'pmisc2', 'pmis', 'falgout', 'hmis', 'cr_pmisc1', 'cljpc', 'cr_cljpc', 'pmisc1'])
#test.matlabPlot(['cljp', 'pmisc2', 'falgout', 'cljpc', 'cr_cljpc', 'pmisc1'])
#test.matlabPlot()
#cg_alg_list = ['cljp', 'falgout', 'cr_cljpc', 'pmis', 'cr_pmisc1']
#cg_alg_list = ['cljpc', 'cr_cljpc', 'pmisc1', 'cr_pmisc1']
cg_alg_list = ['pmisc1']
## figure()
test.plot(const.WORK_PER_ACCURACY, cg_alg_list)
#figure()
#test.plot(const.C_OP, cg_alg_list)
## figure()
## test.plot(const.C_GRID, cg_alg_list)
#figure()
#test.plot(const.SETUP_TIME, cg_alg_list)
#figure()
#test.plot(const.SETUP_TIME_NORMALIZED, cg_alg_list)
## figure()
## test.plot(const.SOLVE_TIME, cg_alg_list)
#figure()
#test.plot(const.CONVERGENCE_FACTOR, cg_alg_list)
show()
