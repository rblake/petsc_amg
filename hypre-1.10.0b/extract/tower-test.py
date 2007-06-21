from extractData import *

testname = "3dLap"
test = extractData(testname)

test.trial_classes['falgout'].trials[0].towerPlot()
figure()
test.trial_classes['falgout'].trials[1].towerPlot()
figure()
test.trial_classes['falgout'].trials[2].towerPlot()
figure()
test.trial_classes['falgout'].trials[3].towerPlot()
