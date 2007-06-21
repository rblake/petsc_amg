import math
#import pickle
from pylab import *
import const

# Define the plot types.
const.WORK_PER_ACCURACY = 0
const.C_OP = 1
const.C_GRID = 2
const.SETUP_TIME = 3
const.SOLVE_TIME = 4
const.CONVERGENCE_FACTOR = 5
const.SETUP_TIME_NORMALIZED = 6

# Colors and line styles for plotting.
const.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
const.line_styles = ['-', ':', '--', '-.']

class Trial:
    """A class to store trial data."""

    def __init__(self, cg_alg, pretty_name, procs, dof, convergence_factors, c_op, c_grid,
                 c_cycle, setup_time, solve_time, operator_info, nonzeros):
        self.cg_alg = cg_alg
        self.pretty_name = pretty_name
        self.procs = procs
        self.dof = dof
        self.convergence_factors = convergence_factors
        self.c_op = c_op
        self.c_grid = c_grid
        self.c_cycle = c_cycle
        self.setup_time = setup_time
        self.solve_time = solve_time
        self.operator_info = operator_info
        self.nonzeros = nonzeros

    def __str__(self):
        # Print data
        out = ""
        out += self.pretty_name + "\n"
        out += "num_procs = " + str(self.procs) + "\n"
        out += "dof = " + str(self.operator_info[0]) + "\n"
        out += str(self.convergence_factors) + "\n"
        out += "c_op = " + str(self.c_op) + "\n"
        out += "c_grid = " + str(self.c_grid) + "\n"
        out += "Setup = " + str(self.setup_time) + "\n"
        out += "Solve = " + str(self.solve_time) + "\n"
        out += str(self.operator_info) + "\n"
        out += str(self.nonzeros) + "\n"
        out += '\n'
        return out

    def __cmp__(self, other):
        return cmp(self.procs, other.procs)

    def printSummary(self):
        print "num_procs = " + str(self.procs) + ", dof = " + str(self.operator_info[0])

    def lastConvergenceFactor(self):
        if abs(self.convergence_factors[-2] - self.convergence_factors[-1]) > 0.001:
            print "EEEEKEEKEKEKEKE!"
        return self.convergence_factors[-1]

    def workPerDigitAcc(self):
        #if self.cg_alg == "pmisc1":
        #    self.convergence_factors[-1] = 0.7
        return self.c_cycle / -math.log(self.lastConvergenceFactor())

    def towerPlot(self):
        hold(True)
        color = "c"
        y_lower = 0
        total_nonzeros = sum(self.nonzeros)
        for level in range(0, len(self.nonzeros)):
            level_dofs = self.operator_info[level]
            level_nonzeros = self.nonzeros[level]
            # Set color of this level.
##             if color == "c":
##                 color = "k"
##             else:
##                 color = "c"
            color = float(level_nonzeros) / pow(level_dofs, 2)
            color = (1, 1-color, 1-color)

            # Calculate width of this level.
            width = float(level_dofs) / self.operator_info[0]
            x_left = 0.5 - width/2
            x_right = 0.5 + width/2

            # Calculate upper height of this level.
            y_upper = y_lower + float(level_nonzeros) / self.nonzeros[0]

            # Draw this level.
            fill([x_left, x_right, x_right, x_left], [y_lower, y_lower, y_upper, y_upper], facecolor=color)
            y_lower = y_upper

        xlabel("# dofs (relative to fine level)")
        ylabel("Operator complexity")


class TrialClass:
    """A class to store sets of trials for a single coarse grid selection algorithm."""

    def __init__(self, cg_alg, pretty_name):
        self.cg_alg = cg_alg
        self.pretty_name = pretty_name
        self.trials = []

    def __str__(self):
        out = "====== " + self.cg_alg.upper() + " ======\n"
        for trial in self.trials:
            out += str(trial)
        return out

    def printSummary(self):
        print "====== " + self.cg_alg.upper() + " ======"
        for trial in self.trials:
            trial.printSummary()

    def addTrial(self, trial):
        self.trials.append(trial)

    def plot(self, plot_data, line_style):
        dofs = []
        data = []
        for trial in self.trials:
            dofs.append(trial.dof)
            if plot_data == const.WORK_PER_ACCURACY:
                data.append(trial.workPerDigitAcc())
            elif plot_data == const.C_OP:
                data.append(trial.c_op)
            elif plot_data == const.C_GRID:
                data.append(trial.c_grid)
            elif plot_data == const.SETUP_TIME:
                data.append(trial.setup_time)
            elif plot_data == const.SETUP_TIME_NORMALIZED:
                data.append(trial.setup_time / self.trials[0].setup_time)
            elif plot_data == const.SOLVE_TIME:
                data.append(trial.solve_time)
            elif plot_data == const.CONVERGENCE_FACTOR:
                data.append(trial.lastConvergenceFactor())
        line, = semilogx(dofs, data, line_style)
        line.set_antialiased(True)
        line.set_linewidth(2)
        print self.cg_alg + "_dofs = " + str(dofs)
        print self.cg_alg + "_data = " + str(data)
        print "line, = semilogx(" + self.cg_alg + "_dofs, " + self.cg_alg + "_data, \'" + line_style + "\')"
        print "line.set_antialiased(True)"
        print "line.set_linewidth(2)"


    def matlabPlot(self):
        dofs = []
        work_per_acc = []
        for trial in self.trials:
            dofs.append(trial.dof)
            work_per_acc.append(trial.workPerDigitAcc())

        print self.cg_alg + "_dofs = " + str(dofs) + ";"
        print self.cg_alg + "_work_per_acc = " + str(work_per_acc) + ";"
        plot_command = self.cg_alg + "_dofs, " + self.cg_alg + "_work_per_acc"
        return plot_command


class Test:
    """A class to store results for entire tests, such as results on a 2D Laplacian problem run on different coarse grid selection algorithms and various problem sizes."""

    def __init__(self, test):
        self.test = test
        self.trial_classes = {}

    def __str__(self):
        out = "****** " + self.test + " ******\n"
        for trial_class in self.trial_classes:
            out += str(self.trial_classes[trial_class])
        return out

    def printSummary(self):
        print "****** Test: " + self.test + " ******"
        for trial_class in self.trial_classes:
            print
            self.trial_classes[trial_class].printSummary()

    def getTrialClass(self, trial_class_name, trial_class_pretty_name):
        """This function will retrieve a trial class from self.trial_classes, if it exists. If that trial does not exist, it will create that trial class, add it to self.trial_classes, and return it."""
        if self.trial_classes.has_key(trial_class_name):
            return self.trial_classes[trial_class_name]
        else:
            new_trial_class = TrialClass(trial_class_name, trial_class_pretty_name)
            self.trial_classes[trial_class_name] = new_trial_class
            return new_trial_class

    def addTrial(self, trial):
        """This function will add a trial to the specified trial class. If that trial class does not exist, a new one will be created."""
        trial_class = self.getTrialClass(trial.cg_alg, trial.pretty_name)
        trial_class.addTrial(trial)

    def lineStyle(self, line_number):
        """Select a plot line style based on line_number."""
        style = const.colors[line_number % len(const.colors)]
        style += const.line_styles[line_number / len(const.colors)]
        style += 'o'
        return style

    def plot(self, plot_data, selected_classes=[]):
        if len(selected_classes) == 0:
            selected_classes = self.trial_classes.keys()

        line_number = 0
        plot_command = []
        legend_text = []
        hold(True)
        
        for trial_class in selected_classes:
            if not self.trial_classes.has_key(trial_class):
                break
            self.trial_classes[trial_class].plot(plot_data, self.lineStyle(line_number))
            line_number += 1
            legend_text.append(self.trial_classes[trial_class].pretty_name)
        # Set title and labels.
        if plot_data == const.WORK_PER_ACCURACY:
            ylabel_text = 'Work per Digit of Accuracy'
        elif plot_data == const.C_OP:
            ylabel_text = 'Operator Complexity'
        elif plot_data == const.C_GRID:
            ylabel_text = 'Grid Complexity'
        elif plot_data == const.SETUP_TIME:
            ylabel_text = 'Setup Time'
        elif plot_data == const.SETUP_TIME_NORMALIZED:
            ylabel_text = 'Normalized Setup Time'
        elif plot_data == const.SOLVE_TIME:
            ylabel_text = 'Solve Time'
        elif plot_data == const.CONVERGENCE_FACTOR:
            ylabel_text = 'Asymptotic Convergence Factor'

        xlabel('Degrees of Freedom')
        print "xlabel(\'Degrees of Freedom\')"
        ylabel(ylabel_text)
        print "ylabel(\'" + ylabel_text + "\')"
        print "legend(" + str(legend_text) + ", loc=\'best\')"
        print
        legend(legend_text, loc='best')

    def matlabPlot(self, selected_classes=[]):
        if len(selected_classes) == 0:
            selected_classes = self.trial_classes.keys()
            
        print "hold on"
        plot_command = "plot("
        legend = "legend("
        for trial_class in selected_classes:
            if not self.trial_classes.has_key(trial_class):
                break
            plot_command += self.trial_classes[trial_class].matlabPlot() + ", "
            legend += "\'" + self.trial_classes[trial_class].pretty_name + "\', "
        plot_command = plot_command[0:-2]
        plot_command += ")"
        print plot_command
        legend = legend[0:-2]
        legend += ")"
        print legend
