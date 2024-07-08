import matplotlib.pyplot as plt
import numpy as np

part_1 = False

if part_1 == True:
    ###############################################################
    species = ("Genenator 1", "Generator 2", "Generator 3")
    penguin_means = {
        'p (%)': (0.40, 0.40, 0.20),
        'alpha (%)': (0, 0, 1),
        'beta (%)': (0, 0, 0),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0


    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Energy/Reserve supply (%)')
    ax.set_title('CC OPF - Primal Results')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.show()

    ###############################################################
    ###############################################################
    ###############################################################

    species = ("Genenator 1", "Generator 2", "Generator 3")
    penguin_means = {
        'p (%)': (0.40, 0.35, 0.25),
        'alpha (%)': (0, 0.5, 0.5),
        'beta (%)': (0, 0.5, 0.5),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0


    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Energy/Reserve supply (%)')
    ax.set_title('LDT-CC OPF - Primal Results')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.show()


    ###############################################################
    ###############################################################
    ###############################################################
    species = ("Genenator 1", "Generator 2", "Generator 3")
    penguin_means = {
        'p (%)': (0.398, 0.332, 0.270),
        'alpha (%)': (0, 1, 0),
        'beta (%)': (1, 0, 0),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Energy/Reserve supply (%)')
    ax.set_title('WCC-LDT OPF - Primal Results')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.show()

'''
###############################################################
################    Network Test     ##########################
###############################################################

###############################ED################################

species = ("Genenator 1", "Generator 2", "Generator 3")
penguin_means = {
    'p (%)': (0.40, 0.40, 0.20),
    'alpha (%)': (0, 0, 1),
    'beta (%)': (0, 0, 0),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0


fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Energy/Reserve supply (%)')
ax.set_title('CC OPF - Primal Results')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = ["L(0,1)","L(1,2)","L(0,2)"]
y = np.array([-15.71, 44.28, 35.71])

plt.plot(x, y)
ax.set_title('CC OPF - Primal Results')
plt.ylabel("Power Flow [MW]")
plt.xlabel("Lines")

plt.show()


#############################WCC##################################

species = ("Genenator 1", "Generator 2", "Generator 3")
penguin_means = {
    'p (%)': (0.40, 0.326, 0.274),
    'alpha (%)': (0, 1, 0),
    'beta (%)': (0, 0, 0),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0


fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Energy/Reserve supply (%)')
ax.set_title('WCC OPF - Primal Results')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = ["L(0,1)","L(1,2)","L(0,2)"]
y = np.array([-18.36, 49.04, 38.36])

plt.plot(x, y)
plt.ylabel("Power Flow [MW]")
ax.set_title('CC OPF - Primal Results')
plt.ylabel("Power Flow [MW]")
plt.xlabel("Lines")

plt.show()

#############################LDT##################################

species = ("Genenator 1", "Generator 2", "Generator 3")
penguin_means = {
    'p (%)': (0.40, 0.347, 0.253),
    'alpha (%)': (0, 0.530, 0.470),
    'beta (%)': (0, 0.530, 0.470),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0


fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Energy/Reserve supply (%)')
ax.set_title('LDT-CC OPF - Primal Results')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = ["L(0,1)","L(1,2)","L(0,2)"]
y = np.array([-17.60, 47.69, 37.60])

plt.plot(x, y)
plt.ylabel("Power Flow [MW]")
ax.set_title('CC OPF - Primal Results')
plt.ylabel("Power Flow [MW]")
plt.xlabel("Lines")

plt.show()

'''

###############NEW Attempt######################

#############################Energy##################################
color = [(63/235, 136/235, 197/235), (201/235, 55/235, 56/235), (235/235, 157/235, 55/235)]
species = ('CT', 'ME', 'NEMASSBOST', 'NH', 'RI', 'SEMASS', 'VT', 'WCMASS')
penguin_means = {
    'CC': (2258.1621365,  2890.54522028,0,2247.6,1843.8,1909.420421,620.2,382.7),
    'WCC-LDT': (2258.1621365,2890.54522028,0,2247.6,1843.8,1909.420421,620.2,382.7),
    'LDT': (2258.565093,2892.63658992,0,2247.27703787,1842.60365404,1908.63989846,620.1413011,382.56420335),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0


fig, ax = plt.subplots(layout='constrained')
counter = 0
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color[counter])
    #ax.bar_label(rects, padding=3)
    multiplier += 1
    counter += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Energy supply (MW)')
#ax.set_title('Energy supply (MW)')
ax.set_xticks(x + width, species)
ax.legend(loc='upper right', ncols=3)
#ax.set_ylim(0, 1)

plt.show()

#############################Standard Reserve##################################

penguin_means = {
    'CC': (1,0,0,0,0,0,0,0),
    'WCC-LDT': (1,0,0,0,0,0,0,0),
    'LDT': (0.3238,0.2634,0,0,0.1801,0.1008,0,0.1318),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0


fig, ax = plt.subplots(layout='constrained')
counter = 0
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color[counter])
    #ax.bar_label(rects, padding=3)
    multiplier += 1
    counter += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Energy/Reserve supply (%)')
#ax.set_title('Standard Reserve supply (%)')
ax.set_xticks(x + width, species)
ax.legend(loc='upper right', ncols=3)
#ax.set_ylim(0, 1)

plt.show()

#############################Extreme Reserve##################################

penguin_means = {
    'CC': (0,0,0,0,0,0,0,0),
    'WCC-LDT': (1,0,0,0,0,0,0,0),
    'LDT': (0.3037,0.1803,0,0,0.4013,0.0918,0,0.0226),
}


x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0


fig, ax = plt.subplots(layout='constrained')

counter = 0
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color[counter])
    #ax.bar_label(rects, padding=3)
    multiplier += 1
    counter += 1


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Reserve supply (%)')
#ax.set_title('Extreme Reserve supply (%)')
ax.set_xticks(x + width, species)
ax.legend(loc='upper right', ncols=3)
#ax.set_ylim(0, 1)

plt.show()

'''
[2258.1621 2890.5452    0.     2247.6    1843.8    1909.4204  620.2
  382.7   ]
[1. 0. 0. 0. 0. 0. 0. 0.]
[0. 0. 0. 0. 0. 0. 0. 0.]
[2258.1621 2890.5452    0.     2247.6    1843.8    1909.4204  620.2
  382.7   ]
[1. 0. 0. 0. 0. 0. 0. 0.]
[1. 0. 0. 0. 0. 0. 0. 0.]
[2258.5649 2892.6365    0.     2247.277  1842.6035 1908.6399  620.1413
  382.5641]
[0.3139 0.2218 0.     0.     0.2906 0.0962 0.     0.0772]
[0.3139 0.2218 0.     0.     0.2906 0.0962 0.     0.0772]
 '''