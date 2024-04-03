import matplotlib.pyplot as plt
import numpy as np


#species = ("CC OPF", "LDT-CC OPF", "WCC-LDT OPF")

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
