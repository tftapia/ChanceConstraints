import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm
import pandas as pd
import random


from main_data import init_dictionaries


data = init_dictionaries(True)
print(data["set_gen_index"])
print(data["set_node_index"])
print(data["set_lines_node_index"])