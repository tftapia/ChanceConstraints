import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm

def cutting_planes_problem_ED():
    flag_print = False
    # Parameters
    N = 2
    T = 2
    d = np.array([1000,600])
    h_t = np.array([1200,7560])

    c_p_n = np.array([30,50]) # 25, 66
    c_X_n = np.array([190000,90000]) # 50


    # Define model
    model = gb.Model()

    # Variables
    p_n = model.addVars(N,T, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    X_n = model.addVars(N, vtype=GRB.CONTINUOUS, name="X_n", lb=0) # Operational reserve variable

    # Constraints
    cons_demand = model.addConstrs((sum(p_n[n,t] for n in range(N)) == d[t] for t in range(T)) , name='cons_demand') # Energy balance constraint
    cons_reserve = model.addConstrs((p_n[n,t] <= X_n[n] for n in range(N) for t in range(T)), name='cons_reserve')   # Operational reserve

    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n,t]*h_t[t] for n in range(N) for t in range(T))  + sum(c_X_n[n]*X_n[n] for n in range(N)),GRB.MINIMIZE)

    # Solve
    model.optimize()
    ## Primal solution
    # Print result
    obj_value = model.getObjective()
    print("\n Solving dispatch_problem")
    print("The objective value is", obj_value.getValue())
    # Primal results
    print("The p_n values are")
    for v in p_n.values():
        print("{}: {}".format(v.varName, v.X))
    for v in X_n.values():
        print("{}: {}".format(v.varName, v.X))

cutting_planes_problem_ED()