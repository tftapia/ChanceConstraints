import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm

def EconomicDispatch():
    # Parameters
    N = 2
    d = np.array([120])

    c_p_n = np.array([30,30, 30]) # 25, 66
    c_alpha_n = np.array([50,60, 60]) # 40, 80
    c_beta_n = np.array([60,60,70]) # 80, 100 
    p_n_max = np.array([120,100,100])

    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)

    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)

    w_bar = 20
    w_sigma = 4

    # Model 
    model = gb.Model()

    # Variables
    p_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    alpha_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="alpha_n", lb=0) # Operational reserve variable

    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar == d, name='cons_demand') # Energy balance constraint
    cons_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_reserve')   # Operational reserve
    model.addConstrs(p_n[n] - p_n_max[n] + alpha_n[n]*inv_phi_eps*w_sigma <= 0 for n in range(N)) # Max generation limit constraint

    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] for n in range(N)),GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()

    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    print("{}: {}".format(p_n.varName, p_n.X))
    print("A solution alpha_n is")
    print("{}: {}".format(alpha_n.varName, alpha_n.X))
    print("The demand cons. dual values are")
    print("{}: {}".format(cons_demand.constrName, cons_demand.Pi))
    print("The reserve cons. dual values are")
    print("{}: {}".format(cons_reserve.constrName, cons_reserve.Pi)) 

def EconomicDispatch_LDT_old():
    # Parameters
    N = 2
    d = np.array([70])

    c_p_n = np.array([25,50, 30]) # 25, 66
    c_alpha_n = np.array([40,30, 35]) # 40, 80
    c_beta_n = np.array([40,100,110]) # 80, 100 
    p_n_max = np.array([100,100,100])

    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)

    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)

    w_bar = 10
    w_sigma = 4

    # Model 
    model = gb.Model()

    # Variables
    p_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    alpha_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="alpha_n", lb=0) # Operational reserve variable
    beta_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="beta_n", lb=0)   # Adversarial reserve variable
    
    omega_star = model.addMVar(1, vtype=GRB.CONTINUOUS, name="omega_star", lb=-GRB.INFINITY)     # Optimal critical value from the extreme event region
    omega_star_p = model.addMVar(1, vtype=GRB.CONTINUOUS, name="omega_star_p", lb=0) # Auxiliar value that represent the positive value of "omega_star" used in a SOC
    omega_star_m = model.addMVar(1, vtype=GRB.CONTINUOUS, name="omega_star_m", lb=0) # Auxiliar value that represent the negative value of "omega_star" used in a SOC

    lambda_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="lambda_n", lb=-GRB.INFINITY)     # Dual variable of the constrainst to obtain "omega_star"
    lambda_n_p = model.addMVar(N, vtype=GRB.CONTINUOUS, name="lambda_n_p", lb=0) # Auxiliar value that represent the positive value of "lambda_n" used in a SOC
    lambda_n_m = model.addMVar(N, vtype=GRB.CONTINUOUS, name="lambda_n_m", lb=0) # Auxiliar value that represent the negative value of "lambda_n" used in a SOC

    aux_omega_ab_p = model.addMVar(N, vtype=GRB.CONTINUOUS, lb=0, name= "aux_omega_ab_p")   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_omega_ab_m = model.addMVar(N, vtype=GRB.CONTINUOUS, lb=0, name= "aux_omega_ab_m")   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_lambda_ab_p = model.addMVar(N, vtype=GRB.CONTINUOUS, lb=0, name= "aux_lambda_ab_p") # Auxilar value to represent the positive value of "lambda_n*(alpha_n +beta_n)" in the SOC
    aux_lambda_ab_m = model.addMVar(N, vtype=GRB.CONTINUOUS, lb=0, name= "aux_lambda_ab_m") # Auxilar value to represent the negative value of "lambda_n*(alpha_n +beta_n)" in the SOC

    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar >= d, name='cons_demand')     # Energy balance constraint
    cons_op_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_op_reserve') # Operational reserve constraint
    cons_ad_reserve = model.addConstr(sum(beta_n[n] for n in range(N)) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint

    #model.addConstrs(p_n[n] - p_n_max[n] - alpha_n[n]*inv_phi_eps*w_sigma <= 0 for n in range(N))     # Max generation limit constraint
    
    model.addConstrs(lambda_n[n]== lambda_n_p[n] - lambda_n_m[n] for n in range(N))
    model.addConstr(omega_star == omega_star_p - omega_star_m)
    
    model.addConstr(w_sigma**(-0.5)*omega_star + inv_phi_ext <= 0)
    model.addConstrs(w_sigma**(-1)*omega_star - aux_lambda_ab_p[n]*aux_lambda_ab_p[n] + aux_lambda_ab_m[n]*aux_lambda_ab_m[n] == 0 for n in range(N))
    model.addConstrs(-p_n_max[n] + p_n[n] - aux_omega_ab_p[n]*aux_omega_ab_p[n] + aux_omega_ab_m[n]*aux_omega_ab_m[n]  == 0 for n in range(N))

    model.addConstrs((2*aux_omega_ab_p)*(2*aux_omega_ab_p) + (alpha_n[n]+beta_n[n] - omega_star_p)*(alpha_n[n]+beta_n[n] - omega_star_p) <= (alpha_n[n]+beta_n[n] + omega_star_p)*(alpha_n[n]+beta_n[n] + omega_star_p) for n in range(N))
    model.addConstrs((2*aux_omega_ab_m)*(2*aux_omega_ab_m) + (alpha_n[n]+beta_n[n] - omega_star_m)*(alpha_n[n]+beta_n[n] - omega_star_m) <= (alpha_n[n]+beta_n[n] + omega_star_m)*(alpha_n[n]+beta_n[n] + omega_star_m) for n in range(N))
    
    model.addConstrs((2*aux_lambda_ab_p[n])*(2*aux_lambda_ab_p[n]) + (alpha_n[n]+beta_n[n] - lambda_n_p[n])*(alpha_n[n]+beta_n[n] - lambda_n_p[n]) <= (alpha_n[n]+beta_n[n] + lambda_n_p[n])*(alpha_n[n]+beta_n[n] + lambda_n_p[n]) for n in range(N))
    model.addConstrs((2*aux_lambda_ab_m[n])*(2*aux_lambda_ab_m[n]) + (alpha_n[n]+beta_n[n] - lambda_n_m[n])*(alpha_n[n]+beta_n[n] - lambda_n_m[n]) <= (alpha_n[n]+beta_n[n] + lambda_n_m[n])*(alpha_n[n]+beta_n[n] + lambda_n_m[n]) for n in range(N))


    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] + c_beta_n[n]*beta_n[n] for n in range(N)),GRB.MINIMIZE)
    ## Allow QCP dual 
    #model.Params.QCPDual = 1
    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    ## Primal solution
    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    print("{}: {}".format(p_n.varName, p_n.X))
    print("A solution alpha_n is")
    print("{}: {}".format(alpha_n.varName, alpha_n.X))
    print("A solution beta_n is")
    print("{}: {}".format(beta_n.varName, beta_n.X ))
    print("A solution omega_star is")
    print("{}: {}".format(omega_star.varName, omega_star.X))
    print("A solution lambda_n is")
    print("{}: {}".format(lambda_n.varName, lambda_n.X ))
    
    print("A solution omega_star_p is")
    print("{}: {}".format(omega_star_p.varName, omega_star_p.X ))
    print("A solution omega_star_m is")
    print("{}: {}".format(omega_star_m.varName, omega_star_m.X ))

    print("A solution aux_omega_ab_p is")
    print("{}: {}".format(aux_omega_ab_p.varName, aux_omega_ab_p.X ))
    print("A solution aux_omega_ab_m is")
    print("{}: {}".format(aux_omega_ab_m.varName, aux_omega_ab_m.X ))
    
    ## Dual solution
    print("The demand cons. dual values are")
    print("{}: {}".format(cons_demand.constrName, cons_demand.Pi))
    print("The operational reserve cons. dual values are")
    print("{}: {}".format(cons_op_reserve.constrName, cons_op_reserve.Pi))
    print("The adversarialreserve cons. dual values are")
    print("{}: {}".format(cons_ad_reserve.constrName, cons_ad_reserve.Pi)) 
    #for v in p_n.values():
    #    print("{}: {}".format(v.varName, v.X))
    #for c in cons_reserve.values():
    #    print("{}: {}".format(c.constrName, c.Pi))  # .QCPi is used for quadratic constraints

def EconomicDispatch_LDT():
    # Parameters
    N = 2
    d = np.array([120])

    c_p_n = np.array([30,30, 30]) # 25, 66
    c_alpha_n = np.array([50,60, 60]) # 40, 80
    c_beta_n = np.array([60,60,70]) # 80, 100 
    p_n_max = np.array([120,100,100])

    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)

    epsilon_ext = 0.01
    inv_phi_ext = norm.ppf(epsilon_ext)

    w_bar = 20
    w_sigma = 4

    # Model 
    model = gb.Model()

     # Variables
    p_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    alpha_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="alpha_n", lb=0) # Operational reserve variable
    beta_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="beta_n", lb=0)   # Adversarial reserve variable
    
    omega_star = model.addMVar(1, vtype=GRB.CONTINUOUS, name="omega_star", lb=-GRB.INFINITY) # Optimal critical value from the extreme event region
    lambda_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="lambda_n", lb=-GRB.INFINITY)     # Dual variable of the constrainst to obtain "omega_star"

    aux_omega_a = model.addMVar(N, vtype=GRB.CONTINUOUS, name= "aux_omega_a", lb=-GRB.INFINITY)   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_omega_b = model.addMVar(N, vtype=GRB.CONTINUOUS, name= "aux_omega_b", lb=-GRB.INFINITY)   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_lambda_a = model.addMVar(N, vtype=GRB.CONTINUOUS, name= "aux_lambda_a", lb=-GRB.INFINITY) # Auxilar value to represent the positive value of "lambda_n*(alpha_n +beta_n)" in the SOC
    aux_lambda_b = model.addMVar(N, vtype=GRB.CONTINUOUS, name= "aux_lambda_b", lb=-GRB.INFINITY) # Auxilar value to represent the negative value of "lambda_n*(alpha_n +beta_n)" in the SOC

    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar == d, name='cons_demand')     # Energy balance constraint
    cons_op_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_op_reserve') # Operational reserve constraint
    cons_ad_reserve = model.addConstr(sum(beta_n[n] for n in range(N)) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint

    model.addConstrs(p_n[n] - p_n_max[n] - alpha_n[n]*inv_phi_eps*w_sigma <= 0 for n in range(N))     # Max generation limit constraint
    
    model.addConstrs(aux_omega_a[n] == omega_star*alpha_n[n] for n in range(N))
    model.addConstrs(aux_omega_b[n] == omega_star*beta_n[n] for n in range(N))
    model.addConstrs(aux_lambda_a[n] == lambda_n[n]*alpha_n[n] for n in range(N))
    model.addConstrs(aux_lambda_b[n] == lambda_n[n]*beta_n[n] for n in range(N))

    ## LDT Constraint
    model.addConstr(w_sigma**(-0.5)*omega_star + inv_phi_ext <= 0)
    model.addConstrs(w_sigma**(-1)*omega_star + aux_lambda_a[n] + aux_lambda_b[n] == 0 for n in range(N))
    model.addConstrs(-p_n_max[n] + p_n[n] - aux_omega_a[n] - aux_omega_b[n] == 0 for n in range(N))

    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] + c_beta_n[n]*beta_n[n] for n in range(N)),GRB.MINIMIZE)
    ## Allow QCP dual 
    #model.Params.QCPDual = 1
    model.Params.NonConvex = 2
    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    ## Primal solution
    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    print("{}: {}".format(p_n.varName, p_n.X))
    print("A solution alpha_n is")
    print("{}: {}".format(alpha_n.varName, alpha_n.X))
    print("A solution beta_n is")
    print("{}: {}".format(beta_n.varName, beta_n.X ))
    print("A solution omega_star is")
    print("{}: {}".format(omega_star.varName, omega_star.X))
    print("A solution lambda_n is")
    print("{}: {}".format(lambda_n.varName, lambda_n.X ))
    

    print("A solution aux_omega_a is")
    print("{}: {}".format(aux_omega_a.varName, aux_omega_a.X ))
    print("A solution aux_omega_b is")
    print("{}: {}".format(aux_omega_b.varName, aux_omega_b.X ))
    
    ## Dual solution
    #print("The demand cons. dual values are")
    #print("{}: {}".format(cons_demand.constrName, cons_demand.Pi))
    #print("The operational reserve cons. dual values are")
    #print("{}: {}".format(cons_op_reserve.constrName, cons_op_reserve.Pi))
    #print("The adversarialreserve cons. dual values are")
    #print("{}: {}".format(cons_ad_reserve.constrName, cons_ad_reserve.Pi)) 
    #for v in p_n.values():
    #    print("{}: {}".format(v.varName, v.X))
    #for c in cons_reserve.values():
    #    print("{}: {}".format(c.constrName, c.Pi))  # .QCPi is used for quadratic constraints

def EconomicDispatch_LDT_2(omega_star, lambda_n):
    # Parameters
    N = 2
    d = np.array([70])

    c_p_n = np.array([25,50, 30]) # 25, 66
    c_alpha_n = np.array([40,30, 35]) # 40, 80
    c_beta_n = np.array([50,100,110]) # 80, 100 
    p_n_max = np.array([40,100,100])

    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)

    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)

    w_bar = 10
    w_sigma = 4

    # Model 
    model = gb.Model()

     # Variables
    p_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    alpha_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="alpha_n", lb=0) # Operational reserve variable
    beta_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="beta_n", lb=0)   # Adversarial reserve variable
 
    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar == d, name='cons_demand')     # Energy balance constraint
    cons_op_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_op_reserve') # Operational reserve constraint
    cons_ad_reserve = model.addConstr(sum(beta_n[n] for n in range(N)) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint

    model.addConstrs(p_n[n] - p_n_max[n] - alpha_n[n]*inv_phi_eps*w_sigma <= 0 for n in range(N))     # Max generation limit constraint
    
    ## LDT Constraint
    model.addConstrs(w_sigma**(-1)*omega_star + lambda_n[n]*alpha_n[n] + lambda_n[n]*beta_n[n] == 0 for n in range(N))
    model.addConstrs(-p_n_max[n] + p_n[n] - omega_star*alpha_n[n] - omega_star*beta_n[n]== 0 for n in range(N))

    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] + c_beta_n[n]*beta_n[n] for n in range(N)),GRB.MINIMIZE)
    
    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    ## Primal solution
    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    print("{}: {}".format(p_n.varName, p_n.X))
    print("A solution alpha_n is")
    print("{}: {}".format(alpha_n.varName, alpha_n.X))
    print("A solution beta_n is")
    print("{}: {}".format(beta_n.varName, beta_n.X ))
    
    ## Dual solution
    print("The demand cons. dual values are")
    print("{}: {}".format(cons_demand.constrName, cons_demand.Pi))
    print("The operational reserve cons. dual values are")
    print("{}: {}".format(cons_op_reserve.constrName, cons_op_reserve.Pi))
    print("The adversarialreserve cons. dual values are")
    print("{}: {}".format(cons_ad_reserve.constrName, cons_ad_reserve.Pi)) 
    #for v in p_n.values():
    #    print("{}: {}".format(v.varName, v.X))
    #for c in cons_reserve.values():
    #    print("{}: {}".format(c.constrName, c.Pi))  # .QCPi is used for quadratic constraints

def EconomicDispatch_LDT_less():
    # Parameters
    N = 2
    d = np.array([70])

    c_p_n = np.array([25,50, 30]) # 25, 66
    c_alpha_n = np.array([40,30, 35]) # 40, 80
    c_beta_n = np.array([40,100,110]) # 80, 100 
    p_n_max = np.array([100,100,100])

    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)

    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)

    w_bar = 10
    w_sigma = 4

    # Model 
    model = gb.Model()

     # Variables
    p_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    alpha_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="alpha_n", lb=0) # Operational reserve variable
    beta_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="beta_n", lb=0)   # Adversarial reserve variable
    
    omega_star = model.addMVar(1, vtype=GRB.CONTINUOUS, name="omega_star", lb=-GRB.INFINITY) # Optimal critical value from the extreme event region
    lambda_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="lambda_n", lb=-GRB.INFINITY)     # Dual variable of the constrainst to obtain "omega_star"

    aux_omega_a = model.addMVar(N, vtype=GRB.CONTINUOUS, name= "aux_omega_a", lb=-GRB.INFINITY)   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_omega_b = model.addMVar(N, vtype=GRB.CONTINUOUS, name= "aux_omega_b", lb=-GRB.INFINITY)   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_lambda_a = model.addMVar(N, vtype=GRB.CONTINUOUS, name= "aux_lambda_a", lb=-GRB.INFINITY) # Auxilar value to represent the positive value of "lambda_n*(alpha_n +beta_n)" in the SOC
    aux_lambda_b = model.addMVar(N, vtype=GRB.CONTINUOUS, name= "aux_lambda_b", lb=-GRB.INFINITY) # Auxilar value to represent the negative value of "lambda_n*(alpha_n +beta_n)" in the SOC

    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar == d, name='cons_demand')     # Energy balance constraint
    cons_op_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_op_reserve') # Operational reserve constraint
    cons_ad_reserve = model.addConstr(sum(beta_n[n] for n in range(N)) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint

    model.addConstrs(aux_omega_a[n] == omega_star*alpha_n[n] for n in range(N))
    model.addConstrs(aux_omega_b[n] == omega_star*beta_n[n] for n in range(N))
    model.addConstrs(aux_lambda_a[n] == lambda_n[n]*alpha_n[n] for n in range(N))
    model.addConstrs(aux_lambda_b[n] == lambda_n[n]*beta_n[n] for n in range(N))

    ## LDT Constraint
    model.addConstr(w_sigma**(-0.5)*omega_star + inv_phi_ext <= 0)
    model.addConstrs(w_sigma**(-1)*omega_star - aux_lambda_a[n] - aux_lambda_b[n] == 0 for n in range(N))
    model.addConstrs(-p_n_max[n] + p_n[n] - aux_omega_a[n] - aux_omega_b[n]== 0 for n in range(N))

    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] + c_beta_n[n]*beta_n[n] for n in range(N)),GRB.MINIMIZE)
    ## Allow QCP dual 
    #model.Params.QCPDual = 1
    model.Params.NonConvex = 2
    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    ## Primal solution
    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    print("{}: {}".format(p_n.varName, p_n.X))
    print("A solution alpha_n is")
    print("{}: {}".format(alpha_n.varName, alpha_n.X))
    print("A solution beta_n is")
    print("{}: {}".format(beta_n.varName, beta_n.X ))
    print("A solution omega_star is")
    print("{}: {}".format(omega_star.varName, omega_star.X))
    print("A solution lambda_n is")
    print("{}: {}".format(lambda_n.varName, lambda_n.X ))
    

    print("A solution aux_omega_a is")
    print("{}: {}".format(aux_omega_a.varName, aux_omega_a.X ))
    print("A solution aux_omega_b is")
    print("{}: {}".format(aux_omega_b.varName, aux_omega_b.X ))
    
    ## Dual solution
    #print("The demand cons. dual values are")
    #print("{}: {}".format(cons_demand.constrName, cons_demand.Pi))
    #print("The operational reserve cons. dual values are")
    #print("{}: {}".format(cons_op_reserve.constrName, cons_op_reserve.Pi))
    #print("The adversarialreserve cons. dual values are")
    #print("{}: {}".format(cons_ad_reserve.constrName, cons_ad_reserve.Pi)) 
    #for v in p_n.values():
    #    print("{}: {}".format(v.varName, v.X))
    #for c in cons_reserve.values():
    #    print("{}: {}".format(c.constrName, c.Pi))  # .QCPi is used for quadratic constraints


EconomicDispatch()

EconomicDispatch_LDT()

#EconomicDispatch_LDT_2(-70, [-30.625, -12.25])
#EconomicDispatch_LDT_2(-40, [8.70313728e+04, 5.00028727e+00])

#EconomicDispatch_LDT_2(-20, [-2.5])

#EconomicDispatch_LDT_less()
