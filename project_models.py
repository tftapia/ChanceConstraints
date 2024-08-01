import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm
import pandas as pd
import random

from project_aux import *

# Models
def ED_standard_CC_network(system_data, system_param, flag_developer_mode = False, digit_round = 4):
    
    #Auxilar parameters
    sys_inv_phi_eps = norm.ppf(1- system_param["sys_epsilon"])
    sys_w_mean = sum(system_data["wind_mean"].values())
    sys_w_sigma = sum(system_data["wind_std"].values())

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(system_data["set_lines_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="alpha_n", lb=system_param["sys_math_error"]) 

    # Set objective function
    obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*sys_w_sigma**2*system_data["gen_c2"][n] for n in system_data["set_gen_index"]) 
                                   + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in system_data["node_gens"][i]) 
                                     + sum(sys_w_mean*system_data["wind_factor"][n] for n in system_data["node_winds"][i]) 
                                     - sum(f_ij[i,j] for (i,j) in system_data["node_lines_in"][i])
                                     + sum(f_ij[i,j] for (i,j) in system_data["node_lines_out"][i])
                                     + s_i[i] == system_data["node_demand"][i] for i in system_data["set_node_index"]), name='const_demand')
    #const_pmax = model.addConstrs((p_n[n] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax')
    const_ft = model.addConstrs((system_data["line_susceptance_matrix"][i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_ft')
    const_fmax = model.addConstrs((f_ij[i,j] <= system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmin')
    const_t0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in system_data["set_gen_index"]) == 1, name='const_op_reserve') # Operational reserve constraint
  
    # Add cutting planes until the optimal solution satisfy the WCC constraint
    for _ in range(system_param["sys_cp_iterations"]):
        model.optimize()
        obj_value = model.getObjective()
        
        if flag_developer_mode:
            print('ITERATION, %s' % _)
            print("The optimal value is", round(obj_value.getValue(),digit_round))
            print("A solution p_n is")
            for v in p_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))
            print("A solution alpha_n is")
            for v in alpha_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))

        # Save variables
        p_n_star = dict()
        alpha_n_star = dict()

        for n in system_data["set_gen_index"]:
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X

        # Check if solution satisfy the WCC constraint
        if all(p_n_star[n] + alpha_n_star[n]*sys_inv_phi_eps*sys_w_sigma <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]):
            break

        # Add a cutting plane to the model 
        for n in system_data["set_gen_index"]:
            if p_n_star[n] + alpha_n_star[n]*sys_inv_phi_eps*sys_w_sigma >= system_data["gen_pmax"][n]:
                if flag_developer_mode:
                    print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(p_n[n] + alpha_n[n]*sys_inv_phi_eps*sys_w_sigma <= system_data["gen_pmax"][n])
        
        if flag_developer_mode:        
            print (model.display())
    
    # Print results
    obj = model.getObjective()
    print("\n[!] Standard-CC problem solved in {} iterations".format(_+1))
    print("The optimal value is", round(obj.getValue(),digit_round))
    if flag_developer_mode:
        ## Primal solution
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution s_i is")
        for v in s_i.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
            
        # Dual results
        for c in const_demand.values():
            print(c.constrName, round(c.Pi, digit_round))
        print(const_op_reserve.constrName, round(const_op_reserve.Pi, digit_round))
 
    # Save solution    
    solution_dict = dict()
    solution_dict["o_opt"] = round(obj.getValue(),digit_round)
    solution_dict["p_opt"] = []
    solution_dict["a_opt"] = []
    solution_dict["p_price"] = []
    solution_dict["a_price"] = round(const_op_reserve.Pi,digit_round)
    
    for v in p_n.values():
        solution_dict["p_opt"].append(round(v.X,digit_round))
    for v in alpha_n.values():
        solution_dict["a_opt"].append(round(v.X,digit_round))
    for c in const_demand.values():
        solution_dict["p_price"].append(round(c.Pi,digit_round))  

    return solution_dict

def ED_LDT_CC_network(system_data, system_param, flag_developer_mode = False, digit_round = 4):
    #Auxilar parameters
    sys_inv_phi_eps = norm.ppf(1- system_param["sys_epsilon"])
    sys_inv_phi_eps_ext = norm.ppf(1-system_param["sys_epsilon_ext"])
    sys_w_mean = sum(system_data["wind_mean"].values())
    sys_w_sigma = sum(system_data["wind_std"].values())

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(system_data["set_lines_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="alpha_n", lb=system_param["sys_math_error"])  # Operational reserve variable
    beta_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="beta_n", lb=system_param["sys_math_error"])   # Adversarial reserve variable

    omega_star = model.addVar(vtype=GRB.CONTINUOUS, name="omega_star", lb=-GRB.INFINITY) # Optimal critical value from the extreme event region
    lambda_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="lambda_n", lb=-GRB.INFINITY)     # Dual variable of the constrainst to obtain "omega_star"

    aux_omega_a = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_omega_a", lb=-GRB.INFINITY)   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_omega_b = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_omega_b", lb=-GRB.INFINITY)   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_lambda_a = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_lambda_a", lb=-GRB.INFINITY) # Auxilar value to represent the positive value of "lambda_n*(alpha_n +beta_n)" in the SOC
    aux_lambda_b = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_lambda_b", lb=-GRB.INFINITY) # Auxilar value to represent the negative value of "lambda_n*(alpha_n +beta_n)" in the SOC
    
    # Set objective function
    #obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
    #                               + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
                                   + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE) 
    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in system_data["node_gens"][i]) 
                                     + sum(sys_w_mean*system_data["wind_factor"][n] for n in system_data["node_winds"][i]) 
                                     - sum(f_ij[i,j] for (i,j) in system_data["node_lines_in"][i])
                                     + sum(f_ij[i,j] for (i,j) in system_data["node_lines_out"][i])
                                     + s_i[i] == system_data["node_demand"][i] for i in system_data["set_node_index"]), name='const_demand')
    const_pmax = model.addConstrs((p_n[n] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax')
    const_ft = model.addConstrs((system_data["line_susceptance_matrix"][i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_ft')
    const_fmax = model.addConstrs((f_ij[i,j] <= system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmin')
    const_t0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in system_data["set_gen_index"]) == 1, name='const_op_reserve') # Operational reserve constraint
    const_ad_reserve = model.addConstr(sum(beta_n[n] for n in system_data["set_gen_index"]) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint
    const_pmax_CC = model.addConstrs(p_n[n] - system_data["gen_pmax"][n] + alpha_n[n]*sys_inv_phi_eps*sys_w_sigma <= 0 for n in system_data["set_gen_index"]) # Max generation limit constraint
    '''
    model.addConstrs(aux_omega_b[n] == (omega_star-sys_inv_phi_eps*sys_w_sigma*alpha_n[n])*beta_n[n] for n in system_data["set_gen_index"])
    model.addConstrs(aux_lambda_b[n] == lambda_n[n]*beta_n[n] for n in system_data["set_gen_index"])
    model.addConstr(-sys_w_sigma**(-1.0)*omega_star - sys_inv_phi_eps_ext <= 0)
    model.addConstrs(sys_w_sigma**(-2.0)*omega_star - aux_lambda_b[n] == 0 for n in system_data["set_gen_index"])
    model.addConstrs(-system_data["gen_pmax"][n] + p_n[n] + sys_inv_phi_eps*sys_w_sigma*alpha_n[n] + aux_omega_b[n] == 0 for n in system_data["set_gen_index"])
    '''
    model.addConstrs(aux_omega_a[n] == omega_star*alpha_n[n] for n in system_data["set_gen_index"])
    model.addConstrs(aux_omega_b[n] == omega_star*beta_n[n] for n in system_data["set_gen_index"])

    model.addConstrs(aux_lambda_a[n] == lambda_n[n]*alpha_n[n] for n in system_data["set_gen_index"])
    model.addConstrs(aux_lambda_b[n] == lambda_n[n]*beta_n[n] for n in system_data["set_gen_index"])
    ## LDT Constraint
    model.addConstrs(-system_data["gen_pmax"][n] + p_n[n] + aux_omega_a[n] + aux_omega_b[n] == 0 for n in system_data["set_gen_index"])
    model.addConstr(-sys_w_sigma**(-1.0)*omega_star - sys_inv_phi_eps_ext <= 0)
    model.addConstrs(sys_w_sigma**(-2.0)*omega_star - aux_lambda_a[n] - aux_lambda_b[n] == 0 for n in system_data["set_gen_index"])
    
    if flag_developer_mode:        
        print (model.display())

    ## Allow QCP dual 
    model.Params.QCPDual = 1
    model.Params.NonConvex = 2
    model.setParam('MIPGap', system_param["sys_MIPGap"])
    model.setParam('Timelimit', system_param["sys_Timelimit"])

    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    print("\n[!] LDT-CC problem solved")
    print("The optimal value is", round(obj.getValue(),digit_round))
    if flag_developer_mode:
        ## Primal solution
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution beta_n is")
        for v in beta_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution s_i is")
        for v in s_i.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution omega_star is")
        print("{}: {}".format(omega_star.varName, omega_star.X))
        print("A solution lambda_n is")
        for v in lambda_n.values():
            print("{}: {}".format(v.varName, v.X))
                
    # Save solution    
    solution_dict = dict()
    solution_dict["o_opt"] = round(obj.getValue(),digit_round)
    solution_dict["p_opt"] = []
    solution_dict["a_opt"] = []
    solution_dict["b_opt"] = []
    solution_dict["w_opt"] = omega_star.X
    solution_dict["l_opt"] = []

    for v in p_n.values():
        solution_dict["p_opt"].append(round(v.X,digit_round))
    for v in alpha_n.values():
        solution_dict["a_opt"].append(round(v.X,digit_round))
    for v in beta_n.values():
        solution_dict["b_opt"].append(round(v.X,digit_round))
    for v in lambda_n.values():
        solution_dict["l_opt"].append(v.X)

    return solution_dict

def ED_WCC_network(system_data, system_param, flag_developer_mode = False, digit_round = 4):
    
    #Auxilar parameters
    sys_inv_phi_eps = norm.ppf(1- system_param["sys_epsilon"])
    sys_w_mean = sum(system_data["wind_mean"].values())
    sys_w_sigma = sum(system_data["wind_std"].values())

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(system_data["set_lines_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="alpha_n", lb=system_param["sys_math_error"]) 

    # Set objective function
    obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] for n in system_data["set_gen_index"]) 
                                   + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in system_data["node_gens"][i]) 
                                     + sum(sys_w_mean*system_data["wind_factor"][n] for n in system_data["node_winds"][i]) 
                                     - sum(f_ij[i,j] for (i,j) in system_data["node_lines_in"][i])
                                     + sum(f_ij[i,j] for (i,j) in system_data["node_lines_out"][i])
                                     + s_i[i] == system_data["node_demand"][i] for i in system_data["set_node_index"]), name='const_demand')
    #const_pmax = model.addConstrs((p_n[n] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax')
    const_ft = model.addConstrs((system_data["line_susceptance_matrix"][i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_ft')
    const_fmax = model.addConstrs((f_ij[i,j] <= system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmin')
    const_t0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in system_data["set_gen_index"]) == 1, name='const_op_reserve') # Operational reserve constraint
  
    # Add cutting planes until the optimal solution satisfy the WCC constraint
    for _ in range(system_param["sys_cp_iterations"]):
        model.optimize()
        obj_value = model.getObjective()
        
        if flag_developer_mode:
            print('ITERATION, %s' % _)
            print("The optimal value is", round(obj_value.getValue(),digit_round))
            print("A solution p_n is")
            for v in p_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))
            print("A solution alpha_n is")
            for v in alpha_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))

        # Save variables
        p_n_star = dict()
        alpha_n_star = dict()

        for n in system_data["set_gen_index"]:
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X

        # Check if solution satisfy the WCC constraint
        if all(truncated_normal_funtion(p_n_star[n]-system_data["gen_pmax"][n], alpha_n_star[n]*sys_w_sigma) <= system_param["sys_epsilon"] for n in system_data["set_gen_index"]):
            break

        # Add a cutting plane to the model 
        for n in system_data["set_gen_index"]:
            mean_star = p_n_star[n]-system_data["gen_pmax"][n]
            std_star = alpha_n_star[n]*sys_w_sigma
            if truncated_normal_funtion(mean_star, std_star) > system_param["sys_epsilon"]:
                if flag_developer_mode:
                    print('ERROR in iterration, %s, with generador, %s' %(_,n))
                    print("tnf {}, epsilon {}".format(round(truncated_normal_funtion(mean_star, std_star), digit_round),system_param["sys_epsilon"]))
                model.addConstr(truncated_normal_funtion(mean_star, std_star) 
                                + ((p_n[n]-system_data["gen_pmax"][n]) - mean_star)*truncated_normal_funtion_dmu(mean_star, std_star) 
                                + ((alpha_n[n]*sys_w_sigma) - std_star)*truncated_normal_funtion_dsigma(mean_star, std_star) 
                                <= system_param["sys_epsilon"])
        
        if flag_developer_mode:        
            print (model.display())
    # Print results
    obj = model.getObjective()
    print("\n[!] WCC problem solved in {} iterations".format(_+1))
    print("The optimal value is", round(obj.getValue(),digit_round))
    if flag_developer_mode:
        ## Primal solution
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution s_i is")
        for v in s_i.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
            
        # Dual results
        for c in const_demand.values():
            print(c.constrName, round(c.Pi, digit_round))
        print(const_op_reserve.constrName, round(const_op_reserve.Pi, digit_round))
 
    # Save solution    
    solution_dict = dict()
    solution_dict["o_opt"] = round(obj.getValue(),digit_round)
    solution_dict["p_opt"] = []
    solution_dict["a_opt"] = []
    solution_dict["p_price"] = []
    solution_dict["a_price"] = round(const_op_reserve.Pi,digit_round)

    for v in p_n.values():
        solution_dict["p_opt"].append(round(v.X,digit_round))
    for v in alpha_n.values():
        solution_dict["a_opt"].append(round(v.X,digit_round))
    for c in const_demand.values():
        solution_dict["p_price"].append(round(c.Pi,digit_round))  

    return solution_dict

def ED_LDT_WCC_network(system_data, system_param, sys_w_star, flag_developer_mode = False, digit_round = 4):
    #Auxilar parameters
    sys_inv_phi_eps = norm.ppf(1- system_param["sys_epsilon"])
    sys_inv_phi_eps_ext = norm.ppf(system_param["sys_epsilon_ext"])
    sys_w_mean = sum(system_data["wind_mean"].values())
    sys_w_sigma = sum(system_data["wind_std"].values())

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(system_data["set_lines_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="alpha_n", lb=system_param["sys_math_error"])  # Operational reserve variable
    beta_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="beta_n", lb=system_param["sys_math_error"])   # Adversarial reserve variable

    # Set objective function
    #obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
    #                               + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
                                   + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in system_data["node_gens"][i]) 
                                     + sum(sys_w_mean*system_data["wind_factor"][n] for n in system_data["node_winds"][i]) 
                                     - sum(f_ij[i,j] for (i,j) in system_data["node_lines_in"][i])
                                     + sum(f_ij[i,j] for (i,j) in system_data["node_lines_out"][i])
                                     + s_i[i] == system_data["node_demand"][i] for i in system_data["set_node_index"]), name='const_demand')
    const_pmax = model.addConstrs((p_n[n] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax')
    const_ft = model.addConstrs((system_data["line_susceptance_matrix"][i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_ft')
    const_fmax = model.addConstrs((f_ij[i,j] <= system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmin')
    const_t0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in system_data["set_gen_index"]) == 1, name='const_op_reserve') # Operational reserve constraint
    const_ad_reserve = model.addConstr(sum(beta_n[n] for n in system_data["set_gen_index"]) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint
    #const_pmax_CC = model.addConstrs(p_n[n] - system_data["gen_pmax"][n] + alpha_n[n]*sys_inv_phi_eps*sys_w_sigma <= 0 for n in system_data["set_gen_index"]) # Max generation limit constraint
    
    # Add cutting planes until the optimal solution satisfy the WCC constraint
    for _ in range(system_param["sys_cp_iterations"]):
        model.optimize()
        obj_value = model.getObjective()
        
        if flag_developer_mode:
            print('ITERATION, %s' % _)
            print("The optimal value is", round(obj_value.getValue(),digit_round))
            print("A solution p_n is")
            for v in p_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))
            print("A solution alpha_n is")
            for v in alpha_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))
            print("A solution beta_n is")
            for v in beta_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))    

        # Save variables
        p_n_star = dict()
        alpha_n_star = dict()
        beta_n_star = dict()

        z_w_star = 0.75*sys_w_star/sys_w_sigma
        for n in system_data["set_gen_index"]:
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X
            beta_n_star[n] = model.getVarByName('beta_n[%d]'%(n)).X

        # Check if solution satisfy the WCC constraint with a piece-wise policy 
        if all(
            truncated_normal_funtion(p_n_star[n]-system_data["gen_pmax"][n] - (alpha_n_star[n])*sys_w_sigma*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"])),
                                    (sys_w_sigma*(alpha_n_star[n]))*np.sqrt(1 - z_w_star*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"]))**2))
                +
            truncated_normal_funtion(p_n_star[n]-system_data["gen_pmax"][n] + (alpha_n_star[n]+beta_n_star[n])*sys_w_sigma*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"])),
                                    (sys_w_sigma*(alpha_n_star[n]+beta_n_star[n]))*np.sqrt(1 + z_w_star*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"]))**2))
            <= system_param["sys_epsilon"] for n in system_data["set_gen_index"]):
            break

        # Add a cutting plane to the model 
        for n in system_data["set_gen_index"]:
            mean_lower_star = p_n_star[n]-system_data["gen_pmax"][n] - (alpha_n_star[n])*sys_w_sigma*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"]))
            std_lower_star = (sys_w_sigma*(alpha_n_star[n]))*np.sqrt(1 - z_w_star*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"]))**2)
            mean_greater_star = p_n_star[n]-system_data["gen_pmax"][n] + (alpha_n_star[n]+beta_n_star[n])*sys_w_sigma*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"]))
            std_greater_star = (sys_w_sigma*(alpha_n_star[n]+beta_n_star[n]))*np.sqrt(1 + z_w_star*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"]))**2)
            if truncated_normal_funtion(mean_lower_star, std_lower_star) + truncated_normal_funtion(mean_greater_star, std_greater_star) > system_param["sys_epsilon"]:
                if flag_developer_mode:
                    print('ERROR in iterration, %s, with generador, %s' %(_,n))
                    print("tnf {}, epsilon {}".format(round(truncated_normal_funtion(mean_lower_star, std_lower_star) + truncated_normal_funtion(mean_greater_star, std_greater_star), digit_round),system_param["sys_epsilon"]))
                model.addConstr(
                    truncated_normal_funtion(mean_lower_star,std_lower_star)
                    + ((p_n[n]-system_data["gen_pmax"][n] - (alpha_n[n])*sys_w_sigma*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)))) - mean_lower_star)*truncated_normal_funtion_dmu(mean_lower_star, std_lower_star) 
                    + (((alpha_n[n])*sys_w_sigma)*np.sqrt(1 - z_w_star*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"]))**2) - std_lower_star)*truncated_normal_funtion_dsigma(mean_lower_star, std_lower_star) 
                    + truncated_normal_funtion(mean_greater_star,std_greater_star)
                    + ((p_n[n]-system_data["gen_pmax"][n] + (alpha_n[n]+beta_n[n])*sys_w_sigma*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star))) ) - mean_greater_star)*truncated_normal_funtion_dmu(mean_greater_star,std_greater_star)
                    + (((alpha_n[n]+beta_n[n])*sys_w_sigma)*np.sqrt(1 + z_w_star*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"]))**2) - std_greater_star)*truncated_normal_funtion_dsigma(mean_greater_star,std_greater_star)
                    <= system_param["sys_epsilon"])

        if flag_developer_mode:        
            print (model.display())
        
    ## Solve parameters
    model.setParam('MIPGap', system_param["sys_MIPGap"])
    model.setParam('Timelimit', system_param["sys_Timelimit"])

    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    print("\n[!] LDT-WCC problem solved in {} iterations".format(_+1))
    print("The optimal value is", round(obj.getValue(),digit_round))
    if flag_developer_mode:
        ## Primal solution
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution beta_n is")
        for v in beta_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution s_i is")
        for v in s_i.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))

        # Dual results
        for c in const_demand.values():
            print(c.constrName, round(c.Pi, digit_round))
        print(const_op_reserve.constrName, round(const_op_reserve.Pi, digit_round))
        print(const_ad_reserve.constrName, round(const_ad_reserve.Pi, digit_round))

    # Save solution    
    solution_dict = dict()
    solution_dict["o_opt"] = round(obj.getValue(),digit_round)
    solution_dict["p_opt"] = []
    solution_dict["a_opt"] = []
    solution_dict["b_opt"] = []
    solution_dict["p_price"] = []
    solution_dict["a_price"] = round(const_op_reserve.Pi,digit_round)
    solution_dict["b_price"] = round(const_ad_reserve.Pi,digit_round)

    for v in p_n.values():
        solution_dict["p_opt"].append(round(v.X,digit_round))
    for v in alpha_n.values():
        solution_dict["a_opt"].append(round(v.X,digit_round))
    for v in beta_n.values():
        solution_dict["b_opt"].append(round(v.X,digit_round))
    for c in const_demand.values():
        solution_dict["p_price"].append(round(c.Pi,digit_round))  

    return solution_dict

def ED_LDT_CC_price(system_data, system_param, sys_w_star, flag_developer_mode = False, digit_round = 4):
    #Auxilar parameters
    sys_inv_phi_eps = norm.ppf(1- system_param["sys_epsilon"])
    sys_inv_phi_eps_ext = norm.ppf(1-system_param["sys_epsilon_ext"])
    sys_w_mean = sum(system_data["wind_mean"].values())
    sys_w_sigma = sum(system_data["wind_std"].values())

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(system_data["set_lines_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="alpha_n", lb=system_param["sys_math_error"])  # Operational reserve variable
    beta_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="beta_n", lb=system_param["sys_math_error"])   # Adversarial reserve variable

    aux_omega_a = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_omega_a", lb=-GRB.INFINITY)   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_omega_b = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_omega_b", lb=-GRB.INFINITY)   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
 
    # Set objective function
    #obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
    #                               + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
                                   + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE) 
    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in system_data["node_gens"][i]) 
                                     + sum(sys_w_mean*system_data["wind_factor"][n] for n in system_data["node_winds"][i]) 
                                     - sum(f_ij[i,j] for (i,j) in system_data["node_lines_in"][i])
                                     + sum(f_ij[i,j] for (i,j) in system_data["node_lines_out"][i])
                                     + s_i[i] == system_data["node_demand"][i] for i in system_data["set_node_index"]), name='const_demand')
    const_pmax = model.addConstrs((p_n[n] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax')
    const_ft = model.addConstrs((system_data["line_susceptance_matrix"][i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_ft')
    const_fmax = model.addConstrs((f_ij[i,j] <= system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmin')
    const_t0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in system_data["set_gen_index"]) == 1, name='const_op_reserve') # Operational reserve constraint
    const_ad_reserve = model.addConstr(sum(beta_n[n] for n in system_data["set_gen_index"]) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint
    #const_pmax_CC = model.addConstrs(p_n[n] - system_data["gen_pmax"][n] + alpha_n[n]*sys_inv_phi_eps*sys_w_sigma <= 0 for n in system_data["set_gen_index"]) # Max generation limit constraint

    model.addConstrs(aux_omega_a[n] == sys_w_star*alpha_n[n] for n in system_data["set_gen_index"])
    model.addConstrs(aux_omega_b[n] == sys_w_star*beta_n[n] for n in system_data["set_gen_index"])
    ## LDT Constraint
    model.addConstrs(-system_data["gen_pmax"][n] + p_n[n] + aux_omega_a[n] + aux_omega_b[n] <= system_param["sys_const_error"] for n in system_data["set_gen_index"])
    
    if flag_developer_mode:        
        print (model.display())

    ## Solver parameters
    model.setParam('MIPGap', system_param["sys_MIPGap"])
    model.setParam('Timelimit', system_param["sys_Timelimit"])

    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    print("\n[!] LDT-CC price problem solved")
    print("The optimal value is", round(obj.getValue(),digit_round))
    if flag_developer_mode:
        ## Primal solution
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution beta_n is")
        for v in beta_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution s_i is")
        for v in s_i.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution omega is")
        print("{}: {}".format("omega_input" ,sys_w_star))
                
    # Save solution    
    solution_dict = dict()
    solution_dict["o_opt"] = round(obj.getValue(),digit_round)
    solution_dict["p_opt"] = []
    solution_dict["a_opt"] = []
    solution_dict["b_opt"] = []
    solution_dict["p_price"] = []
    solution_dict["a_price"] = round(const_op_reserve.Pi,digit_round)
    solution_dict["b_price"] = round(const_ad_reserve.Pi,digit_round)

    for v in p_n.values():
        solution_dict["p_opt"].append(round(v.X,digit_round))
    for v in alpha_n.values():
        solution_dict["a_opt"].append(round(v.X,digit_round))
    for v in beta_n.values():
        solution_dict["b_opt"].append(round(v.X,digit_round))
    for c in const_demand.values():
        solution_dict["p_price"].append(round(c.Pi,digit_round))  

    return solution_dict

def ED_test_scenarios(system_data, system_param, system_solution, system_solution_price, sys_w_mean, sys_w_sigma, flag_developer_mode = False, digit_round = 4):

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(system_data["set_lines_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="alpha_n", lb=system_param["sys_math_error"]) 
    beta_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="beta_n", lb=system_param["sys_math_error"]) 

    # Set objective function
    obj_value = model.setObjective(sum(p_n[n]*system_solution_price["p_price"][n] + alpha_n[n]*system_solution_price["a_price"][n] + beta_n[n]*system_solution_price["b_price"][n] for n in system_data["set_gen_index"]) 
                                   + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE) 
    
    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] + (alpha_n[n]+beta_n[n])*sys_w_sigma[0] for n in system_data["node_gens"][i])
                                 + sum((sys_w_mean-sys_w_sigma[0])*system_data["wind_factor"][n] for n in system_data["node_winds"][i]) 
                                 - sum(f_ij[i,j] for (i,j) in system_data["node_lines_in"][i])
                                 + sum(f_ij[i,j] for (i,j) in system_data["node_lines_out"][i])
                                 + s_i[i] == system_data["node_demand"][i] for i in system_data["set_node_index"]), name='const_demand')    # Demand constraint
    
    const_pmax = model.addConstrs((p_n[n] + (alpha_n[n]+beta_n[n])*sys_w_sigma[0] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax')
    const_ft = model.addConstrs((system_data["line_susceptance_matrix"][i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_ft')
    const_fmax = model.addConstrs((f_ij[i,j] <= system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmin')
    const_t0 = model.addConstr(t_i[0] == 0)
    
    const_pmax_2nd = model.addConstrs((p_n[n] <= system_solution["p_opt"][n] for n in system_data["set_gen_index"]), name='const_pmax_2nd')
    const_op_reserve_2nd = model.addConstrs((alpha_n[n] <= system_solution["a_opt"][n] for n in system_data["set_gen_index"]), name='cons_op_reserve_2nd') # Operational reserve constraint
    const_ad_reserve_2nd = model.addConstrs((beta_n[n] <= system_solution["b_opt"][n] for n in system_data["set_gen_index"]), name='cons_ad_reserve_2nd') # Adversarial reserve constraint

    #const_op_reserve = model.addConstr(sum(alpha_n[n] for n in system_data["set_gen_index"]) <= 1, name='const_op_reserve') # Operational reserve constraint
    #const_ad_reserve = model.addConstr(sum(beta_n[n] for n in system_data["set_gen_index"]) <= 1, name='cons_ad_reserve')  # Adversarial reserve constraint
    const_pmax_3th = model.addConstrs((p_n[n] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax_3th')
    
    # Solve problem
    model.optimize()
    
    # Print results
    obj = model.getObjective()
    print("\n[!] Evaluation scenario solved")
    print("The optimal value is", round(obj.getValue(),digit_round))
    if flag_developer_mode:
        ## Primal solution
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution beta_n is")
        for v in beta_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution s_i is")
        for v in s_i.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")

    # Save solution    
    solution_dict = dict()
    solution_dict["o_opt"] = round(obj.getValue(),digit_round)
    solution_dict["p_opt"] = []
    solution_dict["a_opt"] = []
    solution_dict["b_opt"] = []
    
    for v in p_n.values():
        solution_dict["p_opt"].append(round(v.X,digit_round))
    for v in alpha_n.values():
        solution_dict["a_opt"].append(round(v.X,digit_round))
    for v in beta_n.values():
        solution_dict["b_opt"].append(round(v.X,digit_round))
    
    return solution_dict

def test_scenarios(system_data, system_param, system_solution, scenarios_max, test_list, digit_round = 4, r_seed = 0):
    random.seed(r_seed)

    solution_dict = dict()
    solution_dict["omega"] = dict()
    solution_dict["omega"]["list_omega"] = []

    for m in test_list:
        solution_dict[m] = dict()
        solution_dict[m]["list_opt"] = []

    sys_w_mean = sum(system_data["wind_mean"].values())
    sys_w_sigma = sum(system_data["wind_std"].values())    
    for _ in range(scenarios_max):
        sys_w_sigma_scenario = np.random.normal(0, sys_w_sigma, 1)
        solution_dict["omega"]["list_omega"].append(sys_w_sigma_scenario)

        for m in test_list:
            aux_price = dict()
            if m == "CC":
                system_solution[m]["b_opt"] = np.zeros(len(system_data["set_gen_index"]))
                aux_price["p_price"] = find_price(system_data, system_solution["CC"])
                aux_price["a_price"] = np.ones(len(system_data["set_gen_index"]))*system_solution["CC"]["a_price"]
                aux_price["b_price"] = np.ones(len(system_data["set_gen_index"]))*system_param["sys_cost_voll"]  
            elif m == "WCC":
                system_solution[m]["b_opt"] = np.zeros(len(system_data["set_gen_index"]))
                aux_price["p_price"] = find_price(system_data, system_solution["WCC"])
                aux_price["a_price"] = np.ones(len(system_data["set_gen_index"]))*system_solution["WCC"]["a_price"]
                aux_price["b_price"] = np.ones(len(system_data["set_gen_index"]))*system_param["sys_cost_voll"]      
            elif m == "LDT-CC":
                aux_price["p_price"] = find_price(system_data, system_solution["LDT-CC-price"])
                aux_price["a_price"] = np.ones(len(system_data["set_gen_index"]))*system_solution["LDT-CC-price"]["a_price"]
                aux_price["b_price"] = np.ones(len(system_data["set_gen_index"]))*system_solution["LDT-CC-price"]["b_price"]
            elif m == "LDT-WCC":
                aux_price["p_price"] = find_price(system_data, system_solution["LDT-WCC"])
                aux_price["a_price"] = np.ones(len(system_data["set_gen_index"]))*system_solution["LDT-WCC"]["a_price"]
                aux_price["b_price"] = np.ones(len(system_data["set_gen_index"]))*system_solution["LDT-WCC"]["b_price"]    
            
            opt = ED_test_scenarios(system_data, system_param, system_solution[m], aux_price, sys_w_mean, sys_w_sigma_scenario)
            solution_dict[m]["list_opt"].append(opt["o_opt"])

    for m in test_list:
        solution_dict[m]["mean"] = round(np.mean(solution_dict[m]["list_opt"]),digit_round) 
        solution_dict[m]["std"] = round(np.std(solution_dict[m]["list_opt"]),digit_round)
        solution_dict[m]["max"] = round(np.max(solution_dict[m]["list_opt"]),digit_round)
        solution_dict[m]["min"] = round(np.min(solution_dict[m]["list_opt"]),digit_round)

    solution_dict["omega"]["mean"] = round(np.mean(solution_dict["omega"]["list_omega"]),digit_round)
    solution_dict["omega"]["std"] = round(np.std(solution_dict["omega"]["list_omega"]),digit_round)
    solution_dict["omega"]["max"] = round(np.max(solution_dict["omega"]["list_omega"]),digit_round)
    solution_dict["omega"]["min"] = round(np.min(solution_dict["omega"]["list_omega"]),digit_round) 

    return solution_dict





def ED_LDT_CC_alternative(system_data, system_param, flag_developer_mode = False, digit_round = 4):
    #Auxilar parameters
    sys_inv_phi_eps = norm.ppf(1- system_param["sys_epsilon"])
    sys_inv_phi_eps_ext = norm.ppf(1-system_param["sys_epsilon_ext"])
    sys_w_mean = sum(system_data["wind_mean"].values())
    sys_w_sigma = sum(system_data["wind_std"].values())

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(system_data["set_lines_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="alpha_n", lb=system_param["sys_math_error"])  # Operational reserve variable
    beta_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="beta_n", lb=system_param["sys_math_error"])   # Adversarial reserve variable

    omega_star = model.addVar(vtype=GRB.CONTINUOUS, name="omega_star", lb=-GRB.INFINITY) # Optimal critical value from the extreme event region
    lambda_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="lambda_n", lb=-GRB.INFINITY)     # Dual variable of the constrainst to obtain "omega_star"

    #aux_omega_a = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_omega_a", lb=-GRB.INFINITY)   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_omega_b = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_omega_b", lb=-GRB.INFINITY)   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
    #aux_lambda_a = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_lambda_a", lb=-GRB.INFINITY) # Auxilar value to represent the positive value of "lambda_n*(alpha_n +beta_n)" in the SOC
    aux_lambda_b = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_lambda_b", lb=-GRB.INFINITY) # Auxilar value to represent the negative value of "lambda_n*(alpha_n +beta_n)" in the SOC
    
    # Set objective function
    #obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
    #                               + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
                                   + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE) 
    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in system_data["node_gens"][i]) 
                                     + sum(sys_w_mean*system_data["wind_factor"][n] for n in system_data["node_winds"][i]) 
                                     - sum(f_ij[i,j] for (i,j) in system_data["node_lines_in"][i])
                                     + sum(f_ij[i,j] for (i,j) in system_data["node_lines_out"][i])
                                     + s_i[i] == system_data["node_demand"][i] for i in system_data["set_node_index"]), name='const_demand')
    const_pmax = model.addConstrs((p_n[n] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax')
    const_ft = model.addConstrs((system_data["line_susceptance_matrix"][i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_ft')
    const_fmax = model.addConstrs((f_ij[i,j] <= system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmin')
    const_t0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in system_data["set_gen_index"]) == 1, name='const_op_reserve') # Operational reserve constraint
    const_ad_reserve = model.addConstr(sum(beta_n[n] for n in system_data["set_gen_index"]) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint
    const_pmax_CC = model.addConstrs(p_n[n] - system_data["gen_pmax"][n] + alpha_n[n]*sys_inv_phi_eps*sys_w_sigma <= 0 for n in system_data["set_gen_index"]) # Max generation limit constraint

    #model.addConstrs(aux_omega_a[n] == omega_star*alpha_n[n] for n in system_data["set_gen_index"])
    model.addConstrs(aux_omega_b[n] == (omega_star-sys_w_sigma*sys_inv_phi_eps)*beta_n[n] for n in system_data["set_gen_index"])

    #model.addConstrs(aux_lambda_a[n] == lambda_n[n]*alpha_n[n] for n in system_data["set_gen_index"])
    model.addConstrs(aux_lambda_b[n] == lambda_n[n]*beta_n[n] for n in system_data["set_gen_index"])
    ## LDT Constraint
    model.addConstrs(-system_data["gen_pmax"][n] + p_n[n] + sys_inv_phi_eps*alpha_n[n]*sys_w_sigma + aux_omega_b[n] == 0 for n in system_data["set_gen_index"])
    model.addConstr(-sys_w_sigma**(-1.0)*omega_star - sys_inv_phi_eps_ext <= 0)
    model.addConstrs(sys_w_sigma**(-2.0)*(omega_star) - aux_lambda_b[n] == 0 for n in system_data["set_gen_index"])
    
    if flag_developer_mode:        
        print (model.display())

    ## Allow QCP dual 
    model.Params.QCPDual = 1
    model.Params.NonConvex = 2
    model.setParam('MIPGap', system_param["sys_MIPGap"])
    model.setParam('Timelimit', system_param["sys_Timelimit"])

    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    print("\n[!] LDT-CC problem solved")
    print("The optimal value is", round(obj.getValue(),digit_round))
    if flag_developer_mode:
        ## Primal solution
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution beta_n is")
        for v in beta_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution s_i is")
        for v in s_i.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution omega_star is")
        print("{}: {}".format(omega_star.varName, omega_star.X))
        print("A solution lambda_n is")
        for v in lambda_n.values():
            print("{}: {}".format(v.varName, v.X))
                
    # Save solution    
    solution_dict = dict()
    solution_dict["o_opt"] = round(obj.getValue(),digit_round)
    solution_dict["p_opt"] = []
    solution_dict["a_opt"] = []
    solution_dict["b_opt"] = []
    solution_dict["w_opt"] = omega_star.X
    solution_dict["l_opt"] = []

    for v in p_n.values():
        solution_dict["p_opt"].append(round(v.X,digit_round))
    for v in alpha_n.values():
        solution_dict["a_opt"].append(round(v.X,digit_round))
    for v in beta_n.values():
        solution_dict["b_opt"].append(round(v.X,digit_round))
    for v in lambda_n.values():
        solution_dict["l_opt"].append(v.X)

    return solution_dict

def ED_LDT_CC_alternative_price(system_data, system_param, sys_w_star, flag_developer_mode = False, digit_round = 4):
    #Auxilar parameters
    sys_inv_phi_eps = norm.ppf(1- system_param["sys_epsilon"])
    sys_inv_phi_eps_ext = norm.ppf(system_param["sys_epsilon_ext"])
    sys_w_mean = sum(system_data["wind_mean"].values())
    sys_w_sigma = sum(system_data["wind_std"].values())

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(system_data["set_lines_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="alpha_n", lb=system_param["sys_math_error"])  # Operational reserve variable
    beta_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="beta_n", lb=system_param["sys_math_error"])   # Adversarial reserve variable

    #aux_omega_a = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_omega_a", lb=-GRB.INFINITY)   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_omega_b = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name= "aux_omega_b", lb=-GRB.INFINITY)   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
 
    # Set objective function
    #obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
    #                               + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
                                   + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE) 
    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in system_data["node_gens"][i]) 
                                     + sum(sys_w_mean*system_data["wind_factor"][n] for n in system_data["node_winds"][i]) 
                                     - sum(f_ij[i,j] for (i,j) in system_data["node_lines_in"][i])
                                     + sum(f_ij[i,j] for (i,j) in system_data["node_lines_out"][i])
                                     + s_i[i] == system_data["node_demand"][i] for i in system_data["set_node_index"]), name='const_demand')
    const_pmax = model.addConstrs((p_n[n] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax')
    const_ft = model.addConstrs((system_data["line_susceptance_matrix"][i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_ft')
    const_fmax = model.addConstrs((f_ij[i,j] <= system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmin')
    const_t0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in system_data["set_gen_index"]) == 1, name='const_op_reserve') # Operational reserve constraint
    const_ad_reserve = model.addConstr(sum(beta_n[n] for n in system_data["set_gen_index"]) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint
    const_pmax_CC = model.addConstrs(p_n[n] - system_data["gen_pmax"][n] + alpha_n[n]*sys_inv_phi_eps*sys_w_sigma <= 0 for n in system_data["set_gen_index"]) # Max generation limit constraint

    model.addConstrs(aux_omega_b[n] == (sys_w_star-sys_w_sigma*sys_inv_phi_eps)*beta_n[n] for n in system_data["set_gen_index"])
    ## LDT Constraint
    model.addConstrs(-system_data["gen_pmax"][n] + p_n[n] + sys_inv_phi_eps*alpha_n[n]*sys_w_sigma + aux_omega_b[n] <= system_param["sys_const_error"] for n in system_data["set_gen_index"])
    



    if flag_developer_mode:        
        print (model.display())

    ## Solver parameters
    model.setParam('MIPGap', system_param["sys_MIPGap"])
    model.setParam('Timelimit', system_param["sys_Timelimit"])

    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    print("\n[!] LDT-CC price problem solved")
    print("The optimal value is", round(obj.getValue(),digit_round))
    if flag_developer_mode:
        ## Primal solution
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution beta_n is")
        for v in beta_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution s_i is")
        for v in s_i.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution omega is")
        print("{}: {}".format("omega_input" ,sys_w_star))
                
    # Save solution    
    solution_dict = dict()
    solution_dict["o_opt"] = round(obj.getValue(),digit_round)
    solution_dict["p_opt"] = []
    solution_dict["a_opt"] = []
    solution_dict["b_opt"] = []
    solution_dict["p_price"] = []
    solution_dict["a_price"] = round(const_op_reserve.Pi,digit_round)
    solution_dict["b_price"] = round(const_ad_reserve.Pi,digit_round)

    for v in p_n.values():
        solution_dict["p_opt"].append(round(v.X,digit_round))
    for v in alpha_n.values():
        solution_dict["a_opt"].append(round(v.X,digit_round))
    for v in beta_n.values():
        solution_dict["b_opt"].append(round(v.X,digit_round))
    for c in const_demand.values():
        solution_dict["p_price"].append(round(c.Pi,digit_round))  

    return solution_dict

def ED_LDT_WCC_alternative(system_data, system_param, sys_w_star, flag_developer_mode = False, digit_round = 4):
    #Auxilar parameters
    sys_inv_phi_eps = norm.ppf(1- system_param["sys_epsilon"])
    sys_inv_phi_eps_ext = norm.ppf(system_param["sys_epsilon_ext"])
    sys_w_mean = sum(system_data["wind_mean"].values())
    sys_w_sigma = sum(system_data["wind_std"].values())

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(system_data["set_lines_node_index"], vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(system_data["set_node_index"], vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="alpha_n", lb=system_param["sys_math_error"])  # Operational reserve variable
    beta_n = model.addVars(system_data["set_gen_index"], vtype=GRB.CONTINUOUS, name="beta_n", lb=system_param["sys_math_error"])   # Adversarial reserve variable

    # Set objective function
    #obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
    #                               + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    obj_value = model.setObjective(sum(p_n[n]*system_data["gen_c1"][n] + (p_n[n]**2)*system_data["gen_c2"][n] + (alpha_n[n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n] + beta_n[n]*system_data["gen_cbeta"][n] for n in system_data["set_gen_index"]) 
                                   + sum(s_i[i]*system_param["sys_cost_voll"] for i in system_data["set_node_index"]), GRB.MINIMIZE)
    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in system_data["node_gens"][i]) 
                                     + sum(sys_w_mean*system_data["wind_factor"][n] for n in system_data["node_winds"][i]) 
                                     - sum(f_ij[i,j] for (i,j) in system_data["node_lines_in"][i])
                                     + sum(f_ij[i,j] for (i,j) in system_data["node_lines_out"][i])
                                     + s_i[i] == system_data["node_demand"][i] for i in system_data["set_node_index"]), name='const_demand')
    const_pmax = model.addConstrs((p_n[n] <= system_data["gen_pmax"][n] for n in system_data["set_gen_index"]), name='const_pmax')
    const_ft = model.addConstrs((system_data["line_susceptance_matrix"][i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_ft')
    const_fmax = model.addConstrs((f_ij[i,j] <= system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -system_data["line_fmax_matrix"][i,j] for (i,j) in system_data["set_lines_node_index"]), name='const_fmin')
    const_t0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in system_data["set_gen_index"]) == 1, name='const_op_reserve') # Operational reserve constraint
    const_ad_reserve = model.addConstr(sum(beta_n[n] for n in system_data["set_gen_index"]) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint
    #const_pmax_CC = model.addConstrs(p_n[n] - system_data["gen_pmax"][n] + alpha_n[n]*sys_inv_phi_eps*sys_w_sigma <= 0 for n in system_data["set_gen_index"]) # Max generation limit constraint
    
    # Add cutting planes until the optimal solution satisfy the WCC constraint
    for _ in range(system_param["sys_cp_iterations"]):
        model.optimize()
        obj_value = model.getObjective()
        
        if flag_developer_mode:
            print('ITERATION, %s' % _)
            print("The optimal value is", round(obj_value.getValue(),digit_round))
            print("A solution p_n is")
            for v in p_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))
            print("A solution alpha_n is")
            for v in alpha_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))
            print("A solution beta_n is")
            for v in beta_n.values():
                print("{}: {}".format(v.varName, round(v.X,digit_round)))    

        # Save variables
        p_n_star = dict()
        alpha_n_star = dict()
        beta_n_star = dict()

        z_w_star = 0.75*sys_w_star/sys_w_sigma
        for n in system_data["set_gen_index"]:
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X
            beta_n_star[n] = model.getVarByName('beta_n[%d]'%(n)).X

        # Check if solution satisfy the WCC constraint with a piece-wise policy 
        if all(
            truncated_normal_funtion(p_n_star[n]-system_data["gen_pmax"][n] - (alpha_n_star[n])*sys_w_sigma*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"])),
                                    (sys_w_sigma*(alpha_n_star[n]))*np.sqrt(1 - z_w_star*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"]))**2))
                +
            truncated_normal_funtion(p_n_star[n]-system_data["gen_pmax"][n] + (beta_n_star[n]*sys_w_star) + (alpha_n_star[n]-beta_n_star[n])*sys_w_sigma*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"])),
                                    (sys_w_sigma*(alpha_n_star[n]-beta_n_star[n]))*np.sqrt(1 + z_w_star*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"]))**2))
            <= system_param["sys_epsilon"] for n in system_data["set_gen_index"]):
            break

        # Add a cutting plane to the model 
        for n in system_data["set_gen_index"]:
            mean_lower_star = p_n_star[n]-system_data["gen_pmax"][n] - (alpha_n_star[n])*sys_w_sigma*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"]))
            std_lower_star = (sys_w_sigma*(alpha_n_star[n]))*np.sqrt(1 - z_w_star*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"]))**2)
            mean_greater_star = p_n_star[n]-system_data["gen_pmax"][n] + (beta_n_star[n]*sys_w_star) + (alpha_n_star[n]-beta_n_star[n])*sys_w_sigma*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"]))
            std_greater_star = (sys_w_sigma*(alpha_n_star[n]-beta_n_star[n]))*np.sqrt(1 + z_w_star*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"]))**2)
            if truncated_normal_funtion(mean_lower_star, std_lower_star) + truncated_normal_funtion(mean_greater_star, std_greater_star) > system_param["sys_epsilon"]:
                if flag_developer_mode:
                    print('ERROR in iterration, %s, with generador, %s' %(_,n))
                    print("tnf {}, epsilon {}".format(round(truncated_normal_funtion(mean_lower_star, std_lower_star) + truncated_normal_funtion(mean_greater_star, std_greater_star), digit_round),system_param["sys_epsilon"]))
                model.addConstr(
                    truncated_normal_funtion(mean_lower_star,std_lower_star)
                    + ((p_n[n]-system_data["gen_pmax"][n] - (alpha_n[n])*sys_w_sigma*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)))) - mean_lower_star)*truncated_normal_funtion_dmu(mean_lower_star, std_lower_star) 
                    + (((alpha_n[n])*sys_w_sigma)*np.sqrt(1 - z_w_star*(norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(norm.cdf(z_w_star)+system_param["sys_math_error"]))**2) - std_lower_star)*truncated_normal_funtion_dsigma(mean_lower_star, std_lower_star) 
                    + truncated_normal_funtion(mean_greater_star,std_greater_star)
                    + ((p_n[n]-system_data["gen_pmax"][n] + (beta_n_star[n]*sys_w_star) + (alpha_n[n]-beta_n[n])*sys_w_sigma*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star))) ) - mean_greater_star)*truncated_normal_funtion_dmu(mean_greater_star,std_greater_star)
                    + (((alpha_n[n]-beta_n[n])*sys_w_sigma)*np.sqrt(1 + z_w_star*(norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"])) - (norm.pdf(z_w_star)/(1-norm.cdf(z_w_star)+system_param["sys_math_error"]))**2) - std_greater_star)*truncated_normal_funtion_dsigma(mean_greater_star,std_greater_star)
                    <= system_param["sys_epsilon"])

        if flag_developer_mode:        
            print (model.display())
        
    ## Solve parameters
    model.setParam('MIPGap', system_param["sys_MIPGap"])
    model.setParam('Timelimit', system_param["sys_Timelimit"])

    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    print("\n[!] LDT-WCC problem solved in {} iterations".format(_+1))
    print("The optimal value is", round(obj.getValue(),digit_round))
    if flag_developer_mode:
        ## Primal solution
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution beta_n is")
        for v in beta_n.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))
        print("A solution s_i is")
        for v in s_i.values():
            print("{}: {}".format(v.varName, round(v.X,digit_round)))

        # Dual results
        for c in const_demand.values():
            print(c.constrName, round(c.Pi, digit_round))
        print(const_op_reserve.constrName, round(const_op_reserve.Pi, digit_round))
        print(const_ad_reserve.constrName, round(const_ad_reserve.Pi, digit_round))

    # Save solution    
    solution_dict = dict()
    solution_dict["o_opt"] = round(obj.getValue(),digit_round)
    solution_dict["p_opt"] = []
    solution_dict["a_opt"] = []
    solution_dict["b_opt"] = []
    solution_dict["p_price"] = []
    solution_dict["a_price"] = round(const_op_reserve.Pi,digit_round)
    solution_dict["b_price"] = round(const_ad_reserve.Pi,digit_round)

    for v in p_n.values():
        solution_dict["p_opt"].append(round(v.X,digit_round))
    for v in alpha_n.values():
        solution_dict["a_opt"].append(round(v.X,digit_round))
    for v in beta_n.values():
        solution_dict["b_opt"].append(round(v.X,digit_round))
    for c in const_demand.values():
        solution_dict["p_price"].append(round(c.Pi,digit_round))  

    return solution_dict
