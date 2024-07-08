import numpy as np
from scipy.stats import norm
import random

# Normal density function
def pdf_function(x):
    phi = np.exp((-1/2)*x**2)/np.sqrt(2*np.pi)
    return phi

# WCC - truncated normal function
def truncated_normal_funtion(mu,sigma):
    z_aux = (-mu/sigma) 
    wcc = mu*(1- norm.cdf(z_aux)) + (sigma/(np.sqrt(2*np.pi)))*np.exp((-1/2)*z_aux**2)
    return wcc

# dWCC/dmu - derivative of truncated normal function with respect to mu
def truncated_normal_funtion_dmu(mu,sigma):
    z_aux = (-mu/sigma)
    term_1 = (1-norm.cdf(z_aux))
    term_2 = mu*((1/sigma)*norm.pdf(z_aux))
    term_3 = -(mu/(sigma*np.sqrt(2*np.pi)))*np.exp((-1/2)*z_aux**2)
    wcc_dmu = term_1 + term_2 + term_3
    return wcc_dmu

# dWCC/dmsigma - derivative of truncated normal function with respect to sigma
def truncated_normal_funtion_dsigma(mu,sigma):
    z_aux = (-mu/sigma)
    term_1 = -((mu**2)/(sigma**2))*norm.pdf(z_aux)
    term_2 = (1/np.sqrt(2*np.pi))*(1+((mu**2)/(sigma**2)))*np.exp((-1/2)*z_aux**2)
    wcc_dsigma = term_1 + term_2
    return wcc_dsigma

# First order Taylor aproximation of the truncated normal function 
def taylor_approximation(mu,sigma,a,b):
    term_1 = truncated_normal_funtion(a,b)
    term_2 = truncated_normal_funtion_dmu(a,b)*(mu-a)
    term_3 = truncated_normal_funtion_dsigma(a,b)*(sigma-b)
    t_approx = term_1 + term_2 + term_3
    return t_approx



# For each generator find the its price associated with its node 
def find_price(system_data, system_solution):
    price_per_generator = np.zeros(len(system_data["set_gen_index"]))

    for n in system_data["set_gen_index"]:
        for i in system_data["node_gens"]:
            if system_data["gen_node"][n] == i:
                price_per_generator[n] = system_solution["p_price"][i]
    return price_per_generator

# Compute profit, cost, and revenue for each generator
def gen_profit(system_data, system_solution):
    sys_w_sigma = sum(system_data["wind_std"].values())
    price_per_generator = find_price(system_data, system_solution)

    solution_dict = dict()
    solution_dict["p_revenue"] = dict()
    solution_dict["p_cost"] = dict()
    solution_dict["p_profit"] = dict()
    solution_dict["a_revenue"] = dict()
    solution_dict["a_cost"] = dict()
    solution_dict["a_profit"] = dict()
    if "b_opt" in system_solution.keys():
        solution_dict["b_revenue"] = dict()
        solution_dict["b_cost"] = dict()
        solution_dict["b_profit"] = dict()

    for n in system_data["set_gen_index"]:
        solution_dict["p_revenue"][n] = system_solution["p_opt"][n]*price_per_generator[n]
        solution_dict["p_cost"][n] = system_solution["p_opt"][n]*system_data["gen_c1"][n] + (system_solution["p_opt"][n]**2)*system_data["gen_c2"][n]
        solution_dict["p_profit"][n] = solution_dict["p_revenue"][n] - solution_dict["p_cost"][n]
        solution_dict["a_revenue"][n] = system_solution["a_opt"][n]*system_solution["a_price"]
        solution_dict["a_cost"][n] =  (system_solution["a_opt"][n]**2)*(sys_w_sigma**2)*system_data["gen_c2"][n]
        solution_dict["a_profit"][n] = solution_dict["a_revenue"][n] - solution_dict["a_cost"][n]
        if "b_opt" in system_solution.keys():
            solution_dict["b_revenue"][n] = system_solution["b_opt"][n]*system_solution["b_price"]
            solution_dict["b_cost"][n] = system_solution["b_opt"][n]*3*sys_w_sigma*system_data["gen_cbeta"][n]
            solution_dict["b_profit"][n] = solution_dict["b_revenue"][n] - solution_dict["b_cost"][n]

    return solution_dict

# Maps generators dispatch to zone dispatch
def zone_dispatch(system_data, system_solution):
    solution_dict = dict()
    solution_dict["p_zone"] = dict()
    solution_dict["a_zone"] = dict()
    if "b_opt" in system_solution.keys():
        solution_dict["b_zone"] = dict()

    for i in system_data["set_node_index"]:
        solution_dict["p_zone"][i] = 0 + sum(system_solution["p_opt"][n] for n in system_data["node_gens"][i])
        solution_dict["a_zone"][i] = 0 + sum(system_solution["a_opt"][n] for n in system_data["node_gens"][i])
        if "b_opt" in system_solution.keys():
            solution_dict["b_zone"][i] = 0 + sum(system_solution["b_opt"][n] for n in system_data["node_gens"][i])
    return solution_dict

# Compute profit, cost, and revenue for each zone
def zone_profit(system_data, system_solution):
    solution_dict = dict()
    solution_dict["p_revenue_zone"] = dict()
    solution_dict["p_cost_zone"] = dict()
    solution_dict["p_profit_zone"] = dict()
    solution_dict["a_revenue_zone"] = dict()
    solution_dict["a_cost_zone"] = dict()
    solution_dict["a_profit_zone"] = dict()
    if "b_opt" in system_solution.keys():
        solution_dict["b_revenue_zone"] = dict()
        solution_dict["b_cost_zone"] = dict()
        solution_dict["b_profit_zone"] = dict()
    
    gen_profit_dict = gen_profit(system_data, system_solution)


    for i in system_data["set_node_index"]:
        solution_dict["p_revenue_zone"][i] = 0 + sum(gen_profit_dict["p_revenue"][n] for n in system_data["node_gens"][i])
        solution_dict["p_cost_zone"][i] = 0 + sum(gen_profit_dict["p_cost"][n] for n in system_data["node_gens"][i])
        solution_dict["p_profit_zone"][i] = 0 + sum(gen_profit_dict["p_profit"][n] for n in system_data["node_gens"][i])
        solution_dict["a_revenue_zone"][i] = 0 + sum(gen_profit_dict["a_revenue"][n] for n in system_data["node_gens"][i])
        solution_dict["a_cost_zone"][i] = 0 + sum(gen_profit_dict["a_cost"][n] for n in system_data["node_gens"][i])
        solution_dict["a_profit_zone"][i] = 0 + sum(gen_profit_dict["a_profit"][n] for n in system_data["node_gens"][i])
        if "b_opt" in system_solution.keys():
            solution_dict["b_revenue_zone"][i] = 0 + sum(gen_profit_dict["b_revenue"][n] for n in system_data["node_gens"][i])
            solution_dict["b_cost_zone"][i] = 0 + sum(gen_profit_dict["b_cost"][n] for n in system_data["node_gens"][i])    
            solution_dict["b_profit_zone"][i] = 0 + sum(gen_profit_dict["b_profit"][n] for n in system_data["node_gens"][i])    

    return solution_dict

