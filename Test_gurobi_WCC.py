import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm

# Instruction variables
aux_test = False

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
    term_2 = (1/np.sqrt(2*np.pi))*(1-((mu**2)/(sigma**2)))*np.exp((-1/2)*z_aux**2)
    wcc_dsigma = term_1 + term_2
    return wcc_dsigma

# First order Taylor aproximation of the truncated normal function 
def taylor_approximation(mu,sigma,a,b):
    term_1 = truncated_normal_funtion(a,b)
    term_2 = truncated_normal_funtion_dmu(a,b)*(mu-a)
    term_3 = truncated_normal_funtion_dsigma(a,b)*(sigma-b)
    t_approx = term_1 + term_2 + term_3
    return t_approx

if aux_test == True:
    cdf_mean  = 12
    cdf_deviation = 5
    test_tnf = truncated_normal_funtion(cdf_mean,cdf_deviation)

    ref_mean = 11
    ref_deviation = 5
    test_taylor = taylor_approximation(cdf_mean,cdf_deviation,ref_mean,ref_deviation)

    print('function_tnf, %s' % test_tnf)
    print('function_taylor, %s' % test_taylor)

######## functions ####
# pdf (density)
# norm.pdf(x)
# cummulative cdf   
# norm.cdf(-1.96)
# inverse cdf
# norm.ppf(norm.cdf(1.96))

def cutting_planes_problem_ED():
    flag_print = False
    # Parameters
    N = 3
    d = np.array([120])

    c_p_n = np.array([15,30, 45]) # 25, 66
    c_alpha_n = np.array([50,60, 60]) # 40, 80
    p_n_max = np.array([30,30,50])

    epsilon = 0.07
    inv_phi_eps = norm.ppf(1-epsilon)

    w_bar = 20
    w_sigma = 4

    # Define model
    model = gb.Model()

    # Variables
    p_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    alpha_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="alpha_n", lb=0) # Operational reserve variable

    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar == d, name='cons_demand') # Energy balance constraint
    cons_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_reserve')   # Operational reserve

    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] for n in range(N)),GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Add cutting planes until the optimal solution satisfy the WCC constraint
    n_iterations = 3
    continue_flag = True
    for _ in range(n_iterations):
        model.optimize()
        obj_value = model.getObjective()
        
        if flag_print == True:
            print('ITERATION, %s' % _)
            print("The optimal value is", obj_value.getValue())
            print("A solution p_n is")
            print("{}: {}".format(p_n.varName, p_n.X))
            print("A solution alpha_n is")
            print("{}: {}".format(alpha_n.varName, alpha_n.X))

        # Save variables
        p_n_star = dict()
        alpha_n_star = dict()

        for n in range(N):
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X

        # Check if solution satisfy the WCC constraint
        if all(p_n_star[n] + alpha_n_star[n]*inv_phi_eps*w_sigma <=p_n_max[n] for n in range(N)):
            break

        # Add a cutting plane to the model 
        for n in range(N):
            if p_n_star[n] + alpha_n_star[n]*inv_phi_eps*w_sigma >=p_n_max[n]:
                print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(p_n[n] + alpha_n[n]*inv_phi_eps*w_sigma <=p_n_max[n])

    ## Primal solution
    obj_value = model.getObjective() 
    print("\nStatus:", model.status)        
    print("The optimal value is", obj_value.getValue())
    print("A solution p_n is")
    print("{}: {}".format(p_n.varName, p_n.X))
    print("A solution alpha_n is")
    print("{}: {}".format(alpha_n.varName, alpha_n.X))

def cutting_planes_problem_WCC_ED():
    flag_print = False
    # Parameters
    N = 3
    d = np.array([120])

    c_p_n = np.array([15,30, 45]) # 25, 66
    c_alpha_n = np.array([50,60, 60]) # 40, 80
    p_n_max = np.array([30,30,50])

    epsilon = 0.07
    inv_phi_eps = norm.ppf(1-epsilon)

    w_bar = 20
    w_sigma = 4

    # Define model
    model = gb.Model()

    # Variables
    p_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    alpha_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="alpha_n", lb=1e-8) # Operational reserve variable

    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar == d, name='cons_demand') # Energy balance constraint
    cons_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_reserve')   # Operational reserve

    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] for n in range(N)),GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Add cutting planes until the optimal solution satisfy the WCC constraint
    n_iterations = 100
    continue_flag = True
    for _ in range(n_iterations):
        model.optimize()
        obj_value = model.getObjective()

        if flag_print == True:
            print('ITERATION, %s' % _)
            print("The optimal value is", obj_value.getValue())
            print("A solution p_n is")
            print("{}: {}".format(p_n.varName, p_n.X))
            print("A solution alpha_n is")
            print("{}: {}".format(alpha_n.varName, alpha_n.X))

        # Save variables
        p_n_star = dict()
        alpha_n_star = dict()

        for n in range(N):
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X

        # Check if solution satisfy the WCC constraint
        if all(p_n_star[n] + alpha_n_star[n]*inv_phi_eps*w_sigma <=p_n_max[n] for n in range(N)):
            break

        # Add a cutting plane to the model 
        for n in range(N):
            media = p_n_star[n]-p_n_max[n]
            desviacion = alpha_n_star[n]*w_sigma
            if truncated_normal_funtion(media, desviacion) >= epsilon:
                print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(truncated_normal_funtion(media, desviacion) + ((p_n[n]-p_n_max[n]) - media)*truncated_normal_funtion_dmu(media, desviacion) + ((alpha_n[n]*w_sigma) - desviacion)*truncated_normal_funtion_dsigma(media, desviacion) <= epsilon)
        
        '''
        term_1 = truncated_normal_funtion(a,b)
        term_2 = truncated_normal_funtion_dmu(a,b)*(mu-a)
        term_3 = truncated_normal_funtion_dsigma(a,b)*(sigma-b)
        t_approx = term_1 + term_2 + term_3
        '''

    ## Primal solution
    obj_value = model.getObjective() 
    print("\nStatus:", model.status)        
    print("The optimal value is", obj_value.getValue())
    print("A solution p_n is")
    print("{}: {}".format(p_n.varName, p_n.X))
    print("A solution alpha_n is")
    print("{}: {}".format(alpha_n.varName, alpha_n.X))

def cutting_planes_problem_WCC_PW():
    flag_print = False
    # Parameters
    N = 3
    d = np.array([120])

    c_p_n = np.array([15,30, 45]) # 25, 66
    c_alpha_n = np.array([50,60, 60]) # 40, 80
    p_n_max = np.array([30,30,50])

    epsilon = 0.07
    inv_phi_eps = norm.ppf(1-epsilon)

    w_bar = 20
    w_sigma = 4

    # Define model
    model = gb.Model()

    # Variables
    p_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    alpha_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="alpha_n", lb=1e-8) # Operational reserve variable

    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar == d, name='cons_demand') # Energy balance constraint
    cons_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_reserve')   # Operational reserve

    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] for n in range(N)),GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Add cutting planes until the optimal solution satisfy the WCC constraint
    n_iterations = 100
    continue_flag = True
    for _ in range(n_iterations):
        model.optimize()
        obj_value = model.getObjective()

        if flag_print == True:
            print('ITERATION, %s' % _)
            print("The optimal value is", obj_value.getValue())
            print("A solution p_n is")
            print("{}: {}".format(p_n.varName, p_n.X))
            print("A solution alpha_n is")
            print("{}: {}".format(alpha_n.varName, alpha_n.X))

        # Save variables
        p_n_star = dict()
        alpha_n_star = dict()

        for n in range(N):
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X

        # Check if solution satisfy the WCC constraint
        if all(p_n_star[n] + alpha_n_star[n]*inv_phi_eps*w_sigma <=p_n_max[n] for n in range(N)):
            break

        # Add a cutting plane to the model 
        for n in range(N):
            w_critic = 20
            z_auxiliar = w_critic/w_sigma
            media_lower = p_n_star[n]-p_n_max[n] + alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)
            desviacion_lower = np.sqrt((alpha_n_star[n]*w_sigma)**2 * (1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2))

            media_greater = p_n_star[n]-p_n_max[n] + alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))
            desviacion_greater = np.sqrt((alpha_n_star[n]*w_sigma)**2 * (1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2))

            if truncated_normal_funtion(media_lower, desviacion_lower) + truncated_normal_funtion(media_greater, desviacion_greater) >= epsilon:
                print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(
                    truncated_normal_funtion(media_lower, desviacion_lower) 
                    + ((p_n[n]-p_n_max[n]) - media_lower)*truncated_normal_funtion_dmu(media_lower, desviacion_lower) 
                    + ((alpha_n[n]*w_sigma) - desviacion_lower)*truncated_normal_funtion_dsigma(media_lower, desviacion_lower) 
                    + truncated_normal_funtion(media_greater, desviacion_greater) 
                    + ((p_n[n]-p_n_max[n]) - media_greater)*truncated_normal_funtion_dmu(media_greater, desviacion_greater) 
                    + ((alpha_n[n]*w_sigma) - desviacion_greater)*truncated_normal_funtion_dsigma(media_greater, desviacion_greater)
                    <= epsilon)
        
        '''
        term_1 = truncated_normal_funtion(a,b)
        term_2 = truncated_normal_funtion_dmu(a,b)*(mu-a)
        term_3 = truncated_normal_funtion_dsigma(a,b)*(sigma-b)
        t_approx = term_1 + term_2 + term_3
        '''

    ## Primal solution
    obj_value = model.getObjective() 
    print("\nStatus:", model.status)        
    print("The optimal value is", obj_value.getValue())
    print("A solution p_n is")
    print("{}: {}".format(p_n.varName, p_n.X))
    print("A solution alpha_n is")
    print("{}: {}".format(alpha_n.varName, alpha_n.X))

#cutting_planes_problem_ED()
#cutting_planes_problem_WCC_ED()
cutting_planes_problem_WCC_PW()

'''
Status: 2
The optimal value is 3210.0
A solution p_n is
['p_n[0]' 'p_n[1]' 'p_n[2]']: [30. 30. 40.]
A solution alpha_n is
['alpha_n[0]' 'alpha_n[1]' 'alpha_n[2]']: [0. 0. 1.]

Status: 2
The optimal value is 3207.3433728233554
A solution p_n is
['p_n[0]' 'p_n[1]' 'p_n[2]']: [30.05541946 30.05543125 39.88914929]
A solution alpha_n is
['alpha_n[0]' 'alpha_n[1]' 'alpha_n[2]']: [0.01625746 0.01623161 0.96751093]

Status: 2
The optimal value is 3208.499514123805
A solution p_n is
['p_n[0]' 'p_n[1]' 'p_n[2]']: [30.0310841  30.03145518 39.93746072]
A solution alpha_n is
['alpha_n[0]' 'alpha_n[1]' 'alpha_n[2]']: [0.00961351 0.0072081  0.98317839]
4.225607144489479
PS C:\Users\Tom\Desktop\Codes\Research\ChanceConstraints-1> 

'''