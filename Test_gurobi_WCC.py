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
    c_alpha_n = np.array([30,40, 40]) # 50
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
    c_alpha_n = np.array([30,40, 40]) # 50
    p_n_max = np.array([30,30,50])

    epsilon = 0.08
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
    n_iterations = 150
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
        if all(truncated_normal_funtion(p_n_star[n]-p_n_max[n], alpha_n_star[n]*w_sigma) <= epsilon for n in range(N)):
            for n in range(N):
                media = p_n_star[n]-p_n_max[n]
                desviacion = alpha_n_star[n]*w_sigma
                print('ITERATION, %s, gen, %s' %(_,n))
                print("media {}, sd {}".format(media,desviacion))
                print("tnf {}, epsilon {}".format(truncated_normal_funtion(media, desviacion),epsilon))
            break

        # Add a cutting plane to the model 
        for n in range(N):
            media = p_n_star[n]-p_n_max[n]
            desviacion = alpha_n_star[n]*w_sigma
            print('ITERATION, %s, gen, %s' %(_,n))
            print("media {}, sd {}".format(media,desviacion))
            print("tnf {}, epsilon {}".format(truncated_normal_funtion(media, desviacion),epsilon))
           
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
    c_alpha_n = np.array([30,40, 40]) # 50
    p_n_max = np.array([30,30,50])

    epsilon = 0.08
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

        w_critic = -20
        z_auxiliar = w_critic/w_sigma
        # Check if solution satisfy the WCC constraint
        if all(
            truncated_normal_funtion(p_n_star[n]-p_n_max[n] - alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar),
                np.sqrt((w_sigma*alpha_n_star[n])**2 *(1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2)))+
            truncated_normal_funtion(p_n_star[n]-p_n_max[n] - alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)),
                np.sqrt((w_sigma*alpha_n_star[n])**2 *(1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2))) 
            <= epsilon for n in range(N)):
            break

        # Add a cutting plane to the model 
        for n in range(N):
            media_lower = p_n_star[n]-p_n_max[n] - alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)
            desviacion_lower = np.sqrt((w_sigma*alpha_n_star[n])**2 *(1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2))

            media_greater = p_n_star[n]-p_n_max[n] + alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))
            desviacion_greater = np.sqrt(((w_sigma*alpha_n_star[n])**2) *(1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2))
            print('ITERATION, %s, gen, %s' %(_,n))
            print("media_d {}, sd_d {}, media_u {}, sd_u {}".format(media_lower,desviacion_lower,media_greater,desviacion_greater))
            print("tnf_d {}, tnf_u {}, sum {}".format(truncated_normal_funtion(media_lower, desviacion_lower),truncated_normal_funtion(media_greater, desviacion_greater),
                                                      truncated_normal_funtion(media_lower, desviacion_lower)+truncated_normal_funtion(media_greater, desviacion_greater)))
            
            if truncated_normal_funtion(media_lower, desviacion_lower) + truncated_normal_funtion(media_greater, desviacion_greater) >= epsilon:
                print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(
                    truncated_normal_funtion(media_lower, desviacion_lower) 
                    + ((p_n[n]-p_n_max[n] + alpha_n[n]*w_sigma*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)) - media_lower)*truncated_normal_funtion_dmu(media_lower, desviacion_lower) 
                    + ((alpha_n[n]*w_sigma)*np.sqrt((1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2)) - desviacion_lower)*truncated_normal_funtion_dsigma(media_lower, desviacion_lower) 
                    + truncated_normal_funtion(media_greater, desviacion_greater) 
                    + ((p_n[n]-p_n_max[n] + alpha_n[n]*w_sigma*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) ) - media_greater)*truncated_normal_funtion_dmu(media_greater, desviacion_greater) 
                    + ((alpha_n[n]*w_sigma)*np.sqrt((1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2)) - desviacion_greater)*truncated_normal_funtion_dsigma(media_greater, desviacion_greater)
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

def cutting_planes_problem_WCC_PW_beta():
    flag_print = False
    # Parameters
    N = 3
    d = np.array([120])

    c_p_n = np.array([15,30, 45]) # 25, 66
    c_alpha_n = np.array([30,40, 40]) # 50
    c_beta_n = np.array([30,30, 20]) # 50
    p_n_max = np.array([30,30,50])

    epsilon = 0.08
    inv_phi_eps = norm.ppf(1-epsilon)

    w_bar = 20
    w_sigma = 4

    # Define model
    model = gb.Model()

    # Variables
    p_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)            # Generation variable
    alpha_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="alpha_n", lb=1e-8) # Operational reserve variable
    beta_n = model.addMVar(N, vtype=GRB.CONTINUOUS, name="beta_n", lb=1e-8)   # Adversarial reserve variable


    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar == d, name='cons_demand') # Energy balance constraint
    cons_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_reserve')   # Operational reserve
    cons_r_adv = model.addConstr(sum(beta_n[n] for n in range(N)) == 1, name='cons_r_adv')   # Operational reserve adversarial


    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] + c_beta_n[n]*beta_n[n] for n in range(N)),GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Add cutting planes until the optimal solution satisfy the WCC constraint
    n_iterations = 50
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
        beta_n_star = dict()

        for n in range(N):
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X
            beta_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X

        w_critic = -20
        z_auxiliar = w_critic/w_sigma
        # Check if solution satisfy the WCC constraint
        if all(
            truncated_normal_funtion(p_n_star[n]-p_n_max[n] - alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar),
                np.sqrt((w_sigma*alpha_n_star[n])**2 *(1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2)))+
            truncated_normal_funtion(p_n_star[n]-p_n_max[n] - (alpha_n_star[n]+beta_n_star[n])*w_sigma*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)),
                np.sqrt((w_sigma*(alpha_n_star[n]+beta_n_star[n]))**2 *(1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2))) 
            <= epsilon for n in range(N)):
            break

        # Add a cutting plane to the model 
        for n in range(N):
            media_lower = p_n_star[n]-p_n_max[n] - alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)
            desviacion_lower = np.sqrt((w_sigma*alpha_n_star[n])**2 *(1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2))

            media_greater = p_n_star[n]-p_n_max[n] + (alpha_n_star[n]+beta_n_star[n])*w_sigma*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))
            desviacion_greater = np.sqrt(((w_sigma*(alpha_n_star[n]+beta_n_star[n]))**2) *(1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2))
            print('ITERATION, %s, gen, %s' %(_,n))
            print("media_d {}, sd_d {}, media_u {}, sd_u {}".format(media_lower,desviacion_lower,media_greater,desviacion_greater))
            print("tnf_d {}, tnf_u {}, sum {}".format(truncated_normal_funtion(media_lower, desviacion_lower),truncated_normal_funtion(media_greater, desviacion_greater),
                                                      truncated_normal_funtion(media_lower, desviacion_lower)+truncated_normal_funtion(media_greater, desviacion_greater)))
            
            if truncated_normal_funtion(media_lower, desviacion_lower) + truncated_normal_funtion(media_greater, desviacion_greater) >= epsilon:
                print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(
                    truncated_normal_funtion(media_lower, desviacion_lower) 
                    + ((p_n[n]-p_n_max[n] + alpha_n[n]*w_sigma*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)) - media_lower)*truncated_normal_funtion_dmu(media_lower, desviacion_lower) 
                    + ((alpha_n[n]*w_sigma)*np.sqrt((1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2)) - desviacion_lower)*truncated_normal_funtion_dsigma(media_lower, desviacion_lower) 
                    + truncated_normal_funtion(media_greater, desviacion_greater) 
                    + ((p_n[n]-p_n_max[n] + (alpha_n[n]+beta_n[n])*w_sigma*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) ) - media_greater)*truncated_normal_funtion_dmu(media_greater, desviacion_greater) 
                    + (((alpha_n[n]+beta_n[n])*w_sigma)*np.sqrt((1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) + (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2)) - desviacion_greater)*truncated_normal_funtion_dsigma(media_greater, desviacion_greater)
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
    print("A solution beta_n is")
    print("{}: {}".format(beta_n.varName, beta_n.X))


#cutting_planes_problem_ED()
#cutting_planes_problem_WCC_ED()

#cutting_planes_problem_WCC_PW()
#cutting_planes_problem_WCC_PW_beta()

'''
Status: 2
The optimal value is 3190.0
A solution p_n is
['p_n[0]' 'p_n[1]' 'p_n[2]']: [30. 30. 40.]
A solution alpha_n is
['alpha_n[0]' 'alpha_n[1]' 'alpha_n[2]']: [0. 0. 1.]

Status: 2
The optimal value is 3186.9618363705254
A solution p_n is
['p_n[0]' 'p_n[1]' 'p_n[2]']: [30.06342398 30.06344052 39.8731355 ]
A solution alpha_n is
['alpha_n[0]' 'alpha_n[1]' 'alpha_n[2]']: [0.01838366 0.01834527 0.96327108]

Status: 2
The optimal value is 3189.318035570554
A solution p_n is
['p_n[0]' 'p_n[1]' 'p_n[2]']: [30.0399999  29.9654645  39.99453561]
A solution alpha_n is
['alpha_n[0]' 'alpha_n[1]' 'alpha_n[2]']: [1.0000000e-08 1.0000000e-08 9.9999998e-01]
'''

'''
mu_sigma = 1
w_sigma = 5
w_critic = -20
z_auxiliar = (w_critic- mu_sigma)/w_sigma

media_lower = mu_sigma + w_sigma*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)
desviacion_lower = np.sqrt((w_sigma)**2 *(1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) + (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2))

print("pdf {}, cdf {}".format(norm.pdf(z_auxiliar),norm.cdf(z_auxiliar)))
print("media {}, sd {}".format(media_lower,desviacion_lower))
#print("term 1 {}, term 2 {}, term 3 {}".format(1 , z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar),-(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2))
print(np.sqrt((w_sigma)**2 *(1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) + (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2)))
'''

'''
media_lower = p_n_star[n]-p_n_max[n] - alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)
desviacion_lower = np.sqrt((w_sigma*alpha_n_star[n])**2 *(1 + z_auxiliar*norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2))

media_greater = p_n_star[n]-p_n_max[n] + alpha_n_star[n]*w_sigma*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))
desviacion_greater = np.sqrt(((w_sigma*alpha_n_star[n])**2) *(1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2))
'''
value_crit = -4
mu_known = 2
sigma_known = 3

z_auxiliar = (value_crit- mu_known)/sigma_known
print('z_auxiliar',z_auxiliar)


#parameters
mu = 2
sigma = 3.002
test_conditional_negative = False# True
test_conditional_positive = True


if test_conditional_negative == True:
    frac_auxiliar = norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)
    print("fraction_auxiliar",frac_auxiliar)

    tnf = truncated_normal_funtion(mu,sigma)
    tnf_taylor = truncated_normal_funtion(mu_known,sigma_known) - (mu-mu_known)*truncated_normal_funtion_dmu(mu_known,sigma_known) - (sigma - sigma_known)*truncated_normal_funtion_dsigma(mu_known,sigma_known)
    print("tnf {}, tnf_taylor {}".format(tnf, tnf_taylor))

    media_lower = mu + sigma*frac_auxiliar
    print("term1 {}, term2 {}, term3 {}, sum {}".format(1,z_auxiliar*(frac_auxiliar), -(frac_auxiliar)**2, (1+z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2)))
    sd_lower = (sigma)*np.sqrt((1-z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2))
    print("media_cond {}, sd_cond {}".format(media_lower, sd_lower))

    media_lower_known = mu_known + sigma_known*frac_auxiliar
    sd_lower_known = np.sqrt(((sigma_known)**2)*(1-z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2))
    print("media_cond_know {}, sd_cond_known {}".format(media_lower_known, sd_lower_known))

    tnf_cond = truncated_normal_funtion(media_lower,sd_lower)
    tnf_taylor_cond = truncated_normal_funtion(media_lower_known,sd_lower_known) 
    + (media_lower -media_lower_known)*truncated_normal_funtion_dmu(media_lower_known,sd_lower_known) 
    + (sd_lower- sd_lower_known)*(truncated_normal_funtion_dsigma(media_lower_known,sd_lower_known)+truncated_normal_funtion_dmu(media_lower_known,sd_lower_known))

    print("tnf_cond {}, tnf_taylor_cond {}".format(tnf_cond, tnf_taylor_cond))

if test_conditional_positive == True:
    frac_auxiliar = norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))
    print("fraction_auxiliar",frac_auxiliar)

    tnf = truncated_normal_funtion(mu,sigma)
    tnf_taylor = truncated_normal_funtion(mu_known,sigma_known) - (mu-mu_known)*truncated_normal_funtion_dmu(mu_known,sigma_known) - (sigma - sigma_known)*truncated_normal_funtion_dsigma(mu_known,sigma_known)
    print("tnf {}, tnf_taylor {}".format(tnf, tnf_taylor))

    media_lower = mu + sigma*frac_auxiliar
    print("term1 {}, term2 {}, term3 {}, sum {}".format(1,z_auxiliar*(frac_auxiliar),- (frac_auxiliar)**2, (1+z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2)))
    sd_lower = (sigma)*np.sqrt((1-z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2))
    print("media_cond {}, sd_cond {}".format(media_lower, sd_lower))

    media_lower_known = mu_known + sigma_known*frac_auxiliar
    sd_lower_known = np.sqrt(((sigma_known)**2)*(1-z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2))
    print("media_cond_know {}, sd_cond_known {}".format(media_lower_known, sd_lower_known))

    tnf_cond = truncated_normal_funtion(media_lower,sd_lower)
    tnf_taylor_cond = truncated_normal_funtion(media_lower_known,sd_lower_known) 
    + (media_lower -media_lower_known)*truncated_normal_funtion_dmu(media_lower_known,sd_lower_known) 
    + (sd_lower- sd_lower_known)*(truncated_normal_funtion_dsigma(media_lower_known,sd_lower_known)+truncated_normal_funtion_dmu(media_lower_known,sd_lower_known))

    print("tnf_cond {}, tnf_taylor_cond {}".format(tnf_cond, tnf_taylor_cond))


#truncated_normal_funtion_dmu(mu,sigma) + truncated_normal_funtion_dsigma(mu,sigma)
