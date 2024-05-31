import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm
import pandas as pd
import random

# Test variables
aux_test = False
test_conditional_negative = False
test_conditional_positive = False

# Code flags
power_system = True
flag_solve = True
flag_test = False
flag_test_Network = False

# System parameters
if power_system == False:
    node_set_index = [0] #[0]
    gen_set_index = [0,1,2] #[0,1,2]
    lines_node_index = []
    line_reactance = []
    line_f_max = []
    node_demand = [155] #[125]
    gen_max = [40, 45, 55] #[40, 45, 55]
    gen_cmg = [25, 30, 35] #[20, 30, 35]
    gen_cmg_b = [0.001, 0.003, 0.005] #[20, 30, 35]
    gen_calpha = [55, 40, 90] #[65, 40, 90]
    gen_cbeta = [100, 55, 120] #[210,65,90]
    gen_node = [0,0,0] #[0,0,0]
    wind_set_index = [0] #[0]
    wind_node = [0] #[0]
    wind_factor = [1] #[1]
else:
    # Lines data
    df_2 = pd.read_csv('gridDetails.csv') #'/Users/tftapia/Codes/ChanceConstraints/ChanceConstraints-1/gridDetails.csv'
    df_2.keys()
    zonas = np.unique(df_2[['From Zone', 'To Zone']].values)

    lines_node_index = []
    line_reactance = []
    line_f_max = []
    node_demand = []

    wind_set_index = []
    wind_node = []
    wind_factor = []

    for ind in df_2.index:
        index_barra_in = np.where(zonas == df_2['From Zone'][ind])[0][0]
        index_barra_out = np.where(zonas == df_2['To Zone'][ind])[0][0]
        lines_node_index.append((index_barra_in,index_barra_out))
        line_reactance.append(df_2['Reactance (per unit)'][ind])
        line_f_max.append(df_2['Capacity (MW)'][ind])

    df = pd.read_csv('generator_data.csv') # '/Users/tftapia/Codes/ChanceConstraints/ChanceConstraints-1/generator_data.csv'

    node_set_index = []
    gen_set_index = []
    gen_max = []
    gen_cmg = []
    gen_cmg_b = []
    gen_calpha = [] ## Arreglar
    gen_cbeta = [] ## Arreglar
    gen_node = []

    # Node data
    for ind in range(len(zonas)):
        node_set_index.append(ind)
        df_3 = pd.read_csv(zonas[ind]+'.csv') # '/Users/tftapia/Codes/ChanceConstraints/ChanceConstraints-1/'+zonas[ind]+'.csv'
        demanda_promedio = df_3[zonas[ind]].mean() #the last number is the forecast for the 8 zones system
        node_demand.append(demanda_promedio)

    # Generators data
    for ind in df.index:
        barra_string = df['Zone Location'][ind]
        index_barra = np.where(zonas == barra_string)[0][0]
        costo_marginal = df['Dispatch Cost Coefficient a ($/MWh)'][ind]
        costo_marginal_2 = df['Dispatch Cost Coefficient b ($/MW^2h)'][ind]
        capacidad = df['Capacity (MW)'][ind]*0.69

        costo_marginal_rr = df['Regular Reserve Cost Coefficient ($/MWh)'][ind]
        costo_marginal_er = df['Extreme Reserve Cost Coefficient ($/MWh)'][ind]

        gen_set_index.append(ind)
        gen_node.append(index_barra)
        gen_cmg.append(costo_marginal)
        gen_cmg_b.append(costo_marginal_2)
        gen_calpha.append(costo_marginal_rr)
        gen_cbeta.append(costo_marginal_er)
        gen_max.append(capacidad)

    for ind in range(len(zonas)): #create a wind generator in each zone
        wind_set_index.append(ind)
        wind_node.append(ind)
        wind_factor.append(node_demand[ind]/sum(node_demand))
        #wind_factor.append(1/len(zonas))

    '''
    cap_list = []
    for ind in range(len(zonas)):
        aux_aux = 0
        for ind_g in range(len(gen_set_index)):
            if gen_node[ind_g] == ind:
                aux_aux += gen_max[ind_g]
        cap_list.append(aux_aux)
    print(zonas)
    print(cap_list)
    print(sum(cap_list))

    #['CT' 'ME' 'NEMASSBOST' 'NH' 'RI' 'SEMASS' 'VT' 'WCMASS']
    #[5694.599999999999, 5321.400000000001, 0, 2247.6000000000004, 5026.1, 2962.7000000000003, 620.2, 1227.7]
    #23100.300000000003
    '''
# Set gen parameters in matrices
node_gens = []
for n in node_set_index:
    aux_list = []
    for g in gen_set_index:
        if gen_node[g] == n:
            aux_list.append(g)
    node_gens.append(aux_list)

node_wind = []
for n in node_set_index:
    aux_list = []
    for g in wind_set_index:
        if wind_node[g] == n:
            aux_list.append(g)
    node_wind.append(aux_list)

# Set line parameters in matrices
k_nodes = len(node_set_index)
line_susceptance_matrix = np.zeros((k_nodes,k_nodes))
line_f_max_matrix = np.zeros((k_nodes,k_nodes))
for (i,j) in lines_node_index:
    aux_index = lines_node_index.index((i,j))
    line_susceptance_matrix[i,j] = -1/line_reactance[aux_index]
    line_susceptance_matrix[j,i] = -1/line_reactance[aux_index]
    line_f_max_matrix[i,j] = line_f_max[aux_index]
    line_f_max_matrix[j,i] = line_f_max[aux_index]

# Set of lines in/out in each node
lines_node_in = []
lines_node_out = []
for n in node_set_index:
    aux_list_in = []
    aux_list_out = []
    for [i,j] in lines_node_index:
        if i == n:
            aux_list_in.append([i,j])
        elif j == n:
            aux_list_out.append([i,j])
    lines_node_in.append(aux_list_in)
    lines_node_out.append(aux_list_out)

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

######## functions ####
# pdf (density)
# norm.pdf(x)
# cummulative cdf   
# norm.cdf(-1.96)
# inverse cdf
# norm.ppf(norm.cdf(1.96))

# MODELS

def cutting_planes_problem_ED_network(power_system,node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg, gen_cmg_b,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind):
    flag_print = False
    #System parametrs
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)

    if power_system:
        w_bar = 4000
        w_sigma = 800 
    else:
        w_bar = 40 #20 #50
        w_sigma = 10 #5 #12.5

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(lines_node_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="alpha_n", lb=0) # Operational reserve variable

    # Set objective function
    obj_value = model.setObjective(sum(p_n[n]*gen_cmg[n] + p_n[n]**2*gen_cmg_b[n] + gen_calpha[n]*alpha_n[n] for n in gen_set_index) + sum(s_i[i]*100000 for i in node_set_index), GRB.MINIMIZE)

    # Add constraints
    #for i in node_set_index:
    #    print("wind {}, demand {}".format(sum(w_bar*wind_factor[n] for n in node_wind[i]), node_demand[i]))
    const_demand = model.addConstrs((sum(p_n[n] for n in node_gens[i]) + sum(w_bar*wind_factor[n] for n in node_wind[i])
                                 - sum(f_ij[i,j] for (i,j) in lines_node_in[i])
                                 + sum(f_ij[i,j] for (i,j) in lines_node_out[i])
                                 + s_i[i]  == node_demand[i] for i in node_set_index), name='const_demand') # Demand constraints
    const_pmax = model.addConstrs((p_n[n] <= gen_max[n] for n in gen_set_index), name='const_pmax')
    const_f_t = model.addConstrs((line_susceptance_matrix[i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in lines_node_index), name='const_f_t')
    const_fmax = model.addConstrs((f_ij[i,j] <= line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmax')
    const_fmin = model.addConstrs((f_ij[i,j] >= -line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmin')
    const_t_0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in gen_set_index) == 1, name='const_op_reserve') # Operational reserve constraint
  
    # Add cutting planes until the optimal solution satisfy the WCC constraint
    n_iterations = 100
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

        for n in gen_set_index:
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X

        # Check if solution satisfy the WCC constraint
        if all(p_n_star[n] + alpha_n_star[n]*inv_phi_eps*w_sigma <=gen_max[n] for n in gen_set_index):
            break

        # Add a cutting plane to the model 
        for n in gen_set_index:
            if p_n_star[n] + alpha_n_star[n]*inv_phi_eps*w_sigma >=gen_max[n]:
                print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(p_n[n] + alpha_n[n]*inv_phi_eps*w_sigma <=gen_max[n])

    # Print results
    obj = model.getObjective()
    ## Primal solution
    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    for v in p_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution alpha_n is")
    for v in alpha_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution f_ij is")
    for v in f_ij.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution s_i is")
    for v in s_i.values():
        print("{}: {}".format(v.varName, v.X))
    # Dual results
    for c in const_demand.values():
        print(c.constrName, c.Pi)
    print(const_op_reserve.constrName, const_op_reserve.Pi)

    o_opt = obj.getValue()
    p_opt = []
    a_opt = []
    p_price = []
    a_price = const_op_reserve.Pi

    for v in p_n.values():
        p_opt.append(round(v.X,4))
    for v in alpha_n.values():
        a_opt.append(round(v.X,4))
    for c in const_demand.values():
        p_price .append(round(c.Pi,4))


    return o_opt, p_opt, a_opt, p_price, a_price
    
def cutting_planes_problem_WCC_ED_network(power_system,node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg, gen_cmg_b,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind):
    flag_print = False
    #System parameters
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)


    if power_system:
        w_bar = 600
        w_sigma = 40
    else:
        w_bar = 20
        w_sigma = 5

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(lines_node_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="alpha_n", lb=1e-8) # Operational reserve variable

    # Set objective function
    obj_value = model.setObjective( sum(p_n[n]*gen_cmg[n] + gen_calpha[n]*alpha_n[n] for n in gen_set_index) + sum(s_i[i]*1000000 for i in node_set_index), GRB.MINIMIZE)

    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in node_gens[i]) + sum(w_bar*wind_factor[n] for n in node_wind[i])
                                 - sum(f_ij[i,j] for (i,j) in lines_node_in[i])
                                 + sum(f_ij[i,j] for (i,j) in lines_node_out[i])
                                 + s_i[i]  == node_demand[i] for i in node_set_index), name='const_demand') # Demand constraints
    const_pmax = model.addConstrs((p_n[n] <= gen_max[n] for n in gen_set_index), name='const_pmax')
    const_f_t = model.addConstrs( (line_susceptance_matrix[i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in lines_node_index), name='const_f_t')
    const_fmax = model.addConstrs( (f_ij[i,j] <= line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmax')
    const_fmin = model.addConstrs( (f_ij[i,j] >= -line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmin')
    const_t_0 = model.addConstr(t_i[0] == 0)
    
    cons_op_reserve = model.addConstr(sum(alpha_n[n] for n in gen_set_index) == 1, name='cons_op_reserve') # Operational reserve constraint

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

        for n in gen_set_index:
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X

        # Check if solution satisfy the WCC constraint
        if all(truncated_normal_funtion(p_n_star[n]-gen_max[n], alpha_n_star[n]*w_sigma) <= epsilon for n in gen_set_index):
            for n in gen_set_index:
                media = p_n_star[n]-gen_max[n]
                desviacion = alpha_n_star[n]*w_sigma
                print('ITERATION, %s, gen, %s' %(_,n))
                print("media {}, sd {}".format(media,desviacion))
                print("tnf {}, epsilon {}".format(truncated_normal_funtion(media, desviacion),epsilon))
            break

        # Add a cutting plane to the model 
        for n in gen_set_index:
            media = p_n_star[n]-gen_max[n]
            desviacion = alpha_n_star[n]*w_sigma
            print('ITERATION, %s, gen, %s' %(_,n))
            print("media {}, sd {}".format(media,desviacion))
            print("tnf {}, epsilon {}".format(truncated_normal_funtion(media, desviacion),epsilon))
           
            if truncated_normal_funtion(media, desviacion) >= epsilon:
                print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(truncated_normal_funtion(media, desviacion) + ((p_n[n]-gen_max[n]) - media)*truncated_normal_funtion_dmu(media, desviacion) + ((alpha_n[n]*w_sigma) - desviacion)*truncated_normal_funtion_dsigma(media, desviacion) <= epsilon)
        
        '''
        term_1 = truncated_normal_funtion(a,b)
        term_2 = truncated_normal_funtion_dmu(a,b)*(mu-a)
        term_3 = truncated_normal_funtion_dsigma(a,b)*(sigma-b)
        t_approx = term_1 + term_2 + term_3
        '''

    # Print results
    obj = model.getObjective()
    ## Primal solution
    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    for v in p_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution alpha_n is")
    for v in alpha_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution f_ij is")
    for v in f_ij.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution s_i is")
    for v in s_i.values():
        print("{}: {}".format(v.varName, v.X))

def cutting_planes_problem_WCC_PW():
    flag_print = False
    # Parameters
    N = 3
    d = np.array([120])

    c_p_n = np.array([40, 40, 45]) # 25, 66
    c_alpha_n = np.array([50, 50, 60]) # 50
    c_beta_n = np.array([50, 50, 100]) # 50
    p_n_max = np.array([40,40,30])

    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
   
    w_bar = 600
    w_sigma = 40

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

        for n in range(N):
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X

        w_critic = -10
        z_auxiliar = w_critic/w_sigma

        # Check if solution satisfy the WCC constraint
        if all(
            truncated_normal_funtion(p_n_star[n]-p_n_max[n] + alpha_n_star[n]*w_sigma*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)),
                                    (w_sigma*alpha_n_star[n])*np.sqrt(1 - z_auxiliar*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2))
                +
            truncated_normal_funtion(p_n_star[n]-p_n_max[n] + alpha_n_star[n]*w_sigma*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))),
                                    (w_sigma*alpha_n_star[n])*np.sqrt(1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2))
            <= epsilon for n in range(N)):
            break

        # Add a cutting plane to the model 
        for n in range(N):
            media_lower = p_n_star[n]-p_n_max[n] - alpha_n_star[n]*w_sigma*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))
            desviacion_lower = (w_sigma*alpha_n_star[n])*np.sqrt(1 - z_auxiliar*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2)

            media_greater = p_n_star[n]-p_n_max[n] + alpha_n_star[n]*w_sigma*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))
            desviacion_greater = np.sqrt(((w_sigma*alpha_n_star[n])**2) *(1 + z_auxiliar*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2))
            print('ITERATION, %s, gen, %s' %(_,n))
            print("media_d {}, sd_d {}, media_u {}, sd_u {}".format(media_lower,desviacion_lower,media_greater,desviacion_greater))
            print("tnf_d {}, tnf_u {}, sum {}".format(truncated_normal_funtion(media_lower, desviacion_lower),truncated_normal_funtion(media_greater, desviacion_greater),
                                                      truncated_normal_funtion(media_lower, desviacion_lower)+truncated_normal_funtion(media_greater, desviacion_greater)))
            
            if truncated_normal_funtion(media_lower, desviacion_lower) + truncated_normal_funtion(media_greater, desviacion_greater) >= epsilon:
                print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(
                    truncated_normal_funtion(media_lower, desviacion_lower) 
                    + ((p_n[n]-p_n_max[n] - alpha_n[n]*w_sigma*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))) - media_lower)*truncated_normal_funtion_dmu(media_lower, desviacion_lower) 
                    + ((alpha_n[n]*w_sigma)*np.sqrt(1 - z_auxiliar*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2) - desviacion_lower)*truncated_normal_funtion_dsigma(media_lower, desviacion_lower) 
                    + truncated_normal_funtion(media_greater, desviacion_greater) 
                    + ((p_n[n]-p_n_max[n] + alpha_n[n]*w_sigma*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))) ) - media_greater)*truncated_normal_funtion_dmu(media_greater, desviacion_greater) 
                    + ((alpha_n[n]*w_sigma)*np.sqrt(1 + z_auxiliar*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2) - desviacion_greater)*truncated_normal_funtion_dsigma(media_greater, desviacion_greater)
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

def cutting_planes_problem_WCC_PW_beta(power_system,node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg, gen_cmg_b,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind):
    flag_print = False
    #System parameters
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)

    if power_system:
        w_bar = 4000
        w_sigma = 800
        w_critic = 3291.043651139727

    else:
        w_bar = 40 # 20 #50
        w_sigma = 10 # 5 # 12.5
        w_critic = 12.5 # 17.5 # 32.5 

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(lines_node_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="alpha_n", lb=1e-8) # Operational reserve variable
    beta_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="beta_n", lb=1e-8) # Operational reserve variable adverse

    # Set objective function
    obj_value = model.setObjective( sum(p_n[n]*gen_cmg[n] + p_n[n]**2*gen_cmg_b[n] + gen_calpha[n]*alpha_n[n] + gen_cbeta[n]*beta_n[n] for n in gen_set_index) + sum(s_i[i]*1000000 for i in node_set_index), GRB.MINIMIZE)

    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in node_gens[i]) + sum(w_bar*wind_factor[n] for n in node_wind[i])
                                 - sum(f_ij[i,j] for (i,j) in lines_node_in[i])
                                 + sum(f_ij[i,j] for (i,j) in lines_node_out[i])
                                 + s_i[i]  == node_demand[i] for i in node_set_index), name='const_demand') # Demand constraints
    const_pmax = model.addConstrs((p_n[n] <= gen_max[n] for n in gen_set_index), name='const_pmax')
    const_f_t = model.addConstrs( (line_susceptance_matrix[i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in lines_node_index), name='const_f_t')
    const_fmax = model.addConstrs( (f_ij[i,j] <= line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmax')
    const_fmin = model.addConstrs( (f_ij[i,j] >= -line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmin')
    const_t_0 = model.addConstr(t_i[0] == 0)
    
    const_op_reserve = model.addConstr(sum(alpha_n[n] for n in gen_set_index) == 1, name='const_op_reserve') # Operational reserve constraint
    const_ad_reserve = model.addConstr(sum(beta_n[n] for n in gen_set_index) == 1, name='const_ad_reserve') # Adversarial reserve constraint
    #model.addConstrs(p_n[n] - gen_max[n] + alpha_n[n]*inv_phi_eps*w_sigma <= 0 for n in gen_set_index)     # Max generation limit constraint%

    # Constraints
    cons_pmax = model.addConstrs((p_n[n]<= gen_max[n] for n in gen_set_index), name='cons_pmax') 

    # Solve
    model.optimize()

    # Add cutting planes until the optimal solution satisfy the WCC constraint
    n_iterations = 25
    continue_flag = True
    for _ in range(n_iterations):
        print("ITERATION",_)
        #model.display()
        model.optimize()
        obj_value = model.getObjective()

        if flag_print == True:
            print('ITERATION, %s' % _)
            print("The optimal value is", obj_value.getValue())
            print("A solution p_n is")
            print("{}: {}".format(p_n.varName, p_n.X))
            print("A solution alpha_n is")
            print("{}: {}".format(alpha_n.varName, alpha_n.X))
            print("A solution beta_n is")
            print("{}: {}".format(beta_n.varName, beta_n.X))

        # Save variables
        p_n_star = dict()
        alpha_n_star = dict()
        beta_n_star = dict()

        for n in gen_set_index:
            p_n_star[n] = model.getVarByName('p_n[%d]'%(n)).X
            alpha_n_star[n] = model.getVarByName('alpha_n[%d]'%(n)).X
            beta_n_star[n] = model.getVarByName('beta_n[%d]'%(n)).X
  
        z_auxiliar = w_critic/w_sigma
        # Check if solution satisfy the WCC constraint
        if all(
            truncated_normal_funtion(p_n_star[n]-gen_set_index[n] + (alpha_n_star[n])*w_sigma*(norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar)+1e-7)),
                                    (w_sigma*(alpha_n_star[n]))*np.sqrt(1 - z_auxiliar*(norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar)+1e-7)) - (norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar)+1e-7))**2))
                +
            truncated_normal_funtion(p_n_star[n]-gen_set_index[n] + (alpha_n_star[n]+beta_n_star[n])*w_sigma*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)+1e-7)),
                                    (w_sigma*(alpha_n_star[n]+beta_n_star[n]))*np.sqrt(1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)+1e-7) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)+1e-7))**2))
            <= epsilon for n in gen_set_index):
            break

        # Add a cutting plane to the model 
        for n in gen_set_index:
            media_lower = p_n_star[n]-gen_set_index[n] - (alpha_n_star[n])*w_sigma*(norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar)))
            desviacion_lower = (w_sigma*(alpha_n_star[n]))*np.sqrt(1 - z_auxiliar*(norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar))) - (norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar)))**2)

            media_greater = p_n_star[n]-gen_set_index[n] + (alpha_n_star[n]+beta_n_star[n])*w_sigma*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))
            desviacion_greater = (w_sigma*(alpha_n_star[n]+beta_n_star[n]))*np.sqrt(1 + z_auxiliar*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2)
            #print('ITERATION, %s, gen, %s, z_auxiliar, %s ' %(_,n,z_auxiliar))
            #print("media_d {}, sd_d {}, media_u {}, sd_u {}".format(media_lower,desviacion_lower,media_greater,desviacion_greater))
            #print("tnf_d {}, tnf_u {}, sum {}".format(truncated_normal_funtion(media_lower, desviacion_lower),truncated_normal_funtion(media_greater, desviacion_greater),
            #                                          truncated_normal_funtion(media_lower, desviacion_lower)+truncated_normal_funtion(media_greater, desviacion_greater)))
            #print("tnf_d_dmu {}, tnf_d_dsigma {}, tnf_d_dmu {}, tnf_u_dmsigma {}".format(truncated_normal_funtion_dmu(media_lower, desviacion_lower), truncated_normal_funtion_dsigma(media_lower, desviacion_lower) ,truncated_normal_funtion_dmu(media_greater, desviacion_greater) ,truncated_normal_funtion_dsigma(media_greater, desviacion_greater)))
            #print(norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar)))
            #print("tnf_l {} + tnf_u {} < {}".format(truncated_normal_funtion(media_lower, desviacion_lower),truncated_normal_funtion(media_greater, desviacion_greater),epsilon))
            if truncated_normal_funtion(media_lower, desviacion_lower) + truncated_normal_funtion(media_greater, desviacion_greater) >= epsilon:
                #print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(
                    truncated_normal_funtion(media_lower, desviacion_lower) 
                    + ((p_n[n]-gen_max[n] - (alpha_n[n])*w_sigma*(norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar)))) - media_lower)*truncated_normal_funtion_dmu(media_lower, desviacion_lower) 
                    + (((alpha_n[n])*w_sigma)*np.sqrt(1 - z_auxiliar*(norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar)+1e-8)) - (norm.pdf(z_auxiliar)/(norm.cdf(z_auxiliar)+1e-8))**2) - desviacion_lower)*truncated_normal_funtion_dsigma(media_lower, desviacion_lower) 
                    + truncated_normal_funtion(media_greater, desviacion_greater)
                    + ((p_n[n]-gen_max[n] + (alpha_n[n]+beta_n[n])*w_sigma*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))) ) - media_greater)*truncated_normal_funtion_dmu(media_greater, desviacion_greater)
                    + (((alpha_n[n]+beta_n[n])*w_sigma)*np.sqrt(1 + z_auxiliar*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)+1e-8)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)+1e-8))**2) - desviacion_greater)*truncated_normal_funtion_dsigma(media_greater, desviacion_greater)
                    <= epsilon)
        
    # Print results
    obj = model.getObjective()
    ## Primal solution
    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    for v in p_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution alpha_n is")
    for v in alpha_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution beta_n is")
    for v in beta_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution f_ij is")
    for v in f_ij.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution s_i is")
    for v in s_i.values():
        print("{}: {}".format(v.varName, v.X))
    # Dual results
    for c in const_demand.values():
        print(c.constrName, c.Pi)
    print(const_op_reserve.constrName, const_op_reserve.Pi)
    print(const_ad_reserve.constrName, const_ad_reserve.Pi)


    o_opt = obj.getValue()
    p_opt = []
    a_opt = []
    b_opt = []
    p_price = []
    a_price = const_op_reserve.Pi
    b_price = const_ad_reserve.Pi
    for v in p_n.values():
        p_opt.append(round(v.X,4))
    for v in alpha_n.values():
        a_opt.append(round(v.X,4))
    for v in beta_n.values():
        b_opt.append(round(v.X,4))
    for c in const_demand.values():
        p_price .append(round(c.Pi,4))

    return o_opt, p_opt, a_opt, b_opt, p_price, a_price , b_price 

def EconomicDispatch_LDT_network(power_system,node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg, gen_cmg_b,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind):#
    flag_print = False
    #System parameters
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)

    if power_system:
        w_bar = 4000
        w_sigma = 800
    else:
        w_bar = 40 #20 # 50 
        w_sigma = 10 #5 # 12.5

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(lines_node_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS, name='s_i')

    alpha_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="alpha_n", lb=1e-8) # Operational reserve variable
    beta_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="beta_n", lb=1e-8)   # Adversarial reserve variable

    omega_star = model.addVar(vtype=GRB.CONTINUOUS, name="omega_star", lb=-GRB.INFINITY) # Optimal critical value from the extreme event region
    lambda_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="lambda_n", lb=-GRB.INFINITY)     # Dual variable of the constrainst to obtain "omega_star"

    aux_omega_a = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_omega_a", lb=-GRB.INFINITY)   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_omega_b = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_omega_b", lb=-GRB.INFINITY)   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_lambda_a = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_lambda_a", lb=-GRB.INFINITY) # Auxilar value to represent the positive value of "lambda_n*(alpha_n +beta_n)" in the SOC
    aux_lambda_b = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_lambda_b", lb=-GRB.INFINITY) # Auxilar value to represent the negative value of "lambda_n*(alpha_n +beta_n)" in the SOC
    
    # Set objective function
    obj_value = model.setObjective( sum(p_n[n]*gen_cmg[n] + p_n[n]**2*gen_cmg_b[n] + gen_calpha[n]*alpha_n[n] + gen_cbeta[n]*beta_n[n] for n in gen_set_index) + sum(s_i[i]*100000 for i in node_set_index), GRB.MINIMIZE)

    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in node_gens[i]) + sum(w_bar*wind_factor[n] for n in node_wind[i])
                                 - sum(f_ij[i,j] for (i,j) in lines_node_in[i])
                                 + sum(f_ij[i,j] for (i,j) in lines_node_out[i])
                                 + s_i[i]  == node_demand[i] for i in node_set_index), name='const_demand') # Demand constraints
    const_pmax = model.addConstrs((p_n[n] <= gen_max[n] for n in gen_set_index), name='const_pmax')
    const_f_t = model.addConstrs( (line_susceptance_matrix[i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in lines_node_index), name='const_f_t')
    const_fmax = model.addConstrs( (f_ij[i,j] <= line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmax')
    const_fmin = model.addConstrs( (f_ij[i,j] >= -line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmin')
    const_t_0 = model.addConstr(t_i[0] == 0)
    
    cons_op_reserve = model.addConstr(sum(alpha_n[n] for n in gen_set_index) == 1, name='cons_op_reserve') # Operational reserve constraint
    cons_ad_reserve = model.addConstr(sum(beta_n[n] for n in gen_set_index) == 1, name='cons_ad_reserve')  # Adversarial reserve constraint
    model.addConstrs(p_n[n] - gen_max[n] + alpha_n[n]*inv_phi_eps*w_sigma <= 0 for n in gen_set_index)     # Max generation limit constraint%
    
    model.addConstrs(aux_omega_a[n] == omega_star*alpha_n[n] for n in gen_set_index)
    model.addConstrs(aux_omega_b[n] == omega_star*beta_n[n] for n in gen_set_index)
    model.addConstrs(aux_lambda_a[n] == lambda_n[n]*alpha_n[n] for n in gen_set_index)
    model.addConstrs(aux_lambda_b[n] == lambda_n[n]*beta_n[n] for n in gen_set_index)

    ## LDT Constraint
    model.addConstr(-w_sigma**(-0.5)*omega_star - inv_phi_ext <= 0)
    model.addConstrs(w_sigma**(-1)*omega_star - aux_lambda_a[n] - aux_lambda_b[n] == 0 for n in gen_set_index)
    model.addConstrs(-gen_max[n] + p_n[n] + aux_omega_a[n] + aux_omega_b[n] == 0 for n in gen_set_index)

    ## Allow QCP dual 
    model.Params.QCPDual = 1
    model.Params.NonConvex = 2
    model.setParam('MIPGap', 0.01)
    model.setParam('Timelimit', 600)
    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    ## Primal solution
    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    for v in p_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution alpha_n is")
    for v in alpha_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution beta_n is")
    for v in beta_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution s_i is")    
    for v in s_i.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution omega_star is")
    print("{}: {}".format(omega_star.varName, omega_star.X))
    print("A solution lambda_n is")
    for v in lambda_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution f_ij is")
    for v in f_ij.values():
        print("{}: {}".format(v.varName, v.X))

    o_opt = obj.getValue()
    p_opt = []
    a_opt = []
    b_opt = []
    w_opt = omega_star.X
    l_opt = []

    for v in p_n.values():
        p_opt.append(round(v.X,4))
    for v in alpha_n.values():
        a_opt.append(round(v.X,4))
    for v in beta_n.values():
        b_opt.append(round(v.X,4))
    for v in lambda_n.values():
        l_opt.append(round(v.X,4))

    return o_opt, p_opt, a_opt, b_opt, w_opt, l_opt 

    #print("A solution aux_omega_a is")
    #print("{}: {}".format(aux_omega_a.varName, aux_omega_a.X ))
    #print("A solution aux_omega_b is")
    #print("{}: {}".format(aux_omega_b.varName, aux_omega_b.X ))
    
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

# SOLVE MODELS
if flag_solve:
    print("\nCC Solve")
    o_opt1, p_opt1, a_opt1, p_price1, a_price1= cutting_planes_problem_ED_network(power_system,node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg, gen_cmg_b,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind)
    b_opt1 = np.zeros(len(gen_set_index))
    print("\nWCC Solve")
    o_opt2, p_opt2, a_opt2, b_opt2, p_price2, a_price2, b_price2 = cutting_planes_problem_WCC_PW_beta(power_system,node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg, gen_cmg_b,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind)
    print("\nLDT Solve")
    o_opt3, p_opt3, a_opt3, b_opt3 , w_opt3, l_opt3= EconomicDispatch_LDT_network(power_system,node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg, gen_cmg_b,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind)
    print(w_opt3, l_opt3)

    def EconomicDispatch_LDT_network_price(power_system,node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg, gen_cmg_b,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind, w_opt, l_opt):
        flag_print = False
        #System parameters
        epsilon = 0.05
        inv_phi_eps = norm.ppf(1-epsilon)
        epsilon_ext = 0.05
        inv_phi_ext = norm.ppf(epsilon_ext)

        if power_system:
            w_bar = 4000
            w_sigma = 800
        else:
            w_bar = 40 #20
            w_sigma = 10 #5

        # Create a new model
        model = gb.Model()

        # Create variables
        p_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name='p_n')
        t_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
        f_ij = model.addVars(lines_node_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
        s_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS, name='s_i')
        alpha_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="alpha_n", lb=1e-8) # Operational reserve variable
        beta_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="beta_n", lb=1e-8)   # Adversarial reserve variable

        aux_omega_a = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_omega_a", lb=-GRB.INFINITY)   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
        aux_omega_b = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_omega_b", lb=-GRB.INFINITY)   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
        aux_lambda_a = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_lambda_a", lb=-GRB.INFINITY) # Auxilar value to represent the positive value of "lambda_n*(alpha_n +beta_n)" in the SOC
        aux_lambda_b = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_lambda_b", lb=-GRB.INFINITY) # Auxilar value to represent the negative value of "lambda_n*(alpha_n +beta_n)" in the SOC
        
        # Set objective function
        obj_value = model.setObjective( sum(p_n[n]*gen_cmg[n] + p_n[n]**2*gen_cmg_b[n] + gen_calpha[n]*alpha_n[n] + gen_cbeta[n]*beta_n[n] for n in gen_set_index) + sum(s_i[i]*1000000 for i in node_set_index), GRB.MINIMIZE)

        # Add constraints
        const_demand = model.addConstrs((sum(p_n[n] for n in node_gens[i]) + sum(w_bar*wind_factor[n] for n in node_wind[i])
                                    - sum(f_ij[i,j] for (i,j) in lines_node_in[i])
                                    + sum(f_ij[i,j] for (i,j) in lines_node_out[i])
                                    + s_i[i]  == node_demand[i] for i in node_set_index), name='const_demand') # Demand constraints
        const_pmax = model.addConstrs((p_n[n] <= gen_max[n] for n in gen_set_index), name='const_pmax')
        const_f_t = model.addConstrs( (line_susceptance_matrix[i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in lines_node_index), name='const_f_t')
        const_fmax = model.addConstrs( (f_ij[i,j] <= line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmax')
        const_fmin = model.addConstrs( (f_ij[i,j] >= -line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmin')
        const_t_0 = model.addConstr(t_i[0] == 0)
        
        const_op_reserve = model.addConstr(sum(alpha_n[n] for n in gen_set_index) == 1, name='const_op_reserve') # Operational reserve constraint
        const_ad_reserve = model.addConstr(sum(beta_n[n] for n in gen_set_index) == 1, name='const_ad_reserve')  # Adversarial reserve constraint
        model.addConstrs(p_n[n] - gen_max[n] + alpha_n[n]*inv_phi_eps*w_sigma <= 0 for n in gen_set_index)     # Max generation limit constraint%
        
        model.addConstrs(aux_lambda_a[n] == l_opt[n]*alpha_n[n] for n in gen_set_index)
        model.addConstrs(aux_lambda_b[n] == l_opt[n]*beta_n[n] for n in gen_set_index)
        model.addConstrs(aux_omega_a[n] == w_opt*alpha_n[n] for n in gen_set_index)
        model.addConstrs(aux_omega_b[n] == w_opt*beta_n[n] for n in gen_set_index)

        ## LDT Constrain
        model.addConstrs(-gen_max[n] + p_n[n] + aux_omega_a[n] + aux_omega_b[n] <= 0.0001 for n in gen_set_index) # relaxed to obtain the prices

        ## Allow QCP dual 
        model.setParam('MIPGap', 0.01)
        model.setParam('Timelimit', 600)
        # Solve
        model.optimize()

        # Print results
        obj = model.getObjective()
        ## Primal solution
        print("\nThe optimal value is", obj.getValue())
        print("A solution p_n is")
        for v in p_n.values():
            print("{}: {}".format(v.varName, v.X))
        print("A solution alpha_n is")
        for v in alpha_n.values():
            print("{}: {}".format(v.varName, v.X))
        print("A solution beta_n is")
        for v in beta_n.values():
            print("{}: {}".format(v.varName, v.X))
        print("A solution s_i is")    
        for v in s_i.values():
            print("{}: {}".format(v.varName, v.X))
        print("A solution f_ij is")
        for v in f_ij.values():
            print("{}: {}".format(v.varName, v.X))
        for c in const_demand.values():
            print(c.constrName, c.Pi)
        print(const_op_reserve.constrName, const_op_reserve.Pi)
        print(const_ad_reserve.constrName, const_ad_reserve.Pi)

        o_opt = obj.getValue()
        p_opt = []
        a_opt = []
        b_opt = []
        p_price = []
        a_price = const_op_reserve.Pi
        b_price = const_ad_reserve.Pi

        for v in p_n.values():
            p_opt.append(round(v.X,4))
        for v in alpha_n.values():
            a_opt.append(round(v.X,4))
        for v in beta_n.values():
            b_opt.append(round(v.X,4))
        for c in const_demand.values():
            p_price.append(round(c.Pi,4))

        return o_opt, p_opt, a_opt, b_opt, p_price, a_price , b_price 


        #for v in p_n.values():
        #    print("{}: {}".format(v.varName, v.X))
        #for c in cons_reserve.values():
        #    print("{}: {}".format(c.constrName, c.Pi))  # .QCPi is used for quadratic constraints

        return o_opt, p_opt, a_opt, b_opt, w_opt, l_opt 

    print("\nLDT Solve price")
    o_opt3p, p_opt3p, a_opt3p, b_opt3p, p_price3, a_price3, b_price3 = EconomicDispatch_LDT_network_price(power_system,node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg, gen_cmg_b,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind, w_opt3, l_opt3)

    # PRINT SOLUTIONS
    print("\nCC")
    print(o_opt1)
    #print(p_opt1)
    #print(a_opt1)
    #print(b_opt1)
    print(p_price1, a_price1)

    print("\nWCC")
    print(o_opt2)
    #print(p_opt2)
    #print(a_opt2)
    #print(b_opt2)
    print(p_price2, a_price2, b_price2)

    print("\nLDT")
    print(o_opt3, o_opt3p)
    #print(w_opt3, l_opt3)
    #print(p_opt3, p_opt3p)
    #print(a_opt3, a_opt3p)
    #print(b_opt3, b_opt3p)
    print(p_price3, a_price3, b_price3)

    if not(power_system):
        list_revenue_p1= []
        list_cost_p1= []
        list_profit_p1= []
        list_revenue_a1= []
        list_cost_a1= []
        list_profit_a1= []
        list_revenue_p2= []
        list_cost_p2= []
        list_profit_p2= []
        list_revenue_a2= []
        list_cost_a2= []
        list_profit_a2= []
        list_revenue_b2= []
        list_cost_b2= []
        list_profit_b2= []

        list_revenue_p3= []
        list_cost_p3= []
        list_profit_p3= []
        list_revenue_a3= []
        list_cost_a3= []
        list_profit_a3= []
        list_revenue_b3= []
        list_cost_b3 = []
        list_profit_b3= []

        for gen in gen_set_index:
            list_revenue_p1.append(p_opt1[gen]*p_price1[0])
            list_cost_p1.append(p_opt1[gen]*gen_cmg[gen] + p_opt1[gen]**2*gen_cmg_b[gen])
            list_profit_p1.append(p_opt1[gen]*p_price1[0] - p_opt1[gen]*gen_cmg[gen] - p_opt1[gen]**2*gen_cmg_b[gen])
            list_revenue_a1.append(a_opt1[gen]*a_price1)
            list_cost_a1.append(a_opt1[gen]*gen_calpha[gen])
            list_profit_a1.append(a_opt1[gen]*a_price1 - a_opt1[gen]*gen_calpha[gen])

            list_revenue_p2.append(p_opt2[gen]*p_price2[0])
            list_cost_p2.append(p_opt2[gen]*gen_cmg[gen] + p_opt2[gen]**2*gen_cmg_b[gen])
            list_profit_p2.append(p_opt2[gen]*p_price2[0] - p_opt2[gen]*gen_cmg[gen] - p_opt2[gen]**2*gen_cmg_b[gen])
            list_revenue_a2.append(a_opt2[gen]*a_price2)
            list_cost_a2.append(a_opt2[gen]*gen_calpha[gen])
            list_profit_a2.append(a_opt2[gen]*b_price2 - a_opt2[gen]*gen_calpha[gen])
            list_revenue_b2.append(b_opt2[gen]*b_price2)
            list_cost_b2.append(b_opt2[gen]*gen_cbeta[gen])
            list_profit_b2.append(b_opt2[gen]*b_price2 - b_opt2[gen]*gen_cbeta[gen])

            list_revenue_p3.append(p_opt3[gen]*p_price3[0])
            list_cost_p3.append(p_opt3[gen]*gen_cmg[gen] + p_opt3[gen]**2*gen_cmg_b[gen])
            list_profit_p3.append(p_opt3[gen]*p_price3[0] - p_opt3[gen]*gen_cmg[gen] - p_opt3[gen]**2*gen_cmg_b[gen])
            list_revenue_a3.append(a_opt3[gen]*a_price3)
            list_cost_a3.append(a_opt3[gen]*gen_calpha[gen])
            list_profit_a3.append(a_opt3[gen]*b_price3 - a_opt3[gen]*gen_calpha[gen])
            list_revenue_b3.append(b_opt3[gen]*b_price3)
            list_cost_b3.append(b_opt3[gen]*gen_cbeta[gen])
            list_profit_b3.append(b_opt3[gen]*b_price3 - b_opt3[gen]*gen_cbeta[gen])

        print("\n energy")
        print(list_revenue_p3)
        print(list_cost_p3)
        print(list_profit_p3)

        print("\n reg reserve")
        print(list_revenue_a3)
        print(list_cost_a3)
        print(list_profit_a3)

        print("\n ext reserve")
        print(list_revenue_b3)
        print(list_cost_b3)
        print(list_profit_b3)

    if power_system:
        #convert into a zone info
        energy_list_1 = np.zeros(len(node_set_index))
        sreserve_list_1 = np.zeros(len(node_set_index))

        for gen in gen_set_index:
            energy_list_1[gen_node[gen]] += p_opt1[gen]
            sreserve_list_1[gen_node[gen]] += a_opt1[gen]

        #convert into a zone info
        energy_list_2 = np.zeros(len(node_set_index))
        sreserve_list_2 = np.zeros(len(node_set_index))
        ereserve_list_2 = np.zeros(len(node_set_index))

        for gen in gen_set_index:
            energy_list_2[gen_node[gen]] += p_opt2[gen]
            sreserve_list_2[gen_node[gen]] += a_opt2[gen]
            ereserve_list_2[gen_node[gen]] += b_opt2[gen]

        #convert into a zone info
        energy_list_3 = np.zeros(len(node_set_index))
        sreserve_list_3 = np.zeros(len(node_set_index))
        ereserve_list_3 = np.zeros(len(node_set_index))

        for gen in gen_set_index:
            energy_list_3[gen_node[gen]] += p_opt3[gen]
            sreserve_list_3[gen_node[gen]] += a_opt3[gen]
            ereserve_list_3[gen_node[gen]] += b_opt3[gen]


        print("\nCC")
        print(energy_list_1)
        print(sreserve_list_1)

        print("\nWCC")
        print(energy_list_2)
        print(sreserve_list_2)
        print(ereserve_list_2)

        print("\nLDC")
        print(energy_list_3)
        print(sreserve_list_3)
        print(ereserve_list_3)

        print(zonas)

    if power_system:
        #convert into a zone info
        cc_list_1R = np.zeros(len(node_set_index))
        cc_list_1C = np.zeros(len(node_set_index))
        cc_list_1P = np.zeros(len(node_set_index))
        wcc_list_1R = np.zeros(len(node_set_index))
        wcc_list_1C = np.zeros(len(node_set_index))
        wcc_list_1P = np.zeros(len(node_set_index))
        ldt_list_1R = np.zeros(len(node_set_index))
        ldt_list_1C = np.zeros(len(node_set_index))
        ldt_list_1P = np.zeros(len(node_set_index))

        for gen in gen_set_index:
            cc_list_1R[gen_node[gen]] += p_opt1[gen]*p_price1[gen_node[gen]] + a_opt1[gen]*a_price1
            cc_list_1C[gen_node[gen]] += p_opt1[gen]*gen_cmg[gen] + p_opt1[gen]**2*gen_cmg_b[gen] + a_opt1[gen]*gen_calpha[gen]
            cc_list_1P[gen_node[gen]] += p_opt1[gen]*p_price1[gen_node[gen]] - p_opt1[gen]*gen_cmg[gen] - p_opt1[gen]**2*gen_cmg_b[gen] + a_opt1[gen]*a_price1 - a_opt1[gen]*gen_calpha[gen]

            wcc_list_1R[gen_node[gen]] += p_opt2[gen]*p_price2[gen_node[gen]] + a_opt2[gen]*a_price2 + b_opt2[gen]*b_price2
            wcc_list_1C[gen_node[gen]] += p_opt2[gen]*gen_cmg[gen] + p_opt2[gen]**2*gen_cmg_b[gen] + a_opt2[gen]*gen_calpha[gen] + b_opt2[gen]*gen_cbeta[gen]
            wcc_list_1P[gen_node[gen]] += p_opt2[gen]*p_price2[gen_node[gen]] - p_opt2[gen]*gen_cmg[gen] - p_opt2[gen]**2*gen_cmg_b[gen] + a_opt2[gen]*a_price2 - a_opt2[gen]*gen_calpha[gen] + b_opt2[gen]*b_price2 - b_opt2[gen]*gen_cbeta[gen]
            
            ldt_list_1R[gen_node[gen]] += p_opt3[gen]*p_price3[gen_node[gen]] + a_opt3[gen]*a_price3 + b_opt3[gen]*b_price3
            ldt_list_1C[gen_node[gen]] += p_opt3[gen]*gen_cmg[gen] + p_opt3[gen]**2*gen_cmg_b[gen] + a_opt3[gen]*gen_calpha[gen] + b_opt3[gen]*gen_cbeta[gen]
            ldt_list_1P[gen_node[gen]] += p_opt3[gen]*p_price3[gen_node[gen]] - p_opt3[gen]*gen_cmg[gen] - p_opt3[gen]**2*gen_cmg_b[gen] + a_opt3[gen]*a_price3 - a_opt3[gen]*gen_calpha[gen] + b_opt3[gen]*b_price3 - b_opt3[gen]*gen_cbeta[gen]
            
        print("\nCC")
        print(cc_list_1R)
        print(cc_list_1C)
        print(cc_list_1P)
        print("\nWCC ")
        print(wcc_list_1R)
        print(wcc_list_1C)
        print(wcc_list_1P)
        print("\nCC Extreme")
        print(ldt_list_1R)
        print(ldt_list_1C)
        print(ldt_list_1P)

    # generadores funcionando
    '''
    m = max(p_opt2)
    list_p = []
    # Finding the index of the maximum element
    i = 0
    while(i < len(a_opt2)):
        if p_opt2[i] != 0:
            list_p.append(i)
        if p_opt2[i] == m:
            print("nIndex of the p max element in a list is", i)
        i += 1
    print(list_p)

    m = max(a_opt2)
    list_a = []
    # Finding the index of the maximum element
    i = 0
    while(i < len(a_opt2)):
        if a_opt2[i] != 0:
            list_a.append(i)
        if a_opt2[i] == m:
            print("Index of the a max element in a list is", i)
        i += 1
    print(list_a)
    #print(a_opt2)

    m = max(b_opt2)
    list_b = []
    # Finding the index of the maximum element
    i = 0
    while(i < len(b_opt2)):
        if b_opt2[i] != 0:
            list_b.append(i)
        if b_opt2[i] == m:
            print("Index of the b max element in a list is", i)
        i += 1
    print(list_b)
    #print(b_opt2)
    '''

# TEST SCENARIOS

def find_price(gen_node,list_node_price):
    price_per_generator = np.zeros(len(gen_node))
    for g in range(len(gen_node)):
        for n in range(len(list_node_price)):
            if gen_node[g] == n:
                price_per_generator[g] = list_node_price[n]
    return price_per_generator

def test_funtion(w, w_bar, p_opt,a_opt,b_opt, node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg,gen_calpha,gen_cbeta,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind, w_sigma):
    flag_print = False
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(lines_node_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="alpha_n", lb=1e-8) # Operational reserve variable
    beta_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="beta_n", lb=1e-8) # Operational reserve variable adverse

    # Set objective function
    obj_value = model.setObjective( sum(p_n[n]*gen_cmg[n]+ gen_calpha[n]*alpha_n[n] + gen_cbeta[n]*beta_n[n] for n in gen_set_index) + sum(s_i[i]*9000 for i in node_set_index), GRB.MINIMIZE)

    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] + (alpha_n[n]+beta_n[n])*w[0] for n in node_gens[i]) + sum((w_bar- w[0])*wind_factor[n] for n in node_wind[i])
                                 - sum(f_ij[i,j] for (i,j) in lines_node_in[i])
                                 + sum(f_ij[i,j] for (i,j) in lines_node_out[i])
                                 + s_i[i]  == node_demand[i] for i in node_set_index), name='const_demand') # Demand constraints
    const_pmax = model.addConstrs((p_n[n] +(alpha_n[n]+beta_n[n])*w[0] <= gen_max[n] for n in gen_set_index), name='const_pmax')
    const_f_t = model.addConstrs( (line_susceptance_matrix[i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in lines_node_index), name='const_f_t')
    const_fmax = model.addConstrs( (f_ij[i,j] <= line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmax')
    const_fmin = model.addConstrs( (f_ij[i,j] >= -line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmin')
    const_t_0 = model.addConstr(t_i[0] == 0)
    
    #cons_op_reserve = model.addConstr(sum(alpha_n[n] for n in gen_set_index) == 1, name='cons_op_reserve') # Operational reserve constraint
    #cons_ad_reserve = model.addConstr(sum(beta_n[n] for n in gen_set_index) == 1, name='cons_op_reserve') # Adversarial reserve constraint
    
    const_pmax2 = model.addConstrs((p_n[n] <= p_opt[n] for n in gen_set_index), name='const_pmax2')
    cons_op_reserve = model.addConstrs((alpha_n[n] <= a_opt[n] for n in gen_set_index), name='cons_op_reserve')
    cons_ad_reserve = model.addConstrs((beta_n[n] <= b_opt[n] for n in gen_set_index), name='cons_ad_reserve')

    # Constraints
    cons_pmax = model.addConstrs((p_n[n]<= gen_max[n] for n in gen_set_index), name='cons_pmax')

    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()
    ## Primal solution
    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    for v in p_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution alpha_n is")
    for v in alpha_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution beta_n is")
    for v in beta_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution f_ij is")
    for v in f_ij.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution s_i is")
    for v in s_i.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution w is")
    print(w[0])


    o_opt = obj.getValue()
    p_opt = []
    a_opt = []
    b_opt = []
    for v in p_n.values():
        p_opt.append(round(v.X,4))
    for v in alpha_n.values():
        a_opt.append(round(v.X,4))
    for v in beta_n.values():
        b_opt.append(round(v.X,4))

    return o_opt, p_opt, a_opt, b_opt

def test(p_opt1,a_opt1,b_opt1,p_opt2,a_opt2,b_opt2, p_opt3,a_opt3,b_opt3):
    list_obj1_o = []
    list_obj2_o = []
    list_obj3_o = []
    list_omega = []
    for _ in range(300):
        w_bar = 40; w_sigma = 10
        w = np.random.normal(0, w_sigma, 1)
        list_omega.append(w)
        print(w)
        print("\nTest Solve",_)

        gen_cmg_1 = find_price(gen_node, p_price1)
        gen_calpha_1 = np.ones(len(gen_node))*a_price1
        gen_cbeta_1 = np.ones(len(gen_node))*9000

        o_opt01, p_opt01, a_opt01, b_opt01 = test_funtion(w, w_bar, p_opt1,a_opt1,b_opt1, node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg_1,gen_calpha_1,gen_cbeta_1,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind, w_sigma)
        list_obj1_o.append(o_opt01)

        gen_cmg_2 = find_price(gen_node, p_price2)
        gen_calpha_2 = np.ones(len(gen_node))*a_price2
        gen_cbeta_2 = np.ones(len(gen_node))*b_price2
        o_opt02, p_opt02, a_opt02, b_opt02 = test_funtion(w, w_bar, p_opt2,a_opt2,b_opt2, node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg_2,gen_calpha_2,gen_cbeta_2,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind, w_sigma)
        list_obj2_o.append(o_opt02)
        
        gen_cmg_3 = find_price(gen_node, p_price3)
        gen_calpha_3 = np.ones(len(gen_node))*a_price3
        gen_cbeta_3 = np.ones(len(gen_node))*b_price3
        o_opt03, p_opt03, a_opt03, b_opt03 = test_funtion(w, w_bar, p_opt3,a_opt3,b_opt3, node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg_3,gen_calpha_3,gen_cbeta_3,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind, w_sigma)
        list_obj3_o.append(o_opt03)

    return list_omega, list_obj1_o, list_obj2_o, list_obj3_o

if flag_test:
    list_omega, list_obj1_o, list_obj2_o, list_obj3_o = test(p_opt1,a_opt1,b_opt1,p_opt2,a_opt2,b_opt2, p_opt3,a_opt3,b_opt3)
    print("Omeg, mean {}, std {}, max {}, min {}".format(np.mean(list_omega),np.std(list_omega),np.max(list_omega),np.min(list_omega)))
    print("CC_o, mean {}, std {}, max {}, min {}".format(np.mean(list_obj1_o),np.std(list_obj1_o),np.max(list_obj1_o),np.min(list_obj1_o)))
    print("WCC_o, mean {}, std {}, max {}, min {}".format(np.mean(list_obj2_o),np.std(list_obj2_o),np.max(list_obj2_o),np.min(list_obj2_o)))
    print("LDT_o, mean {}, std {}, max {}, min {}".format(np.mean(list_obj3_o),np.std(list_obj3_o),np.max(list_obj3_o),np.min(list_obj3_o)))

def test_network(p_opt1,a_opt1,b_opt1,p_opt2,a_opt2,b_opt2, p_opt3,a_opt3,b_opt3):
    list_obj1_o = []; list_obj2_o = []; list_obj3_o = []
    list_p1_o = []; list_p2_o = []; list_p3_o = []
    list_a1_o = []; list_a2_o = []; list_a3_o = []
    list_b1_o = []; list_b2_o = []; list_b3_o = []
    list_omega = []

    for _ in range(100):
        w_bar = 4000; w_sigma = 800
        w = np.random.normal(0, w_sigma, 1)
        list_omega.append(w)
        print(w)
        print("\nTest Solve",_)

        ############## model CC ##########
        gen_cmg_1 = find_price(gen_node, p_price1)
        gen_calpha_1 = np.ones(len(gen_node))*a_price1
        gen_cbeta_1 = np.ones(len(gen_node))*9000

        o_opt01, p_opt01, a_opt01, b_opt01 = test_funtion(w, w_bar, p_opt1,a_opt1,b_opt1, node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg_1,gen_calpha_1,gen_cbeta_1,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind, w_sigma)
        list_obj1_o.append(o_opt01)

        #convert into a zone info
        energy_list_10 = np.zeros(len(node_set_index))
        sreserve_list_10 = np.zeros(len(node_set_index))

        for gen in gen_set_index:
            energy_list_10[gen_node[gen]] += p_opt01[gen]
            sreserve_list_10[gen_node[gen]] += a_opt01[gen]

        list_p1_o.append(energy_list_10) # list_p1_o.append(p_opt01)
        list_a1_o.append(sreserve_list_10) # list_a1_o.append(a_opt01)

        ############## model WCC ##########
        gen_cmg_2 = find_price(gen_node, p_price2)
        gen_calpha_2 = np.ones(len(gen_node))*a_price2
        gen_cbeta_2 = np.ones(len(gen_node))*b_price2
        o_opt02, p_opt02, a_opt02, b_opt02 = test_funtion(w, w_bar, p_opt2,a_opt2,b_opt2, node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg_2,gen_calpha_2,gen_cbeta_2,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind, w_sigma)
        list_obj2_o.append(o_opt02)

        #convert into a zone info
        energy_list_20 = np.zeros(len(node_set_index))
        sreserve_list_20 = np.zeros(len(node_set_index))
        ereserve_list_20 = np.zeros(len(node_set_index))

        for gen in gen_set_index:
            energy_list_20[gen_node[gen]] += p_opt02[gen]
            sreserve_list_20[gen_node[gen]] += a_opt02[gen]
            ereserve_list_20[gen_node[gen]] += b_opt02[gen]

        list_p2_o.append(energy_list_20) # list_p2_o.append(p_opt02)
        list_a2_o.append(sreserve_list_20) # list_a2_o.append(a_opt02)
        list_b2_o.append(ereserve_list_20) # list_b2_o.append(b_opt02)
        
        ############## model LDT ##########
        gen_cmg_3 = find_price(gen_node, p_price3)
        gen_calpha_3 = np.ones(len(gen_node))*a_price3
        gen_cbeta_3 = np.ones(len(gen_node))*b_price3
        o_opt03, p_opt03, a_opt03, b_opt03 = test_funtion(w, w_bar, p_opt3,a_opt3,b_opt3, node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg_3,gen_calpha_3,gen_cbeta_3,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind, w_sigma)
        list_obj3_o.append(o_opt03)

        #convert into a zone info
        energy_list_30 = np.zeros(len(node_set_index))
        sreserve_list_30 = np.zeros(len(node_set_index))
        ereserve_list_30 = np.zeros(len(node_set_index))

        for gen in gen_set_index:
            energy_list_30[gen_node[gen]] += p_opt03[gen]
            sreserve_list_30[gen_node[gen]] += a_opt03[gen]
            ereserve_list_30[gen_node[gen]] += b_opt03[gen]

        list_p3_o.append(energy_list_30) # list_p3_o.append(p_opt03)
        list_a3_o.append(sreserve_list_30) # list_a3_o.append(a_opt03)
        list_b3_o.append(ereserve_list_30) # list_b3_o.append(b_opt03)
        
    return list_omega, list_obj1_o, list_obj2_o, list_obj3_o

if flag_test_Network:
    print(zonas)
    list_omega, list_obj1_o, list_obj2_o, list_obj3_o = test_network(p_opt1,a_opt1,b_opt1,p_opt2,a_opt2,b_opt2, p_opt3,a_opt3,b_opt3)
    print("Omeg, mean {}, std {}, max {}, min {}".format(np.mean(list_omega),np.std(list_omega),np.max(list_omega),np.min(list_omega)))
    print("CC_o, mean {}, std {}, max {}, min {}".format(np.mean(list_obj1_o),np.std(list_obj1_o),np.max(list_obj1_o),np.min(list_obj1_o)))
    print("WCC_o, mean {}, std {}, max {}, min {}".format(np.mean(list_obj2_o),np.std(list_obj2_o),np.max(list_obj2_o),np.min(list_obj2_o)))
    print("LDT_o, mean {}, std {}, max {}, min {}".format(np.mean(list_obj3_o),np.std(list_obj3_o),np.max(list_obj3_o),np.min(list_obj3_o)))

if aux_test == True:
    cdf_mean  = 12
    cdf_deviation = 5
    test_tnf = truncated_normal_funtion(cdf_mean,cdf_deviation)

    ref_mean = 11
    ref_deviation = 5
    test_taylor = taylor_approximation(cdf_mean,cdf_deviation,ref_mean,ref_deviation)

    print('function_tnf, %s' % test_tnf)
    print('function_taylor, %s' % test_taylor)

if (test_conditional_negative == True) or  (test_conditional_positive == True):
    # TEST CONDITIONAL
    value_crit = -3
    mu_known = 2
    sigma_known = 3

    z_auxiliar = (value_crit- mu_known)/sigma_known
    print('z_auxiliar',z_auxiliar)

    #parameters
    mu = 2.0
    sigma = 3.02

    if test_conditional_negative == True:
        print("LESS")
        z_auxiliar = (value_crit- mu_known)/sigma_known
        frac_auxiliar = norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)
        print(norm.pdf(z_auxiliar), norm.cdf(z_auxiliar)) 
        print("fraction_auxiliar",frac_auxiliar)

        tnf = truncated_normal_funtion(mu,sigma)
        tnf_taylor = truncated_normal_funtion(mu_known,sigma_known) + (mu-mu_known)*truncated_normal_funtion_dmu(mu_known,sigma_known) + (sigma - sigma_known)*truncated_normal_funtion_dsigma(mu_known,sigma_known)
        print("tnf {}, tnf_taylor {}".format(tnf, tnf_taylor))

        media_lower = mu - sigma*frac_auxiliar
        sd_lower = (sigma)*np.sqrt((1-z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2))
        #print("term1 {}, term2 {}, term3 {}, sum {}".format(1,-z_auxiliar*(frac_auxiliar), -(frac_auxiliar)**2, (1+z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2)))
        print("media_cond {}, sd_cond {}".format(media_lower, sd_lower))

        media_lower_known = mu_known - sigma_known*frac_auxiliar
        sd_lower_known = (sigma_known)*np.sqrt(1-z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2)
        print("media_cond_know {}, sd_cond_known {}".format(media_lower_known, sd_lower_known))

        tnf_cond = truncated_normal_funtion(media_lower,sd_lower)
        tnf_taylor_cond = truncated_normal_funtion(media_lower_known,sd_lower_known) 
        + (media_lower - media_lower_known)*truncated_normal_funtion_dmu(media_lower_known,sd_lower_known) 
        + (sd_lower - sd_lower_known)*(truncated_normal_funtion_dsigma(media_lower_known,sd_lower_known)+truncated_normal_funtion_dmu(media_lower_known,sd_lower_known))

        print("tnf_cond {}, tnf_taylor_cond {}".format(tnf_cond, tnf_taylor_cond))

    if test_conditional_positive == True:
        print("GREATER")
        frac_auxiliar = norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))
        print("fraction_auxiliar",frac_auxiliar)

        tnf = truncated_normal_funtion(mu,sigma)
        tnf_taylor = truncated_normal_funtion(mu_known,sigma_known) + (mu-mu_known)*truncated_normal_funtion_dmu(mu_known,sigma_known) + (sigma - sigma_known)*truncated_normal_funtion_dsigma(mu_known,sigma_known)
        print("tnf {}, tnf_taylor {}".format(tnf, tnf_taylor))

        media_lower = mu + sigma*frac_auxiliar
        #print("term1 {}, term2 {}, term3 {}, sum {}".format(1,z_auxiliar*(frac_auxiliar),- (frac_auxiliar)**2, (1+z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2)))
        sd_lower = (sigma)*np.sqrt((1+z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2))
        print("media_cond {}, sd_cond {}".format(media_lower, sd_lower))

        media_lower_known = mu_known + sigma_known*frac_auxiliar
        sd_lower_known = np.sqrt(((sigma_known)**2)*(1+z_auxiliar*(frac_auxiliar) - (frac_auxiliar)**2))
        print("media_cond_know {}, sd_cond_known {}".format(media_lower_known, sd_lower_known))

        tnf_cond = truncated_normal_funtion(media_lower,sd_lower)
        tnf_taylor_cond = truncated_normal_funtion(media_lower_known,sd_lower_known) 
        + (media_lower -media_lower_known)*truncated_normal_funtion_dmu(media_lower_known,sd_lower_known) 
        + (sd_lower- sd_lower_known)*(truncated_normal_funtion_dsigma(media_lower_known,sd_lower_known)+truncated_normal_funtion_dmu(media_lower_known,sd_lower_known))

        print("tnf_cond {}, tnf_taylor_cond {}".format(tnf_cond, tnf_taylor_cond))


