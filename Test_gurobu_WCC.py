import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm
import pandas as pd

# Instruction variables
aux_test = False
test_conditional_negative = False
test_conditional_positive = False
power_system = True


# System parameters
if power_system == False:
    node_set_index = [0,1,2]
    gen_set_index = [0,1,2]
    lines_node_index = [(0,1),(1,2),(0,2)]
    line_reactance = [2.5,2.5,2]
    line_f_max = [25,1000,1000]
    node_demand = [0,0,120]
    gen_max = [40,40,30]
    gen_cmg = [40, 45, 50]
    gen_calpha = [55, 50, 75]
    gen_cbeta = [65, 70, 95]
    gen_node = [1,2,1]
else:
    # Lines data
    df_2 = pd.read_csv('gridDetails.csv')
    df_2.keys()
    zonas = np.unique(df_2[['From Zone', 'To Zone']].values)

    lines_node_index = []
    line_reactance = []
    line_f_max = []
    node_demand = []

    for ind in df_2.index:
        index_barra_in = np.where(zonas == df_2['From Zone'][ind])[0][0]
        index_barra_out = np.where(zonas == df_2['To Zone'][ind])[0][0]
        lines_node_index.append((index_barra_in,index_barra_out))
        line_reactance.append(df_2['Reactance (per unit)'][ind])
        line_f_max.append(df_2['Capacity (MW)'][ind])

    df = pd.read_csv('generator_data.csv')

    node_set_index = []
    gen_set_index = []
    gen_max = []
    gen_cmg = []
    gen_calpha = [] ## Arreglar
    gen_cbeta = [] ## Arreglar
    gen_node = []

    # Node data
    for ind in range(len(zonas)):
        node_set_index.append(ind)
        df_3 = pd.read_csv(zonas[ind]+'.csv')
        demanda_promedio = df_3[zonas[ind]].mean()
        node_demand.append(demanda_promedio)

    # Generators data
    for ind in df.index:
        barra_string = df['Zone Location'][ind]
        index_barra = np.where(zonas == barra_string)[0][0]
        costo_marginal = df['Dispatch Cost Coefficient a ($/MWh)'][ind]
        capacidad = df['Capacity (MW)'][ind]

        gen_set_index.append(ind)
        gen_node.append(index_barra)
        gen_cmg.append(costo_marginal)
        gen_calpha.append(costo_marginal)
        gen_cbeta.append(costo_marginal)
        gen_max.append(capacidad)

wind_set_index = [0]
wind_node = [0]

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



######## functions ####
# pdf (density)
# norm.pdf(x)
# cummulative cdf   
# norm.cdf(-1.96)
# inverse cdf
# norm.ppf(norm.cdf(1.96))

def cutting_planes_problem_ED_network(node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind):
    flag_print = False
    #System parameters
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)
    w_bar = 20
    w_sigma = 4

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(lines_node_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS, name='s_i')
    alpha_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="alpha_n", lb=0) # Operational reserve variable

    # Set objective function
    obj_value = model.setObjective( sum(p_n[n]*gen_cmg[n] + gen_calpha[n]*alpha_n[n] for n in gen_set_index) + sum(s_i[i]*1000000 for i in node_set_index), GRB.MINIMIZE)

    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in node_gens[i]) + sum(w_bar for n in node_wind[i])
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
    
def cutting_planes_problem_WCC_ED_network(node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind):
    flag_print = False
    #System parameters
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)
    w_bar = 20
    w_sigma = 4

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
    const_demand = model.addConstrs((sum(p_n[n] for n in node_gens[i]) + sum(w_bar for n in node_wind[i])
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
            truncated_normal_funtion(p_n_star[n]-p_n_max[n] - alpha_n_star[n]*w_sigma*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)),
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

def cutting_planes_problem_WCC_PW_beta():
    flag_print = False
    #System parameters
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)
    w_bar = 20
    w_sigma = 4

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
    obj_value = model.setObjective( sum(p_n[n]*gen_cmg[n] + gen_calpha[n]*alpha_n[n] + gen_cbeta[n]*beta_n[n] for n in gen_set_index) + sum(s_i[i]*1000000 for i in node_set_index), GRB.MINIMIZE)

    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in node_gens[i]) + sum(w_bar for n in node_wind[i])
                                 - sum(f_ij[i,j] for (i,j) in lines_node_in[i])
                                 + sum(f_ij[i,j] for (i,j) in lines_node_out[i])
                                 + s_i[i]  == node_demand[i] for i in node_set_index), name='const_demand') # Demand constraints
    const_pmax = model.addConstrs((p_n[n] <= gen_max[n] for n in gen_set_index), name='const_pmax')
    const_f_t = model.addConstrs( (line_susceptance_matrix[i,j]*(t_i[i]-t_i[j]) == f_ij[i,j] for (i,j) in lines_node_index), name='const_f_t')
    const_fmax = model.addConstrs( (f_ij[i,j] <= line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmax')
    const_fmin = model.addConstrs( (f_ij[i,j] >= -line_f_max_matrix[i,j] for (i,j) in lines_node_index), name='const_fmin')
    const_t_0 = model.addConstr(t_i[0] == 0)
    
    cons_op_reserve = model.addConstr(sum(alpha_n[n] for n in gen_set_index) == 1, name='cons_op_reserve') # Operational reserve constraint
    cons_ad_reserve = model.addConstr(sum(beta_n[n] for n in gen_set_index) == 1, name='cons_op_reserve') # Adversarial reserve constraint

    # Constraints
    cons_pmax = model.addConstrs((p_n[n]<= gen_max[n] for n in gen_set_index), name='cons_pmax') 

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

        w_critic = -5
        z_auxiliar = w_critic/w_sigma
        # Check if solution satisfy the WCC constraint
        if all(
            truncated_normal_funtion(p_n_star[n]-gen_set_index[n] - alpha_n_star[n]*w_sigma*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)),
                                    (w_sigma*alpha_n_star[n])*np.sqrt(1 - z_auxiliar*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2))
                +
            truncated_normal_funtion(p_n_star[n]-gen_set_index[n] + (alpha_n_star[n]+beta_n_star[n])*w_sigma*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))),
                                    (w_sigma*(alpha_n_star[n]+beta_n_star[n]))*np.sqrt(1 + z_auxiliar*norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2))
            <= epsilon for n in gen_set_index):
            break

        # Add a cutting plane to the model 
        for n in gen_set_index:
            media_lower = p_n_star[n]-gen_set_index[n] - (alpha_n_star[n]+beta_n_star[n])*w_sigma*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))
            desviacion_lower = (w_sigma*(alpha_n_star[n]+beta_n_star[n]))*np.sqrt(1 - z_auxiliar*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2)

            media_greater = p_n_star[n]-gen_set_index[n] + alpha_n_star[n]*w_sigma*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))
            desviacion_greater = (w_sigma*alpha_n_star[n])*np.sqrt(1 + z_auxiliar*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2)
            print('ITERATION, %s, gen, %s' %(_,n))
            print("media_d {}, sd_d {}, media_u {}, sd_u {}".format(media_lower,desviacion_lower,media_greater,desviacion_greater))
            print("tnf_d {}, tnf_u {}, sum {}".format(truncated_normal_funtion(media_lower, desviacion_lower),truncated_normal_funtion(media_greater, desviacion_greater),
                                                      truncated_normal_funtion(media_lower, desviacion_lower)+truncated_normal_funtion(media_greater, desviacion_greater)))
            
            if truncated_normal_funtion(media_lower, desviacion_lower) + truncated_normal_funtion(media_greater, desviacion_greater) >= epsilon:
                print('ERROR in iterration, %s, with generador, %s' %(_,n))
                model.addConstr(
                    truncated_normal_funtion(media_lower, desviacion_lower) 
                    + ((p_n[n]-gen_max[n] - (alpha_n[n]+beta_n[n])*w_sigma*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))) - media_lower)*truncated_normal_funtion_dmu(media_lower, desviacion_lower) 
                    + (((alpha_n[n]+beta_n[n])*w_sigma)*np.sqrt(1 - z_auxiliar*(norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar)) - (norm.pdf(z_auxiliar)/norm.cdf(z_auxiliar))**2) - desviacion_lower)*truncated_normal_funtion_dsigma(media_lower, desviacion_lower) 
                    + truncated_normal_funtion(media_greater, desviacion_greater) 
                    + ((p_n[n]-gen_max[n] + (alpha_n[n])*w_sigma*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))) ) - media_greater)*truncated_normal_funtion_dmu(media_greater, desviacion_greater) 
                    + ((alpha_n[n]*w_sigma)*np.sqrt(1 + z_auxiliar*(norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar))) - (norm.pdf(z_auxiliar)/(1-norm.cdf(z_auxiliar)))**2) - desviacion_greater)*truncated_normal_funtion_dsigma(media_greater, desviacion_greater)
                    <= epsilon)
        

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


def EconomicDispatch_LDT_network(node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind):
    flag_print = False
    #System parameters
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.05
    inv_phi_ext = norm.ppf(epsilon_ext)
    w_bar = 600
    w_sigma = 40

    # Create a new model
    model = gb.Model()

    # Create variables
    p_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name='p_n')
    t_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='t_i')
    f_ij = model.addVars(lines_node_index, vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY, name='fij')
    s_i = model.addVars(node_set_index, vtype=GRB.CONTINUOUS, name='s_i')
    #p_n = model.addVars(N, vtype=GRB.CONTINUOUS, name="p_n", lb=0)         # Generation variable
    alpha_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="alpha_n", lb=0) # Operational reserve variable
    beta_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="beta_n", lb=0)   # Adversarial reserve variable

    omega_star = model.addVar(vtype=GRB.CONTINUOUS, name="omega_star", lb=-GRB.INFINITY) # Optimal critical value from the extreme event region
    lambda_n = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name="lambda_n", lb=-GRB.INFINITY)     # Dual variable of the constrainst to obtain "omega_star"

    aux_omega_a = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_omega_a", lb=-GRB.INFINITY)   # Auxilar value to represent the positive value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_omega_b = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_omega_b", lb=-GRB.INFINITY)   # Auxilar value to represent the negative value of "omega_star*(alpha_n +beta_n)" in the SOC
    aux_lambda_a = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_lambda_a", lb=-GRB.INFINITY) # Auxilar value to represent the positive value of "lambda_n*(alpha_n +beta_n)" in the SOC
    aux_lambda_b = model.addVars(gen_set_index, vtype=GRB.CONTINUOUS, name= "aux_lambda_b", lb=-GRB.INFINITY) # Auxilar value to represent the negative value of "lambda_n*(alpha_n +beta_n)" in the SOC
    
    # Set objective function
    obj_value = model.setObjective( sum(p_n[n]*gen_cmg[n] + gen_calpha[n]*alpha_n[n] + gen_cbeta[n]*beta_n[n] for n in gen_set_index) + sum(s_i[i]*1000000 for i in node_set_index), GRB.MINIMIZE)

    # Add constraints
    const_demand = model.addConstrs((sum(p_n[n] for n in node_gens[i]) + sum(w_bar for n in node_wind[i])
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
    #model.addConstrs(p_n[n] - p_n_max[n] - alpha_n[n]*inv_phi_eps*w_sigma <= 0 for n in range(N))     # Max generation limit constraint%
    
    model.addConstrs(aux_omega_a[n] == omega_star*alpha_n[n] for n in gen_set_index)
    model.addConstrs(aux_omega_b[n] == omega_star*beta_n[n] for n in gen_set_index)
    model.addConstrs(aux_lambda_a[n] == lambda_n[n]*alpha_n[n] for n in gen_set_index)
    model.addConstrs(aux_lambda_b[n] == lambda_n[n]*beta_n[n] for n in gen_set_index)

    ## LDT Constraint
    model.addConstr(w_sigma**(-0.5)*omega_star + inv_phi_ext <= 0)
    model.addConstrs(w_sigma**(-1)*omega_star + aux_lambda_a[n] + aux_lambda_b[n] == 0 for n in gen_set_index)
    model.addConstrs(-gen_max[n] + p_n[n] - aux_omega_a[n] - aux_omega_b[n] == 0 for n in gen_set_index)

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
    for v in p_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution alpha_n is")
    for v in alpha_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution beta_n is")
    for v in beta_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution omega_star is")
    print("{}: {}".format(omega_star.varName, omega_star.X))
    print("A solution lambda_n is")
    for v in lambda_n.values():
        print("{}: {}".format(v.varName, v.X))
    print("A solution f_ij is")
    for v in f_ij.values():
        print("{}: {}".format(v.varName, v.X))

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


#cutting_planes_problem_ED_network(node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind)
#cutting_planes_problem_WCC_ED_network(node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind)

##cutting_planes_problem_WCC_PW()
##cutting_planes_problem_WCC_PW_beta()

EconomicDispatch_LDT_network(node_set_index,gen_set_index,lines_node_index,node_gens,gen_max,line_f_max_matrix,gen_cmg,line_susceptance_matrix,node_demand,lines_node_in,lines_node_out, node_wind)
'''
Model: ED
The optimal value is 4475.0 # 4525.0 # 4550.0
A solution p_n is
['p_n[0]' 'p_n[1]' 'p_n[2]']: [40. 40. 20.]
A solution alpha_n is
['alpha_n[0]' 'alpha_n[1]' 'alpha_n[2]']: [0. 0. 1.]

Model: WCC LDT
The optimal value is 4550.791130914369  #4600.791130914369 #4615.791225045131 #4620.75177181496
A solution p_n is
['p_n[0]' 'p_n[1]' 'p_n[2]']: [39.81879537 33.2041832  26.97702143]
A solution alpha_n is
['alpha_n[0]' 'alpha_n[1]' 'alpha_n[2]']: [1.0000000e-08 9.9999998e-01 1.0000000e-08]
A solution beta_n is
['beta_n[0]' 'beta_n[1]' 'beta_n[2]']: [9.9999998e-01 1.0000000e-08 1.0000000e-08]

Model: LDT
The optimal value is 4570.000692921458 # 4615.002078281504 # 4620.002078346887 #4640.002078350703
A solution p_n is
['p_n[0]' 'p_n[1]' 'p_n[2]']: [39.99985564 35.00007218 25.00007218]
A solution alpha_n is
['alpha_n[0]' 'alpha_n[1]' 'alpha_n[2]']: [1.15483149e-05 4.99994226e-01 4.99994226e-01]
A solution beta_n is
['beta_n[0]' 'beta_n[1]' 'beta_n[2]']: [1.73236006e-05 4.99991338e-01 4.99991338e-01]
A solution omega_star is
['omega_star[0]']: [-5.]
A solution lambda_n is
['lambda_n[0]' 'lambda_n[1]' 'lambda_n[2]']: [4.32946691e+04 1.25001805e+00 1.25001805e+00]
'''

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


