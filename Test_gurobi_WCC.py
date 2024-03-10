import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm

def EconomicDispatch():
    # Parameters
    N = 1
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
    
    # Test variables
    t_0 = model.addMVar(1, vtype=GRB.CONTINUOUS, name="t_0", lb=0)         # p - pmax
    t_1 = model.addMVar(1, vtype=GRB.CONTINUOUS, name="t_1", lb=0)         # p
    t_2 = model.addMVar(1, vtype=GRB.CONTINUOUS, name="t_2", lb=0)         # sigma*alpha
    t_3 = model.addMVar(1, vtype=GRB.CONTINUOUS, name="t_3", lb=0)         # alpha

    e_1 = model.addMVar(1, vtype=GRB.CONTINUOUS, name="e_1", lb=0)

    # Constraints
    cons_demand = model.addConstr(sum(p_n[n] for n in range(N)) + w_bar == d, name='cons_demand') # Energy balance constraint
    cons_reserve = model.addConstr(sum(alpha_n[n] for n in range(N)) == 1, name='cons_reserve')   # Operational reserve
    model.addConstrs(p_n[n] - p_n_max[n] + alpha_n[n]*inv_phi_eps*w_sigma <= 0 for n in range(N)) # Max generation limit constraint
    
    # Test constraints
    model.addConstr(t_0  == t_1 - 30) # here 30 is pmax
    model.addConstr(t_2  == 5*t_3)    # here 5 is sigma_w

    #model.addGenConstrExp(t_1,t_2)
    #model.addGenConstrLog(t_1,t_2)
    #model.addConstr(t_1 + t_2 >= 5, name='cons_demand')

    # Objective function
    obj = model.setObjective(sum(c_p_n[n]*p_n[n] + c_alpha_n[n]*alpha_n[n] for n in range(N)) + 4*t_1+2*t_2,GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Print results
    obj = model.getObjective()

    print("\nThe optimal value is", obj.getValue())
    print("A solution p_n is")
    print("{}: {}".format(p_n.varName, p_n.X))
    print("A solution alpha_n is")
    print("{}: {}".format(alpha_n.varName, alpha_n.X))
    print("A solution t1 is")
    print("{}: {}".format(t_1.varName, t_1.X))

def test_function():
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    m = gp.Model("Nonlinear_Integer_Programming_with_Cutting_Planes")

    # Define decision variables
    x = m.addVar(name="x", vtype=GRB.INTEGER)
    y = m.addVar(name="y", vtype=GRB.INTEGER)

    # Set objective
    m.setObjective(x**2 + y**2, GRB.MAXIMIZE)

    # Add constraints
    m.addConstr(x + y <= 4, "c0")

    # Optimize the model
    m.optimize()

    # Add cutting planes until the optimal solution is integer
    while True:
        m.optimize()
        print(1)
        # Check if solution is integer
        if all(var.x % 1 == 0 for var in m.getVars()):
            break
        
        # Get fractional solution
        fractional_solution = [var.x for var in m.getVars()]
        
        # Add a cutting plane to the model
        m.addConstr(x**2 + y**2 <= (sum(fractional_solution)) - 1)

    # Print the results
    print("Status:", m.status)
    print("Optimal Solution:")
    for v in m.getVars():
        print(v.varName, "=", v.x)
    print("Objective Value:", m.objVal)

def test_function_2():
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np

    # Parameters for the truncated Gaussian distribution
    mean = 0
    std_dev = 1
    lower_bound = -1
    upper_bound = 1

    # Create a new model
    m = gp.Model("Expectation_of_Truncated_Gaussian")

    # Define the decision variable
    x = m.addVar(name="x")

    # Set objective: maximize the expected value
    m.setObjective(2*x, GRB.MAXIMIZE)

    # Define piecewise linear approximation of the error function
    def piecewise_linear_erf(z, num_segments=10):
        segments = np.linspace(-3, 3, num_segments + 1)
        for i in range(len(segments) - 1):
            if segments[i] <= z < segments[i + 1]:
                return i / num_segments + (z - segments[i]) / (segments[i + 1] - segments[i]) * (1 / num_segments)
        return 1

    # Add constraints to enforce the truncated range using piecewise linear approximation of erf
    cdf_lower = piecewise_linear_erf((lower_bound - mean) / (std_dev * np.sqrt(2)))
    cdf_upper = piecewise_linear_erf((upper_bound - mean) / (std_dev * np.sqrt(2)))

    m.addConstr(cdf_lower + (cdf_upper - cdf_lower) * (x - lower_bound) / (upper_bound - lower_bound) <= 0.5, "truncation_lower")
    m.addConstr(cdf_lower + (cdf_upper - cdf_lower) * (x - lower_bound) / (upper_bound - lower_bound) >= 0.5, "truncation_upper")

    # Optimize the model
    m.optimize()

    # Print the results
    print("Optimal Solution:")
    print("Expected Value:", x.x)

#test_function()



'''
cdf_mean  = 12
cdf_deviation = 5
x = 6
valor = norm.cdf((x-cdf_mean)/cdf_deviation)
z = (x-cdf_mean)/cdf_deviation
'''

# functions
def pdf_function(x):
    phi = np.exp((-1/2)*x**2)/np.sqrt(2*np.pi)
    return phi

# initial function WCC
def truncated_normal_funtion(mu,sigma):
    z_aux = (-mu/sigma) 
    wcc = mu*(1- norm.cdf(z_aux)) + (sigma/(np.sqrt(2*np.pi)))*np.exp((-1/2)*z_aux**2)
    return wcc

def truncated_normal_funtion_dmu(mu,sigma):
    z_aux = (-mu/sigma)
    term_1 = (1-norm.cdf(z_aux))
    term_2 = mu*((1/sigma)*norm.pdf(z_aux))
    term_3 = -(mu/(sigma*np.sqrt(2*np.pi)))*np.exp((-1/2)*z_aux**2)
    wcc_dmu = term_1 + term_2 + term_3
    return wcc_dmu

def truncated_normal_funtion_dsigma(mu,sigma):
    z_aux = (-mu/sigma)
    term_1 = -((mu**2)/(sigma**2))*norm.pdf(z_aux)
    term_2 = (1/np.sqrt(2*np.pi))*(1-((mu**2)/(sigma**2)))*np.exp((-1/2)*z_aux**2)
    wcc_dsigma = term_1 + term_2
    return wcc_dsigma

def taylor_approximation(mu,sigma,a,b):
    term_1 = truncated_normal_funtion(a,b)
    term_2 = truncated_normal_funtion_dmu(a,b)*(mu-a)
    term_3 = truncated_normal_funtion_dsigma(a,b)*(sigma-b)
    t_approx = term_1 + term_2 + term_3
    return t_approx

'''
a = truncated_normal_funtion(cdf_mean,cdf_deviation)
b = truncated_normal_funtion_dmu(cdf_mean,cdf_deviation)
c = truncated_normal_funtion_dsigma(cdf_mean,cdf_deviation)

print('function, %s' % a)
print('dmu, %s' % b)
print('dsigma, %s' % c)
'''

cdf_mean  = 12
cdf_deviation = 5
test_tnf = truncated_normal_funtion(cdf_mean,cdf_deviation)

ref_mean = 11
ref_deviation = 5
test_taylor = taylor_approximation(cdf_mean,cdf_deviation,ref_mean,ref_deviation)

print('function_tnf, %s' % test_tnf)
print('function_taylor, %s' % test_taylor)

# pdf (density)
# norm.pdf(x)
#inverse cdf   
#norm.cdf(-1.96)
#inverse cdf
#norm.ppf(norm.cdf(1.96))