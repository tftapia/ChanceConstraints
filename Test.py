import cvxpy as cp
import numpy as np
from scipy.stats import norm


def simple_problem():
    # Problem data
    n = 2
    d = np.array([70])
    c_n = np.array([25,66])
    p_n_max = np.array([30,100])

    # Variables
    p_n = cp.Variable(n)

    # Objective function
    objective = cp.Minimize(c_n.T @ p_n)
    # Constraints
    constraints = []
    constraints.append(sum(p_n) >= d)
    constraints.append(p_n >= 0)
    constraints.append(p_n <= p_n_max)

    # Solve problem
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    # Print results
    print("\nThe optimal value is", problem.value)
    print("A solution p_n is")
    print(p_n.value)
    print("The demand constraint dual solution is")
    print(problem.constraints[0].dual_value)

def normal_problem():
    # Problem data
    n = 2
    d = np.array([70])
    cp_n = np.array([25,66])
    calpha_n = np.array([40,80])
    p_n_max = np.array([100,100])
    epsilon = 0.05
    inv_phi_e = norm.ppf(1-epsilon)

    w_bar = 5
    w_sigma = 5

    # Variables
    p_n = cp.Variable(n)
    alpha_n = cp.Variable(n)

    # Objective function
    objective = cp.Minimize(cp_n.T @ p_n + calpha_n.T @ alpha_n)
    # Constraints
    constraints = []
    constraints.append(sum(p_n) + w_bar >= d)
    constraints.append(1- sum(alpha_n) == 0)
    constraints.append(p_n >= 0)
    constraints.append(p_n - p_n_max + alpha_n*inv_phi_e*w_sigma <= 0)
    constraints.append(alpha_n >= 0)

    # Solve problem
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    # Print results
    print("\nThe optimal value is", problem.value)
    print("A solution p_n is")
    print(p_n.value)
    print("A solution alpha_n is")
    print(alpha_n.value)
    print("The demand constraint dual solution is")
    print(problem.constraints[0].dual_value)
    print("The reserve constraint dual solution is")
    print(problem.constraints[1].dual_value)

def hard_problem():
    # Problem data
    n = 2
    d = np.array([70])
    cp_n = np.array([25,66])
    calpha_n = np.array([40,80])
    cbeta_n = np.array([80,100])
    p_n_max = np.array([100,100])
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.02
    inv_phi_ext =norm.ppf(1-epsilon_ext)

    w_bar = 5
    w_sigma = 5

    # Variables
    p_n = cp.Variable(n)
    alpha_n = cp.Variable(n)
    beta_n = cp.Variable(n)
    lambda_n = cp.Variable(n)
    omega_star = cp.Variable()

    # Objective function
    objective = cp.Minimize(cp_n.T @ p_n + calpha_n.T @ alpha_n + cbeta_n.T @ beta_n)
    # Constraints
    constraints = []
    constraints.append(sum(p_n) + w_bar >= d)
    constraints.append(1- sum(alpha_n) == 0)
    constraints.append(1- sum(beta_n) == 0)
    constraints.append(p_n >= 0)
    constraints.append(alpha_n >= 0)
    constraints.append(beta_n >= 0)
    constraints.append(p_n - p_n_max + alpha_n*inv_phi_eps*w_sigma <= 0)
    # SOC constraints
    soc_constrainst = []
    soc_constraints.append(alpha_n >= 0)

    # Solve problem
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    # Print results
    print("\nThe optimal value is", problem.value)
    print("A solution p_n is")
    print(p_n.value)
    print("A solution alpha_n is")
    print(alpha_n.value)
    print("The demand constraint dual solution is")
    print(problem.constraints[0].dual_value)
    print("The reserve constraint dual solution is")
    print(problem.constraints[1].dual_value)