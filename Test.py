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
    d = np.array([70])
    cp_n = np.array([25,66])
    calpha_n = np.array([40,80])
    cbeta_n = np.array([80,100])
    p_n_max = np.array([100,100])
    epsilon = 0.05
    inv_phi_eps = norm.ppf(1-epsilon)
    epsilon_ext = 0.02
    inv_phi_ext = norm.ppf(1-epsilon_ext)

    w_bar = 5
    w_sigma = 5

    ## Generator 1
    g1_aux_abwp = cp.Variable(4)
    g1_aux_abwm = cp.Variable(4)
    g1_aux_ablp = cp.Variable(4)
    g1_aux_ablm = cp.Variable(4)

    ## Generator 2
    g2_aux_abwp = cp.Variable(4)
    g2_aux_abwm = cp.Variable(4)
    g2_aux_ablp = cp.Variable(4)
    g2_aux_ablm = cp.Variable(4)

    # (\alpha + \beta) \omega+ and (\alpha + \beta) \omega- 
    # (\alpha + \beta) \lambda+ and (\alpha + \beta) \lambda-
    A_ab = np.matrix([[2,0,0,0],[0,1,1,-1]])
    c_ab = np.array([0,1,1,1])

    p_n = cp.Variable(2)
    alpha_n = cp.Variable(2)
    beta_n = cp.Variable(2)
    w_star = cp.Variable()
    lambda_n1 = cp.Variable()
    lambda_n2 = cp.Variable()

    # Objective function
    objective = cp.Minimize(cp_n.T @ p_n + calpha_n.T @ alpha_n + cbeta_n.T @ beta_n)

    # Constraints
    constraints = []
    constraints.append(sum(p_n) + w_bar == d)
    constraints.append(1- sum(alpha_n) == 0)
    constraints.append(1- sum(beta_n) == 0)

    ## Generator 1
    constraints.append(cp.SOC(c_ab.T @ g1_aux_abwp, A_ab @ g1_aux_abwp))
    constraints.append(cp.SOC(c_ab.T @ g1_aux_abwm, A_ab @ g1_aux_abwm))
    constraints.append(cp.SOC(c_ab.T @ g1_aux_ablp, A_ab @ g1_aux_ablp))
    constraints.append(cp.SOC(c_ab.T @ g1_aux_ablm, A_ab @ g1_aux_ablm))

    constraints.append(alpha_n[0] == g1_aux_abwp[1])
    constraints.append(alpha_n[0] == g1_aux_abwm[1])
    constraints.append(alpha_n[0] == g1_aux_ablp[1])
    constraints.append(alpha_n[0] == g1_aux_ablm[1])

    constraints.append(beta_n[0] == g1_aux_abwp[2])
    constraints.append(beta_n[0] == g1_aux_abwm[2])
    constraints.append(beta_n[0] == g1_aux_ablp[2])
    constraints.append(beta_n[0] == g1_aux_ablm[2])

    constraints.append(g1_aux_abwp >= 0)
    constraints.append(g1_aux_abwm >= 0)
    constraints.append(g1_aux_ablp >= 0)
    constraints.append(g1_aux_ablm >= 0)

    constraints.append(w_star == g1_aux_abwp[3]- g1_aux_abwm[3])
    constraints.append(lambda_n1 == g1_aux_ablp[3]- g1_aux_ablm[3])
    constraints.append(w_star/w_sigma + g1_aux_ablp[0] - g1_aux_ablm[0] == 0)
    constraints.append(p_n[0] - p_n_max[0] - g1_aux_abwp[3] + g1_aux_abwm[3] == 0)

    ## Generator 2
    constraints.append(cp.SOC(c_ab.T @ g2_aux_abwp, A_ab @ g2_aux_abwp))
    constraints.append(cp.SOC(c_ab.T @ g2_aux_abwm, A_ab @ g2_aux_abwm))
    constraints.append(cp.SOC(c_ab.T @ g2_aux_ablp, A_ab @ g2_aux_ablp))
    constraints.append(cp.SOC(c_ab.T @ g2_aux_ablm, A_ab @ g2_aux_ablm))

    constraints.append(beta_n[1] == g2_aux_abwp[2])
    constraints.append(beta_n[1] == g2_aux_abwm[2])
    constraints.append(beta_n[1] == g2_aux_ablp[2])
    constraints.append(beta_n[1] == g2_aux_ablm[2])

    constraints.append(alpha_n[1] == g2_aux_abwp[1])
    constraints.append(alpha_n[1] == g2_aux_abwm[1])
    constraints.append(alpha_n[1] == g2_aux_ablp[1])
    constraints.append(alpha_n[1] == g2_aux_ablm[1])

    constraints.append(g2_aux_abwp >= 0)
    constraints.append(g2_aux_abwm >= 0)
    constraints.append(g2_aux_ablp >= 0)
    constraints.append(g2_aux_ablm >= 0)

    constraints.append(w_star == g2_aux_abwp[3]- g2_aux_abwm[3])
    constraints.append(lambda_n2 == g2_aux_ablp[3]- g2_aux_ablm[3])
    constraints.append(w_star/w_sigma + g2_aux_ablp[0] - g2_aux_ablm[0] == 0)
    constraints.append(p_n[1] - p_n_max[1] - g2_aux_abwp[3] + g2_aux_abwm[3] == 0)

    # Bounds
    constraints.append(w_star/np.power(w_sigma,2) - inv_phi_ext <= 0)
    constraints.append(p_n - p_n_max + alpha_n*inv_phi_eps*w_sigma <= 0)
    
    constraints.append(p_n >= 0)
    constraints.append(alpha_n >= 0)
    constraints.append(beta_n >= 0)

    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    print("\nThe optimal value is", problem.value)
    print("A solution w_star is")
    print(w_star.value)
    print("A solution p_n is")
    print(p_n.value)
    print("A solution alpha_n is")
    print(alpha_n.value)
    print("A solution beta_n is")
    print(beta_n.value)
    print("The demand constraint dual solution is")
    print(problem.constraints[0].dual_value)
    print("The normal reserve constraint dual solution is")
    print(problem.constraints[1].dual_value)
    print("The adversarial reserve constraint dual solution is")
    print(problem.constraints[2].dual_value)

hard_problem()
#cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i])