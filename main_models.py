import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm
import pandas as pd
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

# Models
