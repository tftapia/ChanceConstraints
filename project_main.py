import numpy as np
import gurobipy as gb
from gurobipy import GRB
from scipy.stats import norm
import pandas as pd

from project_data import init_dictionaries, init_parameters
from project_models import ED_standard_CC_network, ED_LDT_CC_network, ED_WCC_network, ED_LDT_WCC_network, ED_LDT_CC_price, test_scenarios
from project_aux import gen_profit, zone_dispatch, zone_profit
from project_plot import *

from project_models import ED_LDT_CC_alternative, ED_LDT_CC_alternative_price, ED_LDT_WCC_alternative

test_list = ["CC","LDT-WCC","LDT-CC"] # options ["CC", "WCC", "LDT-WCC", "LDT-CC"]
developer_mode = False
system_mode = False
scenario_mode = True
plot_mode = True
if system_mode:
    scenarios_max = 3000
    seed = 4 
else: 
    scenarios_max = 3000
    seed = 4

param = init_parameters()
data = init_dictionaries(system_mode)

while 0:
    # Solve
    solution_model = dict()
    if "CC" in test_list:
        solution_model["CC"] = ED_standard_CC_network(data, param,developer_mode)
    if "WCC" in test_list:
        solution_model["WCC"] = ED_WCC_network(data, param, developer_mode)
    if "LDT-CC" in test_list:
        solution_model["LDT-CC"] = ED_LDT_CC_network(data, param, developer_mode)
        solution_model["LDT-CC-price"] = ED_LDT_CC_price(data, param, solution_model["LDT-CC"]["w_opt"], developer_mode)
    if "LDT-CC" in test_list and "LDT-WCC" in test_list:
        solution_model["LDT-WCC"] = ED_LDT_WCC_network(data, param, solution_model["LDT-CC"]["w_opt"], developer_mode)

    # print objective solutions
    if "CC" in test_list:
        print("The Standard CC model objetive value is {}".format(round(solution_model["CC"]["o_opt"],4)))
    if "WCC" in test_list:
        print("The WCC model objetive value is {}".format(round(solution_model["WCC"]["o_opt"],4)))
    if "LDT-CC" in test_list:
        print("The LDT-CC model objetive value is {}".format(round(solution_model["LDT-CC"]["o_opt"],4)))
    if "LDT-CC" in test_list and "LDT-WCC" in test_list:
        print("The LDT-WCC model objetive value is {}".format(round( solution_model["LDT-WCC"]["o_opt"],4)))

    # print price
    if "CC" in test_list:
        print("\nThe Standard CC pricing model objetive value is {}".format(round(solution_model["CC"]["o_opt"],4)))
        print("price vector", solution_model["CC"]["p_price"], solution_model["CC"]["a_price"])
        print("dispatch vector", solution_model["CC"]["p_opt"], solution_model["CC"]["a_opt"])

    if "WCC" in test_list:
        print("The WCC pricing model objetive value is {}".format(round(solution_model["WCC"]["o_opt"],4)))
        print("price vector",solution_model["WCC"]["p_price"], solution_model["WCC"]["a_price"])
        print("dispatch vector", solution_model["WCC"]["p_opt"], solution_model["WCC"]["a_opt"])

    if "LDT-CC" in test_list:
        print("The LDT-CC pricing model objetive value is {}".format(round(solution_model["LDT-CC-price"]["o_opt"],4)))
        print("price vector",solution_model["LDT-CC-price"]["p_price"], solution_model["LDT-CC-price"]["a_price"], solution_model["LDT-CC-price"]["b_price"])
        print("dispatch vector", solution_model["LDT-CC"]["p_opt"], solution_model["LDT-CC"]["a_opt"], solution_model["LDT-CC"]["b_opt"])
        
        #a = gen_profit(data,solution_LDT_CC_price)
        #print(a["b_revenue"][2], a["b_cost"][2], a["b_profit"][2])
        #b = zone_dispatch(data,solution_LDT_CC_price)
        #print(b["b_zone"])
        #c = zone_profit(data,solution_LDT_CC_price)
        #print(c["p_profit_zone"])

    if "LDT-CC" in test_list and "LDT-WCC" in test_list:
        print("The LDT-WCC pricing model objetive value is {}".format(round(solution_model["LDT-WCC"]["o_opt"],4)))
        print("price vector",solution_model["LDT-WCC"]["p_price"], solution_model["LDT-WCC"]["a_price"], solution_model["LDT-WCC"]["b_price"])
        print("dispatch vector", solution_model["LDT-WCC"]["p_opt"], solution_model["LDT-WCC"]["a_opt"], solution_model["LDT-WCC"]["b_opt"])

    if scenario_mode:
        # print scenario objective values
        scenario_results = test_scenarios(data, param, solution_model, scenarios_max, test_list, seed)
        if "CC" in test_list:
            print("\nThe Standard CC model objective {}. Scenarios mean {}, std {}, max {}, min {}".format(solution_model["CC"]["o_opt"],scenario_results["CC"]["mean"],scenario_results["CC"]["std"],scenario_results["CC"]["max"],scenario_results["CC"]["min"]))
        if "WCC" in test_list:
            print("The WCC model objective {}. Scenarios mean {}, std {}, max {}, min {}".format(solution_model["WCC"],scenario_results["WCC"]["mean"],scenario_results["WCC"]["std"],scenario_results["WCC"]["max"],scenario_results["WCC"]["min"]))
        if "LDT-CC" in test_list:
            print("The LDT-CC model objective {}. Scenarios mean {}, std {}, max {}, min {}".format(solution_model["LDT-CC"]["o_opt"],scenario_results["LDT-CC"]["mean"],scenario_results["LDT-CC"]["std"],scenario_results["LDT-CC"]["max"],scenario_results["LDT-CC"]["min"]))
        if "LDT-CC" in test_list and "LDT-WCC" in test_list:
            print("The LDT-WCC model objective {}. Scenarios mean {}, std {}, max {}, min {}".format(solution_model["LDT-WCC"]["o_opt"], scenario_results["LDT-WCC"]["mean"],scenario_results["LDT-WCC"]["std"],scenario_results["LDT-WCC"]["max"],scenario_results["LDT-WCC"]["min"]))
        print("Omega mean {}, std {}, max {}, min {}".format(scenario_results["omega"]["mean"],scenario_results["omega"]["std"],scenario_results["omega"]["max"],scenario_results["omega"]["min"]))

    if plot_mode:
        if not system_mode:
            info_gen = illustrative_dispatch_prelist(solution_model, data, test_list)
            illustrative_dispatch_plot(color_base, info_gen[0], info_gen[1], info_gen[2], 'Figure_illustrative_5.pdf')
            if scenario_mode:
                info_models = scenario_comparison_prelist(solution_model, scenario_results)
                scenario_comparison(color_base, info_models["avg_cost_scheduled"], info_models["avg_cost_outscheduled"], info_models["std_cost_scheduled"], info_models["std_cost_outscheduled"], 'Figure_illustrative_1.pdf')
        else:
            info_zone = extensive_dispatch_prelist(solution_model, data, test_list)
            extensive_dispatch_plot(color_base, info_zone["CC"]["p_zone"], info_zone["LDT-WCC"]["p_zone"], info_zone["LDT-CC"]["p_zone"], 'Figure_extensive_5a.pdf', "energy")
            extensive_dispatch_plot(color_base, info_zone["CC"]["a_zone"], info_zone["LDT-WCC"]["a_zone"], info_zone["LDT-CC"]["a_zone"], 'Figure_extensive_5b.pdf', "reserve")
            extensive_dispatch_plot(color_base, info_zone["CC"]["b_zone"], info_zone["LDT-WCC"]["b_zone"], info_zone["LDT-CC"]["b_zone"], 'Figure_extensive_5c.pdf', "reserve")
            if scenario_mode:
                info_models = scenario_comparison_prelist(solution_model, scenario_results)
                scenario_comparison(color_base, info_models["avg_cost_scheduled"], info_models["avg_cost_outscheduled"], info_models["std_cost_scheduled"], info_models["std_cost_outscheduled"], 'Figure_extensive_1.pdf')

    break

#alternative
while 1:
    # Solve
    solution_model = dict()
    if "CC" in test_list:
        solution_model["CC"] = ED_standard_CC_network(data, param,developer_mode)
    if "WCC" in test_list:
        solution_model["WCC"] = ED_WCC_network(data, param, developer_mode)
    if "LDT-CC" in test_list:
        solution_model["LDT-CC"] = ED_LDT_CC_alternative(data, param, developer_mode)
        solution_model["LDT-CC-price"] = ED_LDT_CC_alternative_price(data, param, solution_model["LDT-CC"]["w_opt"], developer_mode)
    if "LDT-CC" in test_list and "LDT-WCC" in test_list:
        solution_model["LDT-WCC"] = ED_LDT_WCC_alternative(data, param, solution_model["LDT-CC"]["w_opt"], developer_mode)

    # print objective solutions
    if "CC" in test_list:
        print("The Standard CC model objetive value is {}".format(round(solution_model["CC"]["o_opt"],4)))
    if "WCC" in test_list:
        print("The WCC model objetive value is {}".format(round(solution_model["WCC"]["o_opt"],4)))
    if "LDT-CC" in test_list:
        print("The LDT-CC model objetive value is {}".format(round(solution_model["LDT-CC"]["o_opt"],4)))
    if "LDT-CC" in test_list and "LDT-WCC" in test_list:
        print("The LDT-WCC model objetive value is {}".format(round( solution_model["LDT-WCC"]["o_opt"],4)))

    # print price
    if "CC" in test_list:
        print("\nThe Standard CC pricing model objetive value is {}".format(round(solution_model["CC"]["o_opt"],4)))
        print("price vector", solution_model["CC"]["p_price"], solution_model["CC"]["a_price"])
        print("dispatch vector", solution_model["CC"]["p_opt"], solution_model["CC"]["a_opt"])

    if "WCC" in test_list:
        print("The WCC pricing model objetive value is {}".format(round(solution_model["WCC"]["o_opt"],4)))
        print("price vector",solution_model["WCC"]["p_price"], solution_model["WCC"]["a_price"])
        print("dispatch vector", solution_model["WCC"]["p_opt"], solution_model["WCC"]["a_opt"])

    if "LDT-CC" in test_list:
        print("The LDT-CC pricing model objetive value is {}".format(round(solution_model["LDT-CC-price"]["o_opt"],4)))
        print("price vector",solution_model["LDT-CC-price"]["p_price"], solution_model["LDT-CC-price"]["a_price"], solution_model["LDT-CC-price"]["b_price"])
        print("dispatch vector", solution_model["LDT-CC"]["p_opt"], solution_model["LDT-CC"]["a_opt"], solution_model["LDT-CC"]["b_opt"])
        
    if "LDT-CC" in test_list and "LDT-WCC" in test_list:
        print("The LDT-WCC pricing model objetive value is {}".format(round(solution_model["LDT-WCC"]["o_opt"],4)))
        print("price vector",solution_model["LDT-WCC"]["p_price"], solution_model["LDT-WCC"]["a_price"], solution_model["LDT-WCC"]["b_price"])
        print("dispatch vector", solution_model["LDT-WCC"]["p_opt"], solution_model["LDT-WCC"]["a_opt"], solution_model["LDT-WCC"]["b_opt"])

    if scenario_mode:
        # print scenario objective values
        scenario_results = test_scenarios(data, param, solution_model, scenarios_max, test_list, seed)
        if "CC" in test_list:
            print("\nThe Standard CC model objective {}. Scenarios mean {}, std {}, max {}, min {}".format(solution_model["CC"]["o_opt"],scenario_results["CC"]["mean"],scenario_results["CC"]["std"],scenario_results["CC"]["max"],scenario_results["CC"]["min"]))
        if "WCC" in test_list:
            print("The WCC model objective {}. Scenarios mean {}, std {}, max {}, min {}".format(solution_model["WCC"],scenario_results["WCC"]["mean"],scenario_results["WCC"]["std"],scenario_results["WCC"]["max"],scenario_results["WCC"]["min"]))
        if "LDT-CC" in test_list:
            print("The LDT-CC model objective {}. Scenarios mean {}, std {}, max {}, min {}".format(solution_model["LDT-CC"]["o_opt"],scenario_results["LDT-CC"]["mean"],scenario_results["LDT-CC"]["std"],scenario_results["LDT-CC"]["max"],scenario_results["LDT-CC"]["min"]))
        if "LDT-CC" in test_list and "LDT-WCC" in test_list:
            print("The LDT-WCC model objective {}. Scenarios mean {}, std {}, max {}, min {}".format(solution_model["LDT-WCC"]["o_opt"], scenario_results["LDT-WCC"]["mean"],scenario_results["LDT-WCC"]["std"],scenario_results["LDT-WCC"]["max"],scenario_results["LDT-WCC"]["min"]))
        print("Omega mean {}, std {}, max {}, min {}".format(scenario_results["omega"]["mean"],scenario_results["omega"]["std"],scenario_results["omega"]["max"],scenario_results["omega"]["min"]))

    if plot_mode:
        if not system_mode:
            info_gen = illustrative_dispatch_prelist(solution_model, data, test_list)
            illustrative_dispatch_plot(color_base, info_gen[0], info_gen[1], info_gen[2], 'Figure_illustrative_a5.pdf')
            if scenario_mode:
                info_models = scenario_comparison_prelist(solution_model, scenario_results)
                scenario_comparison(color_base, info_models["avg_cost_scheduled"], info_models["avg_cost_outscheduled"], info_models["std_cost_scheduled"], info_models["std_cost_outscheduled"], 'Figure_illustrative_a1.pdf')
        else:
            info_zone = extensive_dispatch_prelist(solution_model, data, test_list)
            extensive_dispatch_plot(color_base, info_zone["CC"]["p_zone"], info_zone["LDT-WCC"]["p_zone"], info_zone["LDT-CC"]["p_zone"], 'Figure_extensive_a5a.pdf', "energy")
            extensive_dispatch_plot(color_base, info_zone["CC"]["a_zone"], info_zone["LDT-WCC"]["a_zone"], info_zone["LDT-CC"]["a_zone"], 'Figure_extensive_a5b.pdf', "reserve")
            extensive_dispatch_plot(color_base, info_zone["CC"]["b_zone"], info_zone["LDT-WCC"]["b_zone"], info_zone["LDT-CC"]["b_zone"], 'Figure_extensive_a5c.pdf', "reserve")
            if scenario_mode:
                info_models = scenario_comparison_prelist(solution_model, scenario_results)
                scenario_comparison(color_base, info_models["avg_cost_scheduled"], info_models["avg_cost_outscheduled"], info_models["std_cost_scheduled"], info_models["std_cost_outscheduled"], 'Figure_extensive_a1.pdf')

    break
