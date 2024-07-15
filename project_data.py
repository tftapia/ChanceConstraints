import numpy as np
import pandas as pd

def init_parameters():
    system_param = dict()
    system_param["sys_epsilon"] = 0.05
    system_param["sys_cost_voll"] = 9000
    system_param["sys_cp_iterations"] = 100
    system_param["sys_epsilon_ext"] = 0.00005
    system_param["sys_math_error"] = 1e-8
    system_param["sys_const_error"] = 1e-5
    system_param["sys_MIPGap"] = 0.01
    system_param["sys_Timelimit"] = 600
    return system_param

def init_dictionaries(system_flag = False):
    string_path = "C:/Users/Tom/Desktop/Codes/ChanceConstraints/ChanceConstraints-1/"
    #string_path = "/Users/tftapia/Desktop/TT_Codes/ChanceContraints/ChanceConstraints/"

    system_dict = dict()
    system_dict["node_demand"] = dict()
    system_dict["node_gens"] = dict()
    system_dict["node_winds"] = dict()
    system_dict["node_lines_in"] = dict()
    system_dict["node_lines_out"] = dict()

    system_dict["gen_pmax"] = dict()
    system_dict["gen_c1"] = dict()
    system_dict["gen_c2"] = dict()
    system_dict["gen_calpha"] = dict()
    system_dict["gen_cbeta"] = dict()
    system_dict["gen_node"] = dict()

    system_dict["wind_node"] = dict()
    system_dict["wind_pmax"] = dict()
    system_dict["wind_factor"] = dict()
    system_dict["wind_mean"] = dict()
    system_dict["wind_std"] = dict()

    system_dict["line_reactance"] = dict()
    system_dict["line_fmax"] = dict()
    system_dict["lines_pairnodes"] = dict()

    system_dict["set_gen_index"] = []
    system_dict["set_node_index"] = []
    system_dict["set_lines_node_index"] = []

    if not system_flag:
        df_gen = pd.read_csv(string_path+'illustrative_gen.csv')
        df_node = pd.read_csv(string_path+'illustrative_node.csv')
        df_wind = pd.read_csv(string_path+'illustrative_wind.csv')

        for node_index in df_node.index:
            system_dict["node_demand"][node_index] = df_node["Demand (MW)"][node_index]

        for gen_index in df_gen.index:
            system_dict["gen_pmax"][gen_index] = df_gen["Capacity (MW)"][gen_index]
            system_dict["gen_c1"][gen_index] = df_gen["Dispatch Cost Coefficient a ($/MWh)"][gen_index]
            system_dict["gen_c2"][gen_index] = df_gen["Dispatch Cost Coefficient b ($/MW^2h)"][gen_index]
            system_dict["gen_calpha"][gen_index] = df_gen["Dispatch Cost Coefficient Alpha ($/%MWh)"][gen_index]
            system_dict["gen_cbeta"][gen_index] = df_gen["Dispatch Cost Coefficient Beta ($/%MWh)"][gen_index]
            system_dict["gen_node"][gen_index] = df_gen["Node Location"][gen_index]
            
        for wind_index in df_wind.index:
            system_dict["wind_node"][wind_index] = df_wind["Node Location"][wind_index]
            system_dict["wind_pmax"][wind_index] = df_wind["Capacity (MW)"][wind_index]
            system_dict["wind_factor"][wind_index] = df_wind["Factor"][wind_index]
            system_dict["wind_mean"][wind_index] = df_wind["Mean (MW)"][wind_index]
            system_dict["wind_std"][wind_index] = df_wind["Std (MW)"][wind_index]

        system_dict["set_node_index"].append(0)
        
        for node_index in system_dict["set_node_index"]:
            aux_list_gen = []
            for gen_index in df_gen.index:
                if system_dict["gen_node"][gen_index] == node_index:
                    aux_list_gen.append(gen_index)
            system_dict["node_gens"][node_index] = aux_list_gen
            system_dict["node_winds"][node_index] = [node_index]
            
        for node_index in system_dict["set_node_index"]:
            aux_list_in = []
            aux_list_out = []
            for (i,j) in system_dict["lines_pairnodes"].values():
                if i == node_index:
                    aux_list_in.append((i,j))
                elif j == node_index:
                    aux_list_out.append((i,j))
            system_dict["node_lines_in"][node_index] = aux_list_in
            system_dict["node_lines_out"][node_index] = aux_list_out

    else:
        df_lines = pd.read_csv(string_path+'extended_gridDetails.csv')
        df_gen = pd.read_csv(string_path+'extended_generator_data.csv')
        df_wind = pd.read_csv(string_path+'extended_wind_data.csv')
        zones = np.unique(df_lines[['From Zone','To Zone']].values)

        for lines_index in df_lines.index:
            index_node_in = np.where(zones == df_lines['From Zone'][lines_index])[0][0]
            index_node_out = np.where(zones == df_lines['To Zone'][lines_index])[0][0]
            
            system_dict["line_reactance"][lines_index] = df_lines['Reactance (per unit)'][lines_index]
            system_dict["line_fmax"][lines_index] = df_lines['Capacity (MW)'][lines_index]
            system_dict["lines_pairnodes"][lines_index] = (index_node_in,index_node_out)

        for node_index in range(len(zones)):
            df_node = pd.read_csv(string_path+'extended_'+zones[node_index]+'.csv')
            system_dict["node_demand"][node_index] = df_node[zones[node_index]].mean()

        for gen_index in df_gen.index:
            system_dict["gen_pmax"][gen_index] = 0.7*df_gen["Capacity (MW)"][gen_index]
            system_dict["gen_c1"][gen_index] = df_gen["Dispatch Cost Coefficient a ($/MWh)"][gen_index]
            system_dict["gen_c2"][gen_index] = 1.1*df_gen["Dispatch Cost Coefficient b ($/MW^2h)"][gen_index]
            system_dict["gen_calpha"][gen_index] = df_gen["Regular Reserve Cost Coefficient ($/MWh)"][gen_index]
            system_dict["gen_cbeta"][gen_index] = 1.1*df_gen["Extreme Reserve Cost Coefficient ($/MWh)"][gen_index]
            system_dict["gen_node"][gen_index] = np.where(zones == df_gen['Zone Location'][gen_index])[0][0]
        
        for wind_index in range(len(zones)): #create a wind generator in each zone
            system_dict["wind_node"][wind_index] = wind_index
            system_dict["wind_pmax"][wind_index] = df_wind["Capacity (MW)"][0]/len(zones)
            system_dict["wind_factor"][wind_index] = system_dict["node_demand"][wind_index]/sum(system_dict["node_demand"].values())
            system_dict["wind_mean"][wind_index] = df_wind["Mean (MW)"][0]/len(zones)
            system_dict["wind_std"][wind_index] = df_wind["Std (MW)"][0]/len(zones)
    
        for node_index in range(len(zones)):
            aux_list_gen = []
            for gen_index in df_gen.index:
                if system_dict["gen_node"][gen_index] == node_index:
                    aux_list_gen.append(gen_index)
            system_dict["node_gens"][node_index] = aux_list_gen
            system_dict["node_winds"][node_index] = [node_index]
        
        k_nodes = len(zones)
        system_dict["line_susceptance_matrix"] = np.zeros((k_nodes,k_nodes))
        system_dict["line_fmax_matrix"] = np.zeros((k_nodes,k_nodes))
        
        for lines_index in df_lines.index:
            index_i = system_dict["lines_pairnodes"][lines_index][0]
            index_j = system_dict["lines_pairnodes"][lines_index][1]
            system_dict["line_susceptance_matrix"][index_i,index_j] = -1/system_dict["line_reactance"][lines_index]
            system_dict["line_susceptance_matrix"][index_j,index_i] = -1/system_dict["line_reactance"][lines_index]
            system_dict["line_fmax_matrix"][index_i,index_j] = system_dict["line_fmax"][lines_index]
            system_dict["line_fmax_matrix"][index_j,index_i] = system_dict["line_fmax"][lines_index]
            system_dict["set_lines_node_index"].append((index_i,index_j))

        for node_index in range(len(zones)):
            aux_list_in = []
            aux_list_out = []
            for (i,j) in system_dict["lines_pairnodes"].values():
                if i == node_index:
                    aux_list_in.append((i,j))
                elif j == node_index:
                    aux_list_out.append((i,j))
            system_dict["node_lines_in"][node_index] = aux_list_in
            system_dict["node_lines_out"][node_index] = aux_list_out
            system_dict["set_node_index"].append(node_index)
    
    for gen_index in df_gen.index:
        system_dict["set_gen_index"].append(gen_index)

    return system_dict
