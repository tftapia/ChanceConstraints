import numpy as np  
import matplotlib.pyplot as plt

from project_aux import zone_dispatch
  
color_red = (215/255, 38/255, 56/255) # (*)
color_blue = (63/255, 136/255, 197/255) # (*)
color_yellow = (244/255, 157/255, 55/255) # (*)

color_dark_blue = (62/255, 58/255, 83/255)
color_mid_blue = (31/255, 119/255, 180/255)
color_light_blue = (63/255, 136/255, 197/255)

color_red_1 = (201/255, 55/255, 56/255)   # brick
color_green_1 = (44/255, 160/255, 44/255)
color_orange_1 = (255/255, 127/255, 14/255)

color_green_2 = (17/255, 139/255, 108/255)
color_purple_2 = (60/255, 33/255, 147/255)
color_blue_2 = (26/255, 85/255, 138/255)

color_orange_2 = (255/255, 69/255, 0/255)
color_pink_2 = (236/255, 0/255, 140/255)
color_gray_2 = (155/255, 155/255, 155/255)

color_base = [color_blue, color_red_1, color_yellow]
color_alt = [color_mid_blue, color_green_1, color_orange_1]
color_base_alt = [color_blue, color_red, color_orange_1]

def illustrative_dispatch_prelist(solution_model, system_data, test_list):
    output_dict = dict()

    for n in system_data["set_gen_index"]:
        output_dict[n] = []
        for m in test_list:
            output_dict[n].append(solution_model[m]["p_opt"][n]*100/(system_data["node_demand"][0]-sum(system_data["wind_mean"].values())))
            output_dict[n].append(solution_model[m]["a_opt"][n]*100)
            if m != "CC":
                output_dict[n].append(solution_model[m]["b_opt"][n]*100)
    return output_dict

def illustrative_dispatch_plot(color_in, list_energy, list_rreserve, list_ereserve, name_fig):
    color = color_in
    fig, ax = plt.subplots(layout='constrained', figsize=(6, 2.5))

    def sum_list(list1, list2):
        return  list( map(np.add, list1, list2) )
    
    x = ['Energy', 'Reg. Res','Energy', 'Reg. Res', 'Ext. Res','Energy', 'Reg. Res', 'Ext. Res']
    cc_y1 = list_energy
    cc_y2 = list_rreserve
    cc_y3 = list_ereserve

    X_axis = np.arange(len(x)) 

    # plot bars in stack manner
    plt.bar(X_axis, cc_y1, 0.7,  color= color[0], label = "Gen 1")
    plt.bar(X_axis, cc_y2, 0.7,  bottom=cc_y1, color=color[1], label = "Gen 2")
    plt.bar(X_axis, cc_y3, 0.7,  bottom=sum_list(cc_y1, cc_y2), color=color[2], label = "Gen 3")


    # label the classes:
    plt.xticks(X_axis, x) 
    plt.ylabel("Energy/Reserve supply [%]") 
    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks([0.5, 3, 6], labels=['\n\nCC', '\n\nLDT-WCC', '\n\nLDT-CC'])
    sec.tick_params('x', length=0)

    # lines between the classes:
    sec2 = ax.secondary_xaxis(location=0)
    sec2.set_xticks([1.5, 4.5, 8.5], labels=[])
    sec2.tick_params('x', length=40, width=1)
    ax.set_xlim(-0.6, 7.6)

    ax.legend(bbox_to_anchor=(0.8, 1.25), ncols=3)
    #plt.show()
    plt.savefig(name_fig)

def extensive_dispatch_prelist(solution_model, system_data, test_list):
    output_dict = dict()

    for m in test_list:
        aux_dict = zone_dispatch(system_data, solution_model[m])
        output_dict[m] = dict()
        output_dict[m]["p_zone"] = []
        output_dict[m]["a_zone"] = []
        output_dict[m]["b_zone"] = []

        for i in system_data["set_node_index"]:
            output_dict[m]["p_zone"].append(aux_dict["p_zone"][i])
            output_dict[m]["a_zone"].append(aux_dict["a_zone"][i]*100)
            if m != "CC":
                output_dict[m]["b_zone"].append(aux_dict["b_zone"][i]*100)
            else:
                output_dict[m]["b_zone"].append(0)
    return output_dict

def extensive_dispatch_plot(color_in, list_energy, list_rreserve, list_ereserve, name_fig, str_type):
    color = color_in

    if str_type =="reserve":
        aux_string = "[%]"
    elif str_type =="energy":
        aux_string = "[MW]"

    def sum_list(list1, list2):
        return  list( map(np.add, list1, list2) )
            
    fig, ax = plt.subplots(layout='constrained', figsize=(7.5, 2.5))

    #x = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone_4', 'Zone 5', 'Zone 6', 'Zone 7', 'Zone 8']
    x = ['CT', 'ME', 'NEMASSBOST', 'NH', 'RI', 'SEMASS', 'VT', 'WCMASS']
    cc_y1 = list_energy
    cc_y2 = list_rreserve
    cc_y3 = list_ereserve

    X_axis = np.arange(len(x)) 

    # plot bars in stack manner
    plt.bar(X_axis - 0.25, cc_y1, 0.2,  color= color[0], label = "CC")
    plt.bar(X_axis, cc_y2, 0.2, color=color[1], label = "LDT-WCC")
    plt.bar(X_axis + 0.25, cc_y3, 0.2, color=color[2], label = "LDT-CC")

    if str_type =="reserve":
        plt.ylim(0, 105)

    plt.ylabel("Energy/Reserve supply "+aux_string) 
    plt.xticks(X_axis, x) 
    ax.legend(bbox_to_anchor=(0.75, 1.25), ncols=3)
    plt.tight_layout()
    plt.savefig(name_fig)
    

def scenario_comparison_prelist(solution_model, scenario_results):
    output_dict = dict()
    output_dict["avg_cost_scheduled"] = []
    output_dict["std_cost_scheduled"] = []
    output_dict["avg_cost_outscheduled"] = []
    output_dict["std_cost_outscheduled"] = []
    for m in ['CC','LDT-WCC','LDT-CC']:
        output_dict["avg_cost_scheduled"].append(solution_model[m]["o_opt"])
        output_dict["std_cost_scheduled"].append(0)
        output_dict["avg_cost_outscheduled"].append(scenario_results[m]["mean"])
        output_dict["std_cost_outscheduled"].append(scenario_results[m]["std"])

    return output_dict

# Average cost and standard deviation for each model
def scenario_comparison(color_in, list_avg_scheduled, list_avg_outschedule, list_std_scheduled, list_std_outschedule, name_fig):
    fig, ax = plt.subplots(layout='constrained', figsize=(5, 2))
    color = color_in

    X = ['CC','LDT-WCC','LDT-CC'] 
    schedule_av =  list_avg_scheduled
    out_av = list_avg_outschedule
    
    X_axis = np.arange(len(X)) 
    
    plt.bar(X_axis - 0.2, schedule_av, 0.4, label = 'Scheduled', color=color[0]) 
    plt.bar(X_axis + 0.2, out_av , 0.4, label = 'Out Schedule', color=color[2]) 

    schedule_std = list_std_scheduled
    out_std = list_std_outschedule

    plt.errorbar(X_axis - 0.2, schedule_av, yerr= schedule_std, fmt="o", color=color[1])
    plt.errorbar(X_axis + 0.2, out_av, yerr= out_std, fmt="o", color=color[1])

    plt.xticks(X_axis, X) 
    #plt.xlabel("Models") 
    plt.ylabel("Total cost [$]") 
    #plt.title("Number of Students in each group") 
    plt.legend(bbox_to_anchor=(0.8, 1.25), ncols=3)
    plt.savefig(name_fig)
    #ax.set_ylim([2500, 3800])
    #plt.show()