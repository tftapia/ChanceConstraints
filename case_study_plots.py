import numpy as np  
import matplotlib.pyplot as plt  
  

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

# Average cost and standard deviation for each model
def plot_1(color_in, list_avg_scheduled, list_avg_outschedule, list_std_scheduled, list_std_outschedule):
    fig, ax = plt.subplots(layout='constrained', figsize=(5, 2))
    color = color_in

    X = ['CC','LDT-CC','LDT-WCC'] 
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
    plt.savefig('Figure_0.pdf')
    #plt.show()


# Scheduled energy and reserve dispatch (simple example)
def plot_2(color_in, list_energy, list_rreserve, list_ereserve):
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
    sec.set_xticks([0.5, 3, 6], labels=['\n\nCC', '\n\nLDT-CC', '\n\nLDT-WCC'])
    sec.tick_params('x', length=0)

    # lines between the classes:
    sec2 = ax.secondary_xaxis(location=0)
    sec2.set_xticks([1.5, 4.5, 8.5], labels=[])
    sec2.tick_params('x', length=40, width=1)
    ax.set_xlim(-0.6, 7.6)

    ax.legend(bbox_to_anchor=(0.8, 1.25), ncols=3)
    #plt.show()
    plt.savefig('Figure_1.pdf')

def plot_3(color_in, list_energy, list_rreserve, list_ereserve):
    color = color_in
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
    plt.bar(X_axis, cc_y2, 0.2, color=color[1], label = "LDT-CC")
    plt.bar(X_axis + 0.25, cc_y3, 0.2, color=color[2], label = "LDT-WCC")

    #plt.ylim(0, 105)
    plt.ylabel("Energy/Reserve supply [MW]") 
    plt.xticks(X_axis, x) 
    plt.legend(bbox_to_anchor=(1.0, 0.9), ncols=3)
    plt.tight_layout()
    plt.show()

def plot_4(color_in, list_dispatch):
    color = color_in
    fig, ax = plt.subplots(layout='constrained', figsize=(7.5, 2.5))
    cc_y = list_dispatch

    X_axis = np.arange(len(cc_y)) 

    # Creating the bar plot
    plt.bar(X_axis, cc_y, color= color[0])

    # Adding title and labels
    plt.title('Sample Bar Plot')
    plt.xlabel('Categories')
    plt.ylabel('Values')

    # Showing the plot
    plt.show()


# Simple experiment plot
p2_gen1 = [40.0/105,    0.0, 39.9995/105, 0.0 , 0.0, 40.0/105, 0.0, 0.0]
p2_gen2 = [40.0654/105, 1.0, 44.9983/105, 0.0, 0.0001, 36.0561/105, 1.0, 0.0]
p2_gen3 = [24.9346/105, 0.0, 20.0022/105, 1.0, 0.9999, 28.9439/105, 0.0, 1.0 ]
plot_2(color_base,p2_gen1,p2_gen2,p2_gen3)

#plot_1(color_base_alt)

list_avg_scheduled   = [2914.7, 3030.07, 3024.79]
list_avg_outschedule = [2823.815564114271,2811.852092814494,2792.2037666072124]
list_std_scheduled   = [0,0,0]
list_std_outschedule = [121.89072106756136,124.14879451679485, 137.60425168597146]
plot_1(color_base, list_avg_scheduled, list_avg_outschedule, list_std_scheduled, list_std_outschedule)


p4_en = [1243.9436, 1234.8398, 880.8694, 684.6583, 620.1412, 611.8285, 140.2216, 2.6347, 243.8302, 243.3293, 149.7404, 144.3339, 81.9404, 79.9404, 47.8397, 47.4397, 0.0, 57.8858, 0.0001, 0.0, 0.0, 0.0, 0.0, 400.1152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 693.7383, 685.2374, 675.4486, 319.7813, 0.0047, 507.9383, 490.3384, 0.0001, 0.0047, 0.0001, 248.4675, 247.5662, 247.5687, 244.8181, 244.7483, 0.0001, 238.2297, 0.0001, 147.1062, 0.0001, 0.0001, 0.0001, 0.0001, 141.0485, 0.0002, 104.8484, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
p4_rr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1074, 0.0907, 0.0007, 0.0805, 0.0002, 0.0781, 0.0002, 0.0, 0.0724, 0.0411, 0.0299, 0.023, 0.0002, 0.0203, 0.001, 0.0, 0.0, 0.0, 0.0002, 0.0003, 0.0, 0.0, 0.0003, 0.0003, 0.0002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0002, 0.0, 0.0033, 0.0002, 0.0002, 0.0002, 0.0002, 0.0004, 0.0, 0.0002, 0.0, 0.0004, 0.0012, 0.0003, 0.0004, 0.0003, 0.0498, 0.0544, 0.0587, 0.0403, 0.0264, 0.0315, 0.0315, 0.0149, 0.0095, 0.0269, 0.036, 0.027, 0.016, 0.0223]
p4_rr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0422, 0.0674, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0023, 0.0008, 0.1003, 0.0014, 0.0792, 0.0006, 0.0742, 0.0, 0.0006, 0.002, 0.0008, 0.0008, 0.0212, 0.0008, 0.0138, 0.0, 0.0, 0.0, 0.0428, 0.0939, 0.0, 0.0, 0.0816, 0.0492, 0.0482, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0434, 0.0, 0.0399, 0.001, 0.027, 0.027, 0.027, 0.0269, 0.0, 0.0256, 0.0, 0.0186, 0.0068, 0.0076, 0.0076, 0.0075, 0.0004, 0.0004, 0.0007, 0.0008, 0.001, 0.0005, 0.0005, 0.0007, 0.0005, 0.0005, 0.0005, 0.0004, 0.0023, 0.0005]
plot_4(color_base,p4_rr)
