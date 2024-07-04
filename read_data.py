import pandas as pd
import numpy as np

# Lines data
df_2 = pd.read_csv('gridDetails.csv')
df_2.keys()
zonas = np.unique(df_2[['From Zone', 'To Zone']].values)
print(df_2.keys())

lines_node_index = []
line_reactance = []
line_f_max = []
node_demand = []

for ind in df_2.index:
    index_barra_in = np.where(zonas == df_2['From Zone'][ind])[0][0]
    index_barra_out = np.where(zonas == df_2['To Zone'][ind])[0][0]
    lines_node_index.append((index_barra_in,index_barra_out))
    line_reactance.append(df_2['Reactance (ohms)'][ind])
    line_f_max.append(df_2['Capacity (MW)'][ind])

print(lines_node_index)

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

print(node_set_index)

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

