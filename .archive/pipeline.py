import pandas as pd
import numpy as np
import subprocess
import os
import joblib
import shutil

from merge_flow import *
from simulate_flow import *
from utils import *

model = joblib.load('RandomForest400IntPortCIC1718-2.pkl')

data_in_dir = 'data/in/'
data_out_dir = 'data/out/'

file_name = 'bruteforce' # input('Enter pcap file name: ')
pcap_file_path = f'{data_in_dir}{file_name}.pcap'
index_file_path = f'{data_out_dir}{file_name}_index.csv'
flow_file_path = f'{data_out_dir}{file_name}_ISCX.csv'
prediction_file_path = f'{data_out_dir}{file_name}_prediction.csv'

try:
    shutil.move(f'{file_name}.pcap', f'{data_in_dir}')
except Exception as e:
    print(e)

subprocess.run("CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter",check=True)

# simulate flow with packet indices
simulated_flows_df, total_packets = extract_flows_from_pcap(pcap_file_path)
simulated_flows_df['Total_Packets'] = total_packets
simulated_flows_df.to_csv(index_file_path, index=False)

# merge packet indices flow with original flow

# Read the data from both CSVs into DataFrames using the helper function
simulated_df,total_packets = read_flows_to_dataframe(index_file_path, is_simulated_output=True)
original_df,placeholder = read_flows_to_dataframe(flow_file_path, is_simulated_output=False)

# Merge the flows and get the resulting DataFrame
merged_df = merge_flows_and_return_dataframe(simulated_df, original_df)
merged_df['Total_Packets'] = total_packets
merged_df.to_csv(index_file_path, index=False)


# Start prediction pipeline
df = preprocess_dataframe(merged_df)
total_packets = df['Total_Packets'].iloc[0]
df = df.drop(columns=['Total_Packets'])

df_packet_index = df['Simulated_Packet_Indices']

df = df.drop(columns=['Simulated_Packet_Indices'])
df_prediction = model.predict(df)

df_prediction = pd.DataFrame({
    'Packet_Indices': df_packet_index,
    'Label': df_prediction
})

df_prediction = df_prediction.explode('Packet_Indices').reset_index(drop=True)

df_prediction = df_prediction.groupby('Packet_Indices')['Label'].apply(list).reset_index()
df_prediction['Label'] = df_prediction['Label'].apply(choose_label)
df_prediction['Packet_Indices'] = df_prediction['Packet_Indices'] + 1

full_indices = set(range(1, total_packets + 1))
existing_indices = set(df_prediction['Packet_Indices'])
missing_indices = sorted(full_indices - existing_indices)

df_missing = pd.DataFrame({
'Packet_Indices': missing_indices,
'Label': ['Benign'] * len(missing_indices),
})

df_prediction['Packet_Indices'] = df_prediction['Packet_Indices'].astype(int)
df_missing['Packet_Indices'] = df_missing['Packet_Indices'].astype(int)

df_prediction = pd.concat([df_prediction, df_missing], ignore_index=True)

df_prediction.sort_values(by='Packet_Indices', inplace=True)
df_prediction.reset_index(drop=True, inplace=True)

df_prediction.to_csv(prediction_file_path, index=False)



print('----------------------------------------')
print(f'total packet : {total_packets}')
print()
print(df_prediction.columns)
print(df_prediction.head())
print(df_prediction.shape)
print('----------------------------------------')

try:
    shutil.move(f'{data_in_dir}{file_name}.pcap','.')
except Exception as e:
    print(e)







