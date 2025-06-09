import pandas as pd
import numpy as np
import subprocess
import os
import joblib
import shutil

# Assuming these are in your project directory
from merge_flow import *
from simulate_flow import *
from utils import * # Assuming choose_label is here

# Load the pre-trained model
model = joblib.load('RandomForest400IntPortCIC1718-2.pkl')

# Define input and output directories
data_in_dir = 'data/in/'
data_out_dir = 'data/out/'

# Define file names and paths
file_name = 'ftpbrute-kali' # You can change this or make it an input
pcap_file_path = f'{data_in_dir}{file_name}.pcap'
index_file_path = f'{data_out_dir}{file_name}_index.csv'
flow_file_path = f'{data_out_dir}{file_name}_ISCX.csv'
prediction_file_path = f'{data_out_dir}{file_name}_prediction.csv'
prompt_file_path =f'{data_out_dir}{file_name}_prompt.csv'

# Move the pcap file to the input directory if it's not already there
try:
    shutil.move(f'{file_name}.pcap', f'{data_in_dir}')
except Exception as e:
    print(f"Could not move pcap file (might already be there): {e}")

# Run CICFlowMeter to generate flow features
# Ensure the CICFlowMeter executable has execute permissions (e.g., chmod +x CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter)
print("Running CICFlowMeter to generate flow features...")
subprocess.run("CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter", check=True)
print("CICFlowMeter finished.")

# Simulate flow with packet indices
print("Extracting flows and simulating packet indices...")
simulated_flows_df, total_packets = extract_flows_from_pcap(pcap_file_path)
simulated_flows_df['Total_Packets'] = total_packets # Keep total packets info
simulated_flows_df.to_csv(index_file_path, index=False)
print(f"Simulated flows extracted. Total packets: {total_packets}")

# Merge packet indices flow with original flow
print("Merging simulated and original flow data...")
simulated_df, _ = read_flows_to_dataframe(index_file_path, is_simulated_output=True)
original_df, _ = read_flows_to_dataframe(flow_file_path, is_simulated_output=False)

merged_df = merge_flows_and_return_dataframe(simulated_df, original_df)
merged_df['Total_Packets'] = total_packets # Ensure total_packets is consistent after merge
merged_df.to_csv(index_file_path, index=False)
print("Flow data merged successfully.")

# --- START OF PREDICTION PIPELINE ---

## 1. Extract Flow-level Metadata

# Define which columns from the merged_df (original names) we want to carry through
# to the final packet-level prediction.
flow_metadata_cols = [
    'Simulated Packet Indices',
    'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol',
    'Fwd Seg Size Min', 'Init Fwd Win Byts', 'Bwd Pkts/s', 'Flow IAT Max',
    'Flow Duration', 'Pkt Len Mean', 'Flow Pkts/s', 'Fwd Header Len',
    'TotLen Fwd Pkts', 'Pkt Size Avg', 'Init Bwd Win Byts', 'Flow IAT Mean',
    'Subflow Fwd Byts', 'Bwd Pkt Len Mean', 'Bwd Header Len',
    'Bwd Seg Size Avg', 'PSH Flag Cnt', 'Flow Byts/s', 'Fwd Pkts/s'
]

# Create a DataFrame containing only the metadata columns
flow_metadata_df = merged_df[flow_metadata_cols].copy()

## 2. Preprocess Data for Model Prediction

# Preprocess the main DataFrame for model prediction (e.g., handle missing values, map ports, rename columns)
df = preprocess_dataframe(merged_df)

# Retrieve total_packets from the preprocessed DataFrame, then drop the column
total_packets = df['Total_Packets'].iloc[0]
df = df.drop(columns=['Total_Packets'])

# Store the index of `df` to align predictions back to the original flow entries
original_flow_indices = df.index

# Explicitly drop 'Simulated_Packet_Indices' (the name after preprocessing)
# as the model was not trained on this feature.
if 'Simulated_Packet_Indices' in df.columns:
    df_for_prediction = df.drop(columns=['Simulated_Packet_Indices'])
else:
    df_for_prediction = df.copy() # If it's already gone, just copy

## 3. Perform Prediction (per-flow prediction)

print("Performing flow-level predictions...")
flow_predictions = model.predict(df_for_prediction)
print("Predictions complete.")

## 4. Link Predictions with Flow Metadata

# Align predictions back to the original flow metadata using their shared index
df_flow_level_result = flow_metadata_df.loc[original_flow_indices].copy()
df_flow_level_result['Label'] = flow_predictions

## 5. Explode Packet Indices

# Explode 'Simulated Packet Indices' to create one row per packet, duplicating flow features and labels
df_prediction_expanded = df_flow_level_result.explode('Simulated Packet Indices').reset_index(drop=True)

# Rename the exploded column for clarity
df_prediction_expanded = df_prediction_expanded.rename(columns={'Simulated Packet Indices': 'Packet_Indices'})

# Convert Packet_Indices to integer type
df_prediction_expanded['Packet_Indices'] = df_prediction_expanded['Packet_Indices'].astype(int)

## 6. Group by Packet Indices and Aggregate Features

# Group by packet index and aggregate features to resolve multiple flows per packet
final_df_prediction = df_prediction_expanded.groupby('Packet_Indices').agg(
    Label=('Label', lambda x: choose_label(list(x))), # Apply choose_label for the final packet label
    Source_IP=('Src IP', 'first'),
    Destination_IP=('Dst IP', 'first'),
    Source_Port=('Src Port', 'first'),
    Destination_Port=('Dst Port', 'first'),
    Protocol=('Protocol', 'first'),
    Fwd_Seg_Size_Min=('Fwd Seg Size Min', 'first'),
    Init_Fwd_Win_Byts=('Init Fwd Win Byts', 'first'),
    Bwd_Pkts_s=('Bwd Pkts/s', 'first'),
    Flow_IAT_Max=('Flow IAT Max', 'first'),
    Flow_Duration=('Flow Duration', 'first'),
    Pkt_Len_Mean=('Pkt Len Mean', 'first'),
    Flow_Pkts_s=('Flow Pkts/s', 'first'),
    Fwd_Header_Len=('Fwd Header Len', 'first'),
    TotLen_Fwd_Pkts=('TotLen Fwd Pkts', 'first'),
    Pkt_Size_Avg=('Pkt Size Avg', 'first'),
    Init_Bwd_Win_Byts=('Init Bwd Win Byts', 'first'),
    Flow_IAT_Mean=('Flow IAT Mean', 'first'),
    Subflow_Fwd_Byts=('Subflow Fwd Byts', 'first'),
    Bwd_Pkt_Len_Mean=('Bwd Pkt Len Mean', 'first'),
    Bwd_Header_Len=('Bwd Header Len', 'first'),
    Bwd_Seg_Size_Avg=('Bwd Seg Size Avg', 'first'),
    PSH_Flag_Cnt=('PSH Flag Cnt', 'first'),
    Flow_Byts_s=('Flow Byts/s', 'first'),
    Fwd_Pkts_s=('Fwd Pkts/s', 'first')
).reset_index()

# Convert appropriate columns to nullable integer type ('Int64') for columns that are counts/lengths/IDs
for col in [
    'Source_Port', 'Destination_Port', 'Protocol',
    'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Flow_Duration',
    'Fwd_Header_Len', 'TotLen_Fwd_Pkts', 'Init_Bwd_Win_Byts',
    'Subflow_Fwd_Byts', 'Bwd_Header_Len', 'PSH_Flag_Cnt'
]:
    final_df_prediction[col] = final_df_prediction[col].astype('Int64')

## 7. Adjust Packet Indices to be 1-based

# Packet indices from pcap are 0-based, so adjust to 1-based for display
final_df_prediction['Packet_Indices'] = final_df_prediction['Packet_Indices'] + 1
final_df_prediction['Packet_Indices'] = final_df_prediction['Packet_Indices'].astype(int)

## 8. Handle Missing Indices and Add Features

# --- DEBUGGING PRINTS for missing_indices calculation ---
print(f"\nDEBUG: total_packets value: {total_packets}")
print(f"DEBUG: Columns in final_df_prediction before calculating missing_indices: {final_df_prediction.columns.tolist()}")
print(f"DEBUG: Head of final_df_prediction before calculating missing_indices:\n{final_df_prediction.head()}")
print(f"DEBUG: Shape of final_df_prediction before calculating missing_indices: {final_df_prediction.shape}")
# --- END DEBUGGING PRINTS ---

# Identify any packet indices that are missing from the processed flows
full_indices = set(range(1, total_packets + 1))
existing_indices = set(final_df_prediction['Packet_Indices'])
missing_indices = sorted(list(full_indices - existing_indices)) # Convert to list for DataFrame creation

# Create a DataFrame for missing packets, assigning 'Benign' and NaN for features
df_missing = pd.DataFrame({
    'Packet_Indices': missing_indices,
    'Label': ['Benign'] * len(missing_indices),
    'Source_IP': [np.nan] * len(missing_indices),
    'Destination_IP': [np.nan] * len(missing_indices),
    'Source_Port': [pd.NA] * len(missing_indices),
    'Destination_Port': [pd.NA] * len(missing_indices),
    'Protocol': [pd.NA] * len(missing_indices),
    'Fwd_Seg_Size_Min': [pd.NA] * len(missing_indices),
    'Init_Fwd_Win_Byts': [pd.NA] * len(missing_indices),
    'Bwd_Pkts_s': [np.nan] * len(missing_indices),
    'Flow_IAT_Max': [np.nan] * len(missing_indices),
    'Flow_Duration': [pd.NA] * len(missing_indices),
    'Pkt_Len_Mean': [np.nan] * len(missing_indices),
    'Flow_Pkts_s': [np.nan] * len(missing_indices),
    'Fwd_Header_Len': [pd.NA] * len(missing_indices),
    'TotLen_Fwd_Pkts': [pd.NA] * len(missing_indices),
    'Pkt_Size_Avg': [np.nan] * len(missing_indices),
    'Init_Bwd_Win_Byts': [pd.NA] * len(missing_indices),
    'Flow_IAT_Mean': [np.nan] * len(missing_indices),
    'Subflow_Fwd_Byts': [pd.NA] * len(missing_indices),
    'Bwd_Pkt_Len_Mean': [np.nan] * len(missing_indices),
    'Bwd_Header_Len': [pd.NA] * len(missing_indices),
    'Bwd_Seg_Size_Avg': [np.nan] * len(missing_indices),
    'PSH_Flag_Cnt': [pd.NA] * len(missing_indices),
    'Flow_Byts_s': [np.nan] * len(missing_indices),
    'Fwd_Pkts_s': [np.nan] * len(missing_indices),
})

# Explicitly set nullable integer types for df_missing columns to match final_df_prediction
for col in [
    'Source_Port', 'Destination_Port', 'Protocol',
    'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Flow_Duration',
    'Fwd_Header_Len', 'TotLen_Fwd_Pkts', 'Init_Bwd_Win_Byts',
    'Subflow_Fwd_Byts', 'Bwd_Header_Len', 'PSH_Flag_Cnt'
]:
    df_missing[col] = df_missing[col].astype('Int64')

df_missing['Packet_Indices'] = df_missing['Packet_Indices'].astype(int)

## 9. Ensure Consistent Column Order

# Define the final desired order of all columns in the output DataFrame
final_columns_order = [
    'Packet_Indices', 'Label', 'Source_IP', 'Destination_IP',
    'Source_Port', 'Destination_Port', 'Protocol',
    'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Bwd_Pkts_s', 'Flow_IAT_Max',
    'Flow_Duration', 'Pkt_Len_Mean', 'Flow_Pkts_s', 'Fwd_Header_Len',
    'TotLen_Fwd_Pkts', 'Pkt_Size_Avg', 'Init_Bwd_Win_Byts', 'Flow_IAT_Mean',
    'Subflow_Fwd_Byts', 'Bwd_Pkt_Len_Mean', 'Bwd_Header_Len',
    'Bwd_Seg_Size_Avg', 'PSH_Flag_Cnt', 'Flow_Byts_s', 'Fwd_Pkts_s'
]
final_df_prediction = final_df_prediction.reindex(columns=final_columns_order)
df_missing = df_missing.reindex(columns=final_columns_order)

## 10. Concatenate and Sort Final DataFrame

# Combine the predicted packets with the missing packets
final_df_prediction = pd.concat([final_df_prediction, df_missing], ignore_index=True)

# Sort by Packet_Indices and reset index for a clean output
final_df_prediction.sort_values(by='Packet_Indices', inplace=True)
final_df_prediction.reset_index(drop=True, inplace=True)

# Save the final predictions to a CSV file
final_df_prediction.to_csv(prompt_file_path, index=False)

columns_to_keep = [
    'Packet_Indices',
    'Label',
    'Source_IP',
    'Destination_IP',
    'Source_Port',
    'Destination_Port',
    'Protocol',
]

final_df_prediction = final_df_prediction[columns_to_keep]
final_df_prediction.to_csv(prediction_file_path, index=False)


# Print final results summary
print('----------------------------------------')
print(f'Total packets processed: {total_packets}')
print("\nFinal DataFrame Columns:")
print(final_df_prediction.columns.tolist())
print("\nFirst 5 rows of the final DataFrame:")
print(final_df_prediction.head())
print(f"\nFinal DataFrame shape: {final_df_prediction.shape}")
print('----------------------------------------')

# Move the pcap file back to its original location (optional)
try:
    shutil.move(f'{data_in_dir}{file_name}.pcap','.')
except Exception as e:
    print(f"Could not move pcap file back: {e}")