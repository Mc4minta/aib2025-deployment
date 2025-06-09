import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import requests
import joblib
import os
import shutil # For moving files
# add something
# Assuming these are in your project directory
# Make sure these files (pipeline.py, merge_flow.py, simulate_flow.py, utils.py)
# are accessible in the same directory as your Streamlit app.
# And ensure they contain the functions expected by your pipeline.
from pipeline import * # This is where your preprocess_dataframe and other pipeline steps might be
from merge_flow import *
from simulate_flow import *
from utils import * # This is where choose_label is expected to be


def display_setup_logs():
    # CICFlowMeter setup
    with st.status("Setting up CICFlowMeter-3.0...",expanded=True, state="running") as status:
        try:
            # install libpcap-dev library
            st.write(":arrow_down: Installing libpcap-dev...")
            subprocess.run(["sudo","apt-get", "install", "-y", "libpcap-dev"], check=True, capture_output=True, text=True)
            st.write(":white_check_mark: libpcap-dev installed.")
            
            # CICflowmeter download if not exist
            if not os.path.exists("CICFlowMeter-3.0"):
                # Download CICFlowMeter.zip from codeberg
                st.write(":arrow_down: Downloading CICFlowMeter-3.0.zip...")
                url = "https://codeberg.org/iortega/TCPDUMP_and_CICFlowMeter/archive/master:CICFlowMeters/CICFlowMeter-3.0.zip"
                response = requests.get(url, stream=True)
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                with open("CICFlowMeter-3.0.zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write(":white_check_mark: CICFlowMeter-3.0.zip downloaded.")
                
                # Extracting CICFlowMeter from codeberge
                st.write(":open_file_folder: Extracting CICFlowMeter-3.0...")
                subprocess.run(["unzip", "CICFlowMeter-3.0.zip", "-d", "CICFlowMeter-3.0"], check=True, capture_output=True, text=True)
                st.write(":white_check_mark: CICFlowMeter extracted.")
                
                # Setting executable permission 
                st.write(":wrench: Configuring executable permission...")
                subprocess.run(["chmod", "+x", "CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter"], check=True, capture_output=True, text=True)
                st.write(":white_check_mark: Permission configured")
                
                # Clearing unused zip file
                st.write(":wastebasket: Clearing .zip file...")
                subprocess.run(["rm","CICFlowMeter-3.0.zip"], check=True, capture_output=True, text=True)
                st.write(":white_check_mark: CICFlowMeter-3.0.zip Cleared")
            else:
                st.write(":information_source: CICFlowMeter-3.0 existed. Skipping...")
            
            # Creating data/in data/out directories
            st.write(":file_folder: Creating data/in data/out directories...")
            os.makedirs("data/in", exist_ok=True)
            os.makedirs("data/out", exist_ok=True)
            st.write(":white_check_mark: Directories created.")
            
            status.update(label=":white_check_mark: CICFlowMeter Setup Complete!", state="complete", expanded=False)
            
        except subprocess.CalledProcessError as e:
            st.error(f":x: Error during CICFlowMeter setup. Command '{e.cmd}' returned non-zero exit status {e.returncode}. Output: {e.stdout}\nError: {e.stderr}")
            status.update(label=":x: CICFlowMeter Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True # Set flag to true on error
            st.session_state.initial_setup_completed = False
            return False # Indicate failure
        except requests.exceptions.RequestException as e:
            st.error(f":x: Error downloading CICFlowMeter: {e}")
            status.update(label=":x: CICFlowMeter Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False
        except Exception as e:
            st.error(f":x: An unexpected error occurred during setup: {e}")
            status.update(label=":x: CICFlowMeter Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False
    
    # Classification Model setup
    with st.status("Setting up ML Model...",expanded=True, state="running") as status:
        try:
            # downloading model from hugging face if not exist
            if not os.path.exists("RandomForest400IntPortCIC1718-2.pkl"):
                st.write(":hugging_face: Downloading ML model...")
                model_url = "https://huggingface.co/Mc4minta/RandomForest400IntPortCIC1718/resolve/main/RandomForest400IntPortCIC1718-2.pkl"
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                with open("RandomForest400IntPortCIC1718-2.pkl", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write(":white_check_mark: ML Model downloaded.")

            # import model as using joblib
            st.write(":robot_face: Loading ML model...")
            model = joblib.load('RandomForest400IntPortCIC1718-2.pkl')
            st.session_state.model_state = model
            st.write(":white_check_mark: ML Model loaded successfully.")
            # st.info(model) # Removed this as it can be verbose
            
            status.update(label=":white_check_mark: ML Model Setup Complete", state="complete", expanded=False)
            return True # Indicate success
        except requests.exceptions.RequestException as e:
            st.error(f":x: Error downloading ML Model: {e}")
            status.update(label=":x: ML Model Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False
        except Exception as e:
            st.error(f"An unexpected error occurred during ML Model setup: {e}")
            status.update(label=":x: ML Model Setup Failed", state="error", expanded=True)
            st.session_state.setup_failed = True
            st.session_state.initial_setup_completed = False
            return False

@st.cache_data(show_spinner=False) # Cache setup to avoid rerunning if setup is already complete
def initial_setup_cached():
    """Wrapper to cache initial setup for efficiency."""
    if st.session_state.get('initial_setup_completed', False) and not st.session_state.get('setup_failed', False):
        return True # Already completed successfully
    
    # Reset flags before attempting setup
    st.session_state.initial_setup_completed = False
    st.session_state.setup_failed = False
    
    success = display_setup_logs()
    
    st.session_state.initial_setup_completed = success
    st.session_state.setup_failed = not success
    st.session_state.show_setup_logs = True # Keep this for visibility
    return success

def clear_uploaded_files():
    # remove pcap files
    if os.path.exists("data/in"):
        for filename in os.listdir("data/in"):
            filepath = os.path.join("data/in", filename)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except Exception as e:
                st.error(f"Failed to delete {filename} from data/in: {e}")
    # remove csv files
    if os.path.exists("data/out"):
        for filename in os.listdir("data/out"):
            filepath = os.path.join("data/out", filename)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except Exception as e:
                st.error(f"Failed to delete {filename} from data/out: {e}")
    
    # Clear model if it exists in session_state, to ensure it's reloaded if setup fails
    if 'model_state' in st.session_state:
        del st.session_state.model_state

# --- NEW: Function to run the entire prediction pipeline ---
def run_prediction_pipeline(uploaded_pcap_name, model):
    data_in_dir = 'data/in/'
    data_out_dir = 'data/out/'
    
    # Dynamically set file_name based on uploaded pcap
    file_name_without_ext = os.path.splitext(uploaded_pcap_name)[0]
    pcap_file_path = f'{data_in_dir}{uploaded_pcap_name}' # Use full uploaded name
    index_file_path = f'{data_out_dir}{file_name_without_ext}_index.csv'
    flow_file_path = f'{data_out_dir}{file_name_without_ext}_ISCX.csv'
    prediction_file_path = f'{data_out_dir}{file_name_without_ext}_prediction.csv'
    prompt_file_path = f'{data_out_dir}{file_name_without_ext}_prompt.csv'

    try:
        with st.status("Running analysis pipeline...", expanded=True) as pipeline_status:
            pipeline_status.write(":mag: Starting analysis...")
            
            # Step 1: Run CICFlowMeter to generate flow features
            pipeline_status.write(":rocket: Running CICFlowMeter to generate flow features...")
            subprocess.run("CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter", check=True, capture_output=True, text=True)
            pipeline_status.write(":white_check_mark: CICFlowMeter finished.")

            # Step 2: Simulate flow with packet indices
            pipeline_status.write(":chart_with_upwards_trend: Extracting flows and simulating packet indices...")
            # Ensure pcap_file_path passed here is the absolute path or correct relative path
            simulated_flows_df, total_packets = extract_flows_from_pcap(pcap_file_path)
            simulated_flows_df['Total_Packets'] = total_packets # Keep total packets info
            simulated_flows_df.to_csv(index_file_path, index=False)
            pipeline_status.write(f":white_check_mark: Simulated flows extracted. Total packets: {total_packets}")

            # Step 3: Merge packet indices flow with original flow
            pipeline_status.write(":handshake: Merging simulated and original flow data...")
            simulated_df, _ = read_flows_to_dataframe(index_file_path, is_simulated_output=True)
            original_df, _ = read_flows_to_dataframe(flow_file_path, is_simulated_output=False) # This file should be created by CICFlowMeter
            
            merged_df = merge_flows_and_return_dataframe(simulated_df, original_df)
            merged_df['Total_Packets'] = total_packets # Ensure total_packets is consistent after merge
            merged_df.to_csv(index_file_path, index=False) # Overwrite index file with merged data
            pipeline_status.write(":white_check_mark: Flow data merged successfully.")

            # --- START OF PREDICTION PIPELINE (integrated) ---
            # 1. Extract Flow-level Metadata
            flow_metadata_cols = [
                'Simulated Packet Indices',
                'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol',
                'Fwd Seg Size Min', 'Init Fwd Win Byts', 'Bwd Pkts/s', 'Flow IAT Max',
                'Flow Duration', 'Pkt Len Mean', 'Flow Pkts/s', 'Fwd Header Len',
                'TotLen Fwd Pkts', 'Pkt Size Avg', 'Init Bwd Win Byts', 'Flow IAT Mean',
                'Subflow Fwd Byts', 'Bwd Pkt Len Mean', 'Bwd Header Len',
                'Bwd Seg Size Avg', 'PSH Flag Cnt', 'Flow Byts/s', 'Fwd Pkts/s'
            ]
            flow_metadata_df = merged_df[flow_metadata_cols].copy()

            # 2. Preprocess Data for Model Prediction
            pipeline_status.write(":gear: Preprocessing data for prediction...")
            df = preprocess_dataframe(merged_df.copy()) # Pass a copy to avoid modifying original merged_df
            total_packets = df['Total_Packets'].iloc[0] # Get total_packets before dropping
            df = df.drop(columns=['Total_Packets'])
            original_flow_indices = df.index

            if 'Simulated_Packet_Indices' in df.columns:
                df_for_prediction = df.drop(columns=['Simulated_Packet_Indices'])
            else:
                df_for_prediction = df.copy()
            pipeline_status.write(":white_check_mark: Data preprocessed.")

            # 3. Perform Prediction (per-flow prediction)
            pipeline_status.write(":robot_face: Performing flow-level predictions...")
            flow_predictions = model.predict(df_for_prediction)
            pipeline_status.write(":white_check_mark: Predictions complete.")

            # 4. Link Predictions with Flow Metadata
            df_flow_level_result = flow_metadata_df.loc[original_flow_indices].copy()
            df_flow_level_result['Label'] = flow_predictions

            # 5. Explode Packet Indices
            df_prediction_expanded = df_flow_level_result.explode('Simulated Packet Indices').reset_index(drop=True)
            df_prediction_expanded = df_prediction_expanded.rename(columns={'Simulated Packet Indices': 'Packet_Indices'})
            df_prediction_expanded['Packet_Indices'] = df_prediction_expanded['Packet_Indices'].astype(int)

            # 6. Group by Packet Indices and Aggregate Features
            pipeline_status.write(":clipboard: Aggregating features by packet index...")
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
            pipeline_status.write(":white_check_mark: Aggregation complete.")

            # Convert appropriate columns to nullable integer type ('Int64')
            for col in [
                'Source_Port', 'Destination_Port', 'Protocol',
                'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Flow_Duration',
                'Fwd_Header_Len', 'TotLen_Fwd_Pkts', 'Init_Bwd_Win_Byts',
                'Subflow_Fwd_Byts', 'Bwd_Header_Len', 'PSH_Flag_Cnt'
            ]:
                final_df_prediction[col] = final_df_prediction[col].astype('Int64')

            # 7. Adjust Packet Indices to be 1-based
            final_df_prediction['Packet_Indices'] = final_df_prediction['Packet_Indices'] + 1
            final_df_prediction['Packet_Indices'] = final_df_prediction['Packet_Indices'].astype(int)

            # 8. Handle Missing Indices and Add Features
            full_indices = set(range(1, total_packets + 1))
            existing_indices = set(final_df_prediction['Packet_Indices'])
            missing_indices = sorted(list(full_indices - existing_indices))

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

            for col in [
                'Source_Port', 'Destination_Port', 'Protocol',
                'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Flow_Duration',
                'Fwd_Header_Len', 'TotLen_Fwd_Pkts', 'Init_Bwd_Win_Byts',
                'Subflow_Fwd_Byts', 'Bwd_Header_Len', 'PSH_Flag_Cnt'
            ]:
                df_missing[col] = df_missing[col].astype('Int64')
            df_missing['Packet_Indices'] = df_missing['Packet_Indices'].astype(int)

            # 9. Ensure Consistent Column Order
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

            # 10. Concatenate and Sort Final DataFrame
            final_df_prediction = pd.concat([final_df_prediction, df_missing], ignore_index=True)
            final_df_prediction.sort_values(by='Packet_Indices', inplace=True)
            final_df_prediction.reset_index(drop=True, inplace=True)

            final_df_prediction.to_csv(prompt_file_path, index=False) # Full details
            
            # Prepare the result for display
            columns_to_keep_for_display = [
                'Packet_Indices',
                'Label',
                'Source_IP',
                'Destination_IP',
                'Source_Port',
                'Destination_Port',
                'Protocol',
            ]
            display_df = final_df_prediction[columns_to_keep_for_display].copy()
            display_df.to_csv(prediction_file_path, index=False) # Reduced details for prediction output file

            pipeline_status.update(label=":white_check_mark: Analysis Complete!", state="complete", expanded=False)
            return display_df, total_packets # Return the result DataFrame and total packets
    except subprocess.CalledProcessError as e:
        st.error(f":x: Pipeline Error during command execution: {e.cmd} returned {e.returncode}. Output: {e.stdout}\nError: {e.stderr}")
        return None, 0
    except Exception as e:
        st.error(f":x: An unexpected error occurred during the analysis pipeline: {e}")
        return None, 0


def main():
    # Page config
    st.set_page_config(
        page_title="Malicious .PCAP File Classifier",
        page_icon=":peacock:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Initial state setup
    if 'initial_setup_completed' not in st.session_state:
        st.session_state.initial_setup_completed = False
    if 'setup_failed' not in st.session_state:
        st.session_state.setup_failed = False
    if 'show_setup_logs' not in st.session_state:
        st.session_state.show_setup_logs = False
    if 'proceed_clicked' not in st.session_state:
        st.session_state.proceed_clicked = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'file_uploaded_successfully' not in st.session_state:
        st.session_state.file_uploaded_successfully = False
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None
    if 'prediction_results_df' not in st.session_state:
        st.session_state.prediction_results_df = None
    if 'total_packets' not in st.session_state:
        st.session_state.total_packets = 0

    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #b5213b;'>
            AI Builders 2025
        </h1> 
        <h1 style='text-align: center;'>
            Malicious <span style='color: #074ec0;'>.pcap</span> File Classifier
        </h1>
        <h3 style='text-align: center;'><span style='color: #1abc9c;'>
            By: MiN - Vibrant Peacock</span> 🦚
        </h3>
        """, unsafe_allow_html=True)

    # CSS style for centered st.button
    st.markdown("""
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Setup Logic ---
    if not st.session_state.initial_setup_completed and not st.session_state.setup_failed:
        if st.button("Start Setup"):
            # This button initiates the cached setup function
            initial_setup_cached()
            st.rerun() # Rerun to reflect setup status

    # Logic handling StartSetup button failing
    elif st.session_state.setup_failed:
        st.warning("Setup failed. Please try again.")
        if st.button("Start Setup"):
            # This button initiates the cached setup function
            initial_setup_cached()
            st.rerun() # Rerun to reflect setup status

    # Show success and proceed after setup
    if st.session_state.initial_setup_completed and not st.session_state.proceed_clicked:
        st.success(":tada: Setup Completed")
        if st.button("Proceed"):
            st.session_state.show_setup_logs = False
            st.session_state.proceed_clicked = True
            st.rerun()

    # --- Main Application Logic (after setup and proceed) ---
    if st.session_state.initial_setup_completed and st.session_state.proceed_clicked:
        if not st.session_state.show_results:
            st.info(":file_folder: Browse your file here")

            uploaded_file = st.file_uploader(
                "Choose a PCAP file", accept_multiple_files=False, type=["pcap"]
            )

            if uploaded_file is not None:
                # Use a unique key for the button to prevent issues if file changes
                if st.button("Upload File and Analyze", key="upload_and_analyze_button"):
                    bytes_data = uploaded_file.read()
                    
                    # Ensure data/in directory exists before writing
                    os.makedirs('data/in', exist_ok=True)
                    
                    pcap_save_path = os.path.join("data/in", uploaded_file.name)
                    try:
                        with open(pcap_save_path, "wb") as f:
                            f.write(bytes_data)
                        mb_size = len(bytes_data) / (1024 * 1024)
                        st.session_state.uploaded_filename = uploaded_file.name
                        st.success(f":file_folder: {uploaded_file.name} size {mb_size:.2f} MB uploaded successfully. Starting analysis...")
                        st.session_state.file_uploaded_successfully = True
                        
                        # --- Run the new prediction pipeline ---
                        model = st.session_state.model_state # Get the loaded model from session state
                        if model:
                            results_df, total_p = run_prediction_pipeline(uploaded_file.name, model)
                            st.session_state.prediction_results_df = results_df
                            st.session_state.total_packets = total_p
                            st.session_state.show_results = True
                            st.rerun() # Rerun to display results
                        else:
                            st.error("ML model not loaded. Please ensure setup completed successfully.")
                            st.session_state.file_uploaded_successfully = False # Reset if analysis fails
                            st.session_state.show_results = False # Don't show results
                            clear_uploaded_files() # Clean up

                    except Exception as e:
                        st.error(f"Error uploading or initiating analysis: {e}")
                        st.session_state.file_uploaded_successfully = False
                        st.session_state.show_results = False
                        clear_uploaded_files() # Clean up failed upload/analysis

        # Display results once available
        if st.session_state.show_results and st.session_state.prediction_results_df is not None:
            uploaded_filename = st.session_state.uploaded_filename
            total_packets = st.session_state.total_packets
            st.subheader(f"Analysis Results for '{uploaded_filename}'")
            st.info(f"Total packets analyzed: **{total_packets}**")
            
            prediction_df = st.session_state.prediction_results_df
            
            # Initialize display toggles if not already set (for first time results are shown)
            if "show_df" not in st.session_state:
                st.session_state.show_df = False
            if "show_predictions" not in st.session_state:
                st.session_state.show_predictions = True # default

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Show Packet-Level Details"):
                    st.session_state.show_df = True
                    st.session_state.show_predictions = False

            with col2:
                if st.button("Show Prediction Summary"):
                    st.session_state.show_df = False
                    st.session_state.show_predictions = True

            # Display according to state
            if st.session_state.show_predictions:
                st.write("### Prediction Summary (Packet Count by Label)")
                if not prediction_df.empty:
                    prediction_counts = prediction_df['Label'].value_counts().sort_index()
                    st.bar_chart(prediction_counts)
                    st.dataframe(prediction_counts.reset_index().rename(columns={'index': 'Label', 'Label': 'Count'}), use_container_width=True)
                else:
                    st.warning("No predictions to display.")
            
            if st.session_state.show_df:
                st.write("### Packet-Level Prediction Details")
                st.dataframe(prediction_df, use_container_width=True)

            # Option to download the prediction CSV
            # Read the file again to ensure it's fresh for download
            output_prediction_file = os.path.join('data', 'out', os.path.splitext(uploaded_filename)[0] + '_prediction.csv')
            if os.path.exists(output_prediction_file):
                with open(output_prediction_file, "rb") as file:
                    btn = st.download_button(
                        label="Download Packet Predictions (CSV)",
                        data=file,
                        file_name=os.path.basename(output_prediction_file),
                        mime="text/csv",
                        help="Download the CSV file containing packet-level predictions."
                    )
            
            if st.button("Upload another file", key="upload_another_file_button"):
                clear_uploaded_files() # Clear all data/in and data/out
                st.session_state.show_results = False
                st.session_state.file_uploaded_successfully = False
                st.session_state.proceed_clicked = True # Remain in 'proceed' state
                st.session_state.prediction_results_df = None # Clear previous results
                st.session_state.total_packets = 0
                # Reset display toggles for the next upload
                st.session_state.show_df = False
                st.session_state.show_predictions = True
                st.rerun()

    # Handle if setup was successful but proceed wasn't clicked, and then user navigates away
    # or if some state gets messed up. This ensures the app doesn't get stuck.
    # No additional code needed here, the above logic handles it.

if __name__ == "__main__":
    main()