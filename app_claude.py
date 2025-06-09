import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import requests
import joblib
import os
import shutil
import google.generativeai as genai
import sys
import logging
from pathlib import Path
import time
import threading
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importing helper functions from separate modules
from merge_flow import read_flows_to_dataframe, merge_flows_and_return_dataframe
from simulate_flow import extract_flows_from_pcap
from utils import map_port, preprocess_dataframe, choose_label

# Configuration class
class Config:
    GEMINI_MODEL = "gemini-1.5-flash"
    CICFLOWMETER_URL = "https://codeberg.org/iortega/TCPDUMP_and_CICFlowMeter/archive/master:CICFlowMeters/CICFlowMeter-3.0.zip"
    MODEL_URL = "https://huggingface.co/Mc4minta/RandomForest400IntPortCIC1718/resolve/main/RandomForest400IntPortCIC1718-2.pkl"
    MAX_FILE_SIZE_MB = 100
    DATA_IN_DIR = Path("data/in")
    DATA_OUT_DIR = Path("data/out")
    CICFLOWMETER_DIR = Path("CICFlowMeter-3.0")
    MODEL_FILE = "RandomForest400IntPortCIC1718-2.pkl"
    CICFLOWMETER_BINARY = "CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter"

# Custom exceptions
class SetupError(Exception):
    """Raised when setup operations fail"""
    pass

class AnalysisError(Exception):
    """Raised when analysis operations fail"""
    pass

class FileOperationError(Exception):
    """Raised when file operations fail"""
    pass

# --- LLM Configuration Function ---
@st.cache_resource(show_spinner="Connecting to Gemini...")
def configure_gemini(api_key: str):
    """Configure Gemini API with proper error handling"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        # Test the model with a simple prompt
        test_response = model.generate_content("Hello")
        return model
    except Exception as e:
        logger.error(f"Error configuring Gemini: {e}")
        return None

# --- Utility Functions ---
def safe_subprocess_run(cmd: list, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run subprocess with proper error handling and timeout"""
    try:
        return subprocess.run(
            cmd, 
            check=check, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise SetupError(f"Command timed out after {timeout} seconds: {' '.join(cmd)}")
    except subprocess.CalledProcessError as e:
        raise SetupError(f"Command failed: {' '.join(cmd)}\nReturn code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}")
    except Exception as e:
        raise SetupError(f"Unexpected error running command {' '.join(cmd)}: {e}")

def safe_download_file(url: str, filename: str, chunk_size: int = 8192) -> None:
    """Download file with proper error handling and progress tracking"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        st.write(f"Downloaded: {progress:.1f}%")
        
        logger.info(f"Successfully downloaded {filename}")
    except requests.exceptions.RequestException as e:
        raise SetupError(f"Failed to download {url}: {e}")
    except Exception as e:
        raise SetupError(f"Unexpected error downloading {filename}: {e}")

def safe_create_directories(directories: list) -> None:
    """Create directories with proper error handling"""
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            raise FileOperationError(f"Failed to create directory {directory}: {e}")

def safe_file_cleanup(directory: str, file_patterns: list = None) -> None:
    """Clean up files with proper error handling and locking"""
    if not os.path.exists(directory):
        return
    
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            try:
                if os.path.isfile(filepath):
                    # Check if file is in use (basic check)
                    if os.path.getsize(filepath) == 0:
                        continue
                    os.remove(filepath)
                    logger.info(f"Removed file: {filepath}")
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                    logger.info(f"Removed directory: {filepath}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not remove {filepath}: {e}")
                continue
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")

def validate_file_upload(file_data: bytes, filename: str) -> bool:
    """Validate uploaded file"""
    if not file_data:
        st.error("File is empty")
        return False
    
    file_size_mb = len(file_data) / (1024 * 1024)
    if file_size_mb > Config.MAX_FILE_SIZE_MB:
        st.error(f"File too large: {file_size_mb:.2f} MB (max: {Config.MAX_FILE_SIZE_MB} MB)")
        return False
    
    if not filename.lower().endswith('.pcap'):
        st.error("File must be a .pcap file")
        return False
    
    return True

# --- Setup Functions ---
def setup_cicflowmeter() -> bool:
    """Setup CICFlowMeter with comprehensive error handling"""
    try:
        # Install libpcap-dev library
        st.write("📦 Installing libpcap-dev...")
        safe_subprocess_run(["sudo", "apt-get", "update"])
        safe_subprocess_run(["sudo", "apt-get", "install", "-y", "libpcap-dev"])
        st.write("✅ libpcap-dev installed.")
        
        # Download CICFlowMeter if not exists
        if not Config.CICFLOWMETER_DIR.exists():
            st.write("📥 Downloading CICFlowMeter-3.0...")
            safe_download_file(Config.CICFLOWMETER_URL, "CICFlowMeter-3.0.zip")
            st.write("✅ CICFlowMeter-3.0.zip downloaded.")
            
            # Extract CICFlowMeter
            st.write("📂 Extracting CICFlowMeter-3.0...")
            safe_subprocess_run(["unzip", "-o", "CICFlowMeter-3.0.zip", "-d", str(Config.CICFLOWMETER_DIR)])
            st.write("✅ CICFlowMeter extracted.")
            
            # Verify extraction and set permissions
            binary_path = Path(Config.CICFLOWMETER_BINARY)
            if binary_path.exists():
                st.write("🔧 Configuring executable permission...")
                safe_subprocess_run(["chmod", "+x", str(binary_path)])
                st.write("✅ Permission configured")
            else:
                # Try to find the binary in different locations
                possible_paths = list(Config.CICFLOWMETER_DIR.rglob("CICFlowMeter"))
                if possible_paths:
                    actual_binary = possible_paths[0]
                    st.write(f"🔍 Found CICFlowMeter at: {actual_binary}")
                    safe_subprocess_run(["chmod", "+x", str(actual_binary)])
                    # Update the binary path in config
                    Config.CICFLOWMETER_BINARY = str(actual_binary)
                    st.write("✅ Permission configured")
                else:
                    raise SetupError("CICFlowMeter binary not found after extraction")
            
            # Clean up zip file
            st.write("🗑️ Cleaning up zip file...")
            os.remove("CICFlowMeter-3.0.zip")
            st.write("✅ Cleanup complete")
        else:
            st.write("ℹ️ CICFlowMeter-3.0 already exists. Skipping...")
            # Verify binary exists
            if not Path(Config.CICFLOWMETER_BINARY).exists():
                possible_paths = list(Config.CICFLOWMETER_DIR.rglob("CICFlowMeter"))
                if possible_paths:
                    Config.CICFLOWMETER_BINARY = str(possible_paths[0])
                else:
                    raise SetupError("CICFlowMeter binary not found")
        
        # Create data directories
        st.write("📁 Creating data directories...")
        safe_create_directories([str(Config.DATA_IN_DIR), str(Config.DATA_OUT_DIR)])
        st.write("✅ Directories created.")
        
        return True
        
    except SetupError as e:
        st.error(f"❌ CICFlowMeter setup failed: {e}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error during CICFlowMeter setup: {e}")
        return False

def setup_ml_model() -> Optional[Any]:
    """Setup ML model with proper error handling"""
    try:
        # Download model if not exists
        if not os.path.exists(Config.MODEL_FILE):
            st.write("🤗 Downloading ML model...")
            safe_download_file(Config.MODEL_URL, Config.MODEL_FILE)
            st.write("✅ ML Model downloaded.")
        
        # Load model
        st.write("🤖 Loading ML model...")
        model = joblib.load(Config.MODEL_FILE)
        st.write("✅ ML Model loaded successfully.")
        
        return model
        
    except Exception as e:
        st.error(f"❌ ML Model setup failed: {e}")
        return None

def display_setup_logs() -> bool:
    """Display setup logs with proper status tracking"""
    success = True
    
    # CICFlowMeter setup
    with st.status("Setting up CICFlowMeter-3.0...", expanded=True, state="running") as status:
        cicflowmeter_success = setup_cicflowmeter()
        if cicflowmeter_success:
            status.update(label="✅ CICFlowMeter Setup Complete!", state="complete", expanded=False)
        else:
            status.update(label="❌ CICFlowMeter Setup Failed", state="error", expanded=True)
            success = False
    
    # ML Model setup
    with st.status("Setting up ML Model...", expanded=True, state="running") as status:
        model = setup_ml_model()
        if model is not None:
            st.session_state.model_state = model
            status.update(label="✅ ML Model Setup Complete", state="complete", expanded=False)
        else:
            status.update(label="❌ ML Model Setup Failed", state="error", expanded=True)
            success = False
    
    return success

# Non-cached setup function to avoid session state issues
def run_initial_setup() -> bool:
    """Run initial setup without caching to avoid session state issues"""
    return display_setup_logs()

def clear_uploaded_files() -> None:
    """Clear uploaded files with proper error handling"""
    directories_to_clean = [str(Config.DATA_IN_DIR), str(Config.DATA_OUT_DIR)]
    
    for directory in directories_to_clean:
        safe_file_cleanup(directory)
    
    # Clear model state safely
    if 'model_state' in st.session_state:
        del st.session_state.model_state
        logger.info("Model state cleared from session")

# --- Prediction Pipeline ---
def run_prediction_pipeline(pcap_file_path: str, uploaded_pcap_name: str, model) -> Tuple[Optional[pd.DataFrame], int, Optional[str]]:
    """Run prediction pipeline with comprehensive error handling"""
    try:
        file_name_without_ext = os.path.splitext(uploaded_pcap_name)[0]
        index_file_path = Config.DATA_OUT_DIR / f'{file_name_without_ext}_index.csv'
        flow_file_path = Config.DATA_OUT_DIR / f'{file_name_without_ext}_ISCX.csv'
        prediction_file_path = Config.DATA_OUT_DIR / f'{file_name_without_ext}_prediction.csv'
        prompt_file_path = Config.DATA_OUT_DIR / f'{file_name_without_ext}_prompt.csv'

        with st.status("Running analysis pipeline...", expanded=True) as pipeline_status:
            pipeline_status.write("🔍 Starting analysis...")
            
            # Verify CICFlowMeter binary exists
            binary_path = Path(Config.CICFLOWMETER_BINARY)
            if not binary_path.exists():
                raise AnalysisError(f"CICFlowMeter binary not found at {binary_path}")
            
            # Step 1: Run CICFlowMeter
            pipeline_status.write("🚀 Running CICFlowMeter to generate flow features...")
            try:
                safe_subprocess_run([str(binary_path)], timeout=600)  # 10 minute timeout
                pipeline_status.write("✅ CICFlowMeter finished.")
            except SetupError as e:
                raise AnalysisError(f"CICFlowMeter execution failed: {e}")

            # Verify CICFlowMeter output
            if not flow_file_path.exists():
                raise AnalysisError(f"CICFlowMeter did not generate expected output file: {flow_file_path}")

            # Step 2: Extract flows and simulate packet indices
            pipeline_status.write("📊 Extracting flows and simulating packet indices...")
            try:
                simulated_flows_df, total_packets = extract_flows_from_pcap(pcap_file_path)
                if simulated_flows_df.empty:
                    raise AnalysisError("No flows extracted from PCAP file")
                
                simulated_flows_df['Total_Packets'] = total_packets
                simulated_flows_df.to_csv(index_file_path, index=False)
                pipeline_status.write(f"✅ Simulated flows extracted. Total packets: {total_packets}")
            except Exception as e:
                raise AnalysisError(f"Flow extraction failed: {e}")

            # Step 3: Merge flows
            pipeline_status.write("🤝 Merging flow data...")
            try:
                simulated_df, _ = read_flows_to_dataframe(str(index_file_path), is_simulated_output=True)
                original_df, _ = read_flows_to_dataframe(str(flow_file_path), is_simulated_output=False)
                
                merged_df = merge_flows_and_return_dataframe(simulated_df, original_df)
                merged_df['Total_Packets'] = total_packets
                merged_df.to_csv(index_file_path, index=False)
                pipeline_status.write("✅ Flow data merged successfully.")
            except Exception as e:
                raise AnalysisError(f"Flow merging failed: {e}")

            # Prediction pipeline
            pipeline_status.write("🔮 Starting prediction pipeline...")
            
            # Extract flow metadata
            flow_metadata_cols = [
                'Simulated Packet Indices', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol',
                'Fwd Seg Size Min', 'Init Fwd Win Byts', 'Bwd Pkts/s', 'Flow IAT Max',
                'Flow Duration', 'Pkt Len Mean', 'Flow Pkts/s', 'Fwd Header Len',
                'TotLen Fwd Pkts', 'Pkt Size Avg', 'Init Bwd Win Byts', 'Flow IAT Mean',
                'Subflow Fwd Byts', 'Bwd Pkt Len Mean', 'Bwd Header Len',
                'Bwd Seg Size Avg', 'PSH Flag Cnt', 'Flow Byts/s', 'Fwd Pkts/s'
            ]
            
            # Verify required columns exist
            missing_cols = [col for col in flow_metadata_cols if col not in merged_df.columns]
            if missing_cols:
                raise AnalysisError(f"Missing required columns: {missing_cols}")
            
            flow_metadata_df = merged_df[flow_metadata_cols].copy()

            # Preprocess data
            pipeline_status.write("⚙️ Preprocessing data for prediction...")
            df = preprocess_dataframe(merged_df.copy())
            total_packets = df['Total_Packets'].iloc[0] if 'Total_Packets' in df.columns and not df.empty else 0
            
            if 'Total_Packets' in df.columns:
                df = df.drop(columns=['Total_Packets'])
            
            original_flow_indices = df.index

            if 'Simulated_Packet_Indices' in df.columns:
                df_for_prediction = df.drop(columns=['Simulated_Packet_Indices'])
            else:
                df_for_prediction = df.copy()
            
            pipeline_status.write("✅ Data preprocessed.")

            # Perform predictions
            pipeline_status.write("🤖 Performing flow-level predictions...")
            try:
                flow_predictions = model.predict(df_for_prediction)
                pipeline_status.write("✅ Predictions complete.")
            except Exception as e:
                raise AnalysisError(f"Model prediction failed: {e}")

            # Process predictions
            df_flow_level_result = flow_metadata_df.loc[original_flow_indices].copy()
            df_flow_level_result['Label'] = flow_predictions

            # Explode packet indices
            df_prediction_expanded = df_flow_level_result.explode('Simulated Packet Indices').reset_index(drop=True)
            df_prediction_expanded = df_prediction_expanded.rename(columns={'Simulated Packet Indices': 'Packet_Indices'})
            df_prediction_expanded['Packet_Indices'] = pd.to_numeric(df_prediction_expanded['Packet_Indices'], errors='coerce')
            df_prediction_expanded = df_prediction_expanded.dropna(subset=['Packet_Indices'])
            df_prediction_expanded['Packet_Indices'] = df_prediction_expanded['Packet_Indices'].astype(int)

            # Aggregate features
            pipeline_status.write("📋 Aggregating features by packet index...")
            final_df_prediction = df_prediction_expanded.groupby('Packet_Indices').agg(
                Label=('Label', lambda x: choose_label(list(x))),
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
            
            pipeline_status.write("✅ Aggregation complete.")

            # Handle data types consistently
            integer_columns = [
                'Source_Port', 'Destination_Port', 'Protocol',
                'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Flow_Duration',
                'Fwd_Header_Len', 'TotLen_Fwd_Pkts', 'Init_Bwd_Win_Byts',
                'Subflow_Fwd_Byts', 'Bwd_Header_Len', 'PSH_Flag_Cnt'
            ]
            
            for col in integer_columns:
                if col in final_df_prediction.columns:
                    final_df_prediction[col] = pd.to_numeric(final_df_prediction[col], errors='coerce').astype('Int64')

            # Adjust packet indices to 1-based
            final_df_prediction['Packet_Indices'] = final_df_prediction['Packet_Indices'] + 1

            # Handle missing indices
            if total_packets > 0:
                full_indices = set(range(1, total_packets + 1))
                existing_indices = set(final_df_prediction['Packet_Indices'])
                missing_indices = sorted(list(full_indices - existing_indices))

                if missing_indices:
                    # Create DataFrame for missing indices
                    missing_data = {
                        'Packet_Indices': missing_indices,
                        'Label': ['Benign'] * len(missing_indices)
                    }
                    
                    # Add other columns with appropriate default values
                    for col in final_df_prediction.columns:
                        if col not in missing_data:
                            if col in integer_columns:
                                missing_data[col] = [pd.NA] * len(missing_indices)
                            else:
                                missing_data[col] = [np.nan] * len(missing_indices)
                    
                    df_missing = pd.DataFrame(missing_data)
                    
                    # Ensure consistent data types
                    for col in integer_columns:
                        if col in df_missing.columns:
                            df_missing[col] = df_missing[col].astype('Int64')
                    
                    df_missing['Packet_Indices'] = df_missing['Packet_Indices'].astype(int)
                    
                    # Combine DataFrames
                    final_df_prediction = pd.concat([final_df_prediction, df_missing], ignore_index=True)

            # Sort and clean up
            final_df_prediction = final_df_prediction.sort_values(by='Packet_Indices').reset_index(drop=True)

            # Save results
            try:
                final_df_prediction.to_csv(prompt_file_path, index=False)
                
                # Create display version
                display_columns = [
                    'Packet_Indices', 'Label', 'Source_IP', 'Destination_IP',
                    'Source_Port', 'Destination_Port', 'Protocol'
                ]
                display_df = final_df_prediction[display_columns].copy()
                display_df.to_csv(prediction_file_path, index=False)
                
                pipeline_status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                return display_df, total_packets, str(prompt_file_path)
                
            except Exception as e:
                raise AnalysisError(f"Failed to save results: {e}")

    except AnalysisError as e:
        st.error(f"❌ Analysis Error: {e}")
        return None, 0, None
    except Exception as e:
        st.error(f"❌ Unexpected error during analysis: {e}")
        logger.error(f"Unexpected analysis error: {e}", exc_info=True)
        return None, 0, None

# --- LLM Functions ---
def create_gemini_prompt_for_streamlit(df_row: pd.Series) -> str:
    """Create Gemini prompt with proper data handling"""
    def safe_value(value):
        """Safely convert value to string, handling NaN/None"""
        if pd.isna(value) or value is None:
            return "N/A"
        return str(value)
    
    prompt = f"""
I have a network flow extracted from a .pcap file using CICFlowMeter, predicted by a machine learning model.
This flow was classified as: **{safe_value(df_row['Label'])}**.

Here are the flow details:
- **Packet Index**: {safe_value(df_row['Packet_Indices'])}
- **Source IP**: {safe_value(df_row['Source_IP'])}
- **Destination IP**: {safe_value(df_row['Destination_IP'])}
- **Destination Port**: {safe_value(df_row['Destination_Port'])}
- **Protocol**: {safe_value(df_row['Protocol'])}
- **Minimum segment size observed in the forward direction**: {safe_value(df_row['Fwd_Seg_Size_Min'])} bytes
- **Total initial window bytes in forward direction**: {safe_value(df_row['Init_Fwd_Win_Byts'])} bytes
- **Backward packets per second**: {safe_value(df_row['Bwd_Pkts_s'])}
- **Maximum time between two packets (Flow IAT Max)**: {safe_value(df_row['Flow_IAT_Max'])} ms
- **Flow Duration**: {safe_value(df_row['Flow_Duration'])} ms
- **Mean packet length**: {safe_value(df_row['Pkt_Len_Mean'])} bytes
- **Flow packets per second**: {safe_value(df_row['Flow_Pkts_s'])}
- **Forward header length**: {safe_value(df_row['Fwd_Header_Len'])} bytes
- **Total length of forward packets**: {safe_value(df_row['TotLen_Fwd_Pkts'])} bytes
- **Average packet size**: {safe_value(df_row['Pkt_Size_Avg'])} bytes
- **Total initial window bytes in backward direction**: {safe_value(df_row['Init_Bwd_Win_Byts'])} bytes
- **Mean time between two packets (Flow IAT Mean)**: {safe_value(df_row['Flow_IAT_Mean'])} ms
- **Average forward subflow bytes**: {safe_value(df_row['Subflow_Fwd_Byts'])} bytes
- **Mean backward packet length**: {safe_value(df_row['Bwd_Pkt_Len_Mean'])} bytes
- **Backward header length**: {safe_value(df_row['Bwd_Header_Len'])} bytes
- **Average segment size in backward direction**: {safe_value(df_row['Bwd_Seg_Size_Avg'])} bytes
- **PSH Flag Count**: {safe_value(df_row['PSH_Flag_Cnt'])}
- **Flow bytes per second**: {safe_value(df_row['Flow_Byts_s'])}
- **Forward packets per second**: {safe_value(df_row['Fwd_Pkts_s'])}

Given these details, please explain:
1. **Why** might the model classify this flow as **{safe_value(df_row['Label'])}**? Elaborate on the features that strongly suggest this classification.
2. What **suspicious behaviors** or **flow characteristics** directly support this classification, if any?
3. What **insights** does this specific prediction provide about the network activity?
4. What parts of the original **.pcap file** (e.g., specific filters to apply in Wireshark/tcpdump, packet types) should I examine further to confirm or understand this flow better?
5. Based on this classification, what are the **immediate next steps** for investigation or mitigation from a cybersecurity perspective?
"""
    return prompt.strip()

# --- Main Streamlit Application ---
def initialize_session_state():
    """Initialize all session state variables"""
    session_defaults = {
        'initial_setup_completed': False,
        'setup_failed': False,
        'show_setup_logs': False,
        'proceed_clicked': False,
        'show_results': False,
        'file_selected_successfully': False,
        'selected_filename': None,
        'prediction_results_df': None,
        'total_packets': 0,
        'prompt_csv_path': None,
        'setup_cache_key': 0,
        'llm_chat_history': {},
        'current_llm_flow_index': None,
        'show_llm_chat': False,
        'show_df': False,
        'show_predictions': True,
        'full_prompt_df': None
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_application_state():
    """Reset application state for new analysis"""
    keys_to_reset = [
        'show_results', 'file_selected_successfully', 'selected_filename',
        'prediction_results_df', 'total_packets', 'prompt_csv_path',
        'llm_chat_history', 'current_llm_flow_index', 'show_llm_chat',
        'show_df', 'show_predictions', 'full_prompt_df'
    ]
    
    for key in keys_to_reset: