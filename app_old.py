import os
import zipfile
import requests
import joblib
import streamlit as st
import subprocess
import pandas as pd
import shutil
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import tempfile
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
class Config:
    CIC_URL = "https://codeberg.org/iortega/TCPDUMP_and_CICFlowMeter/archive/master:CICFlowMeters/CICFlowMeter-3.0.zip"
    MODEL_URL = "https://huggingface.co/Mc4minta/RandomForest400IntPortCIC1718/resolve/main/RandomForest400IntPortCIC1718-2.pkl"
    CLASS_LABELS = ['Benign', 'DoS-HTTP-Flood', 'DoS-Slow-Rate', 'FTP-Bruteforce', 'PortScan', 'SSH-Bruteforce']
    PORT_MAPPING = {21: 1, 22: 2, 53: 3, 80: 4, 443: 5}  # FTP, SSH, DNS, HTTP, HTTPS
    COLUMNS_TO_DROP = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Src_Port', 'Protocol', 'Label']

# --- Initialize Session State ---
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'setup_started': False,
        'last_uploaded_pcap': None,
        'model': None,
        'analysis_results': None,
        'setup_complete': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Utility Functions ---
def map_port(port: int) -> int:
    """Maps common destination ports to numerical categories (1-5) and others to 6."""
    return Config.PORT_MAPPING.get(port, 6)

def safe_download(url: str, filepath: str, description: str = "file") -> bool:
    """Safely download a file with progress indication"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        st.progress(progress, text=f"Downloading {description}: {progress:.1%}")

        return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False

def validate_pcap_file(file_path: str) -> bool:
    """Basic validation for PCAP files"""
    try:
        # Check file size (not empty, not too large)
        size = os.path.getsize(file_path)
        if size == 0:
            return False, "File is empty"
        if size > 100 * 1024 * 1024:  # 100MB limit
            return False, "File too large (>100MB)"

        # Check magic bytes for PCAP formats
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            pcap_magic = [b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4', b'\n\r\r\n']
            if not any(magic.startswith(m) for m in pcap_magic):
                return False, "Invalid PCAP format"

        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"

# --- Enhanced Preprocessing Function ---
def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced preprocessing with detailed statistics and error handling

    Returns:
        Tuple of (processed_dataframe, preprocessing_stats)
    """
    stats = {
        'original_rows': len(df),
        'original_cols': len(df.columns),
        'duplicates_removed': 0,
        'inf_values_found': 0,
        'nan_values_found': 0,
        'rows_after_cleaning': 0,
        'features_used': 0
    }

    try:
        # 1. Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_')

        # 2. Remove duplicates
        before_dup = len(df)
        df.drop_duplicates(inplace=True)
        stats['duplicates_removed'] = before_dup - len(df)

        # 3. Handle infinity and NaN values
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]).values).any(axis=1)
        stats['inf_values_found'] = inf_mask.sum()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        stats['nan_values_found'] = df.isnull().sum().sum()
        df.dropna(inplace=True)

        # 4. Drop unused columns (only existing ones)
        cols_to_drop = [col for col in Config.COLUMNS_TO_DROP if col in df.columns]
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        # 5. Enhanced port encoding with error handling
        if 'Destination_Port' in df.columns:
            df['Destination_Port'] = pd.to_numeric(
                df['Destination_Port'], errors='coerce'
            ).fillna(0).astype(int)
            df['Destination_Port'] = df['Destination_Port'].apply(map_port)

        # 6. Convert all columns to numeric with better error handling
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 7. Remove any remaining problematic rows
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

        stats['rows_after_cleaning'] = len(df)
        stats['features_used'] = len(df.columns)

        return df, stats

    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise ValueError(f"Preprocessing failed: {e}")

# --- Enhanced Setup Function ---
@st.cache_resource
def perform_enhanced_setup():
    """Enhanced setup with better error handling and progress tracking"""

    progress_bar = st.progress(0, text="Starting setup...")

    try:
        # Step 1: Create directories (10%)
        progress_bar.progress(0.1, text="Creating directories...")
        for dir_path in ["data/in", "data/out", "CICFlowMeter"]:
            os.makedirs(dir_path, exist_ok=True)

        # Step 2: Download CICFlowMeter (40%)
        progress_bar.progress(0.2, text="Downloading CICFlowMeter...")
        cic_zip_path = "CICFlowMeter/CICFlowMeter-3.0.zip"
        if not safe_download(Config.CIC_URL, cic_zip_path, "CICFlowMeter"):
            raise Exception("Failed to download CICFlowMeter")

        # Step 3: Extract CICFlowMeter (60%)
        progress_bar.progress(0.4, text="Extracting CICFlowMeter...")
        extract_cicflowmeter(cic_zip_path)

        # Step 4: Download model (80%)
        progress_bar.progress(0.6, text="Downloading ML model...")
        model_path = "RandomForest400IntPortCIC1718.pkl"
        if not safe_download(Config.MODEL_URL, model_path, "ML model"):
            raise Exception("Failed to download ML model")

        # Step 5: Load model (100%)
        progress_bar.progress(0.8, text="Loading ML model...")
        model = joblib.load(model_path)

        progress_bar.progress(1.0, text="Setup complete!")
        time.sleep(1)  # Brief pause to show completion
        progress_bar.empty()

        return model

    except Exception as e:
        progress_bar.empty()
        logger.error(f"Setup failed: {e}")
        raise e

def extract_cicflowmeter(zip_path: str):
    """Enhanced CICFlowMeter extraction with better error handling"""
    try:
        zip_target_dir = "CICFlowMeter"

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_target_dir)

        os.remove(zip_path)  # Clean up

        # Find and organize the extracted content
        extracted_dirs = [d for d in os.listdir(zip_target_dir)
                         if os.path.isdir(os.path.join(zip_target_dir, d))
                         and d.startswith("TCPDUMP_and_CICFlowMeter")]

        if not extracted_dirs:
            raise FileNotFoundError("Could not find extracted CICFlowMeter directory")

        source_path = os.path.join(zip_target_dir, extracted_dirs[0],
                                  "CICFlowMeters", "CICFlowMeter-3.0",
                                  "tcpdump_and_cicflowmeter")
        dest_path = os.path.join(zip_target_dir, "tcpdump_and_cicflowmeter")

        if os.path.exists(source_path):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.move(source_path, dest_path)
            shutil.rmtree(os.path.join(zip_target_dir, extracted_dirs[0]))
        else:
            raise FileNotFoundError(f"CICFlowMeter content not found at {source_path}")

    except Exception as e:
        logger.error(f"CICFlowMeter extraction failed: {e}")
        raise e

# --- Enhanced Analysis Function with Debug Mode ---
def perform_traffic_analysis(pcap_filename: str) -> Dict:
    """Enhanced traffic analysis with comprehensive debugging and error handling"""

    results = {
        'success': False,
        'error': None,
        'preprocessing_stats': {},
        'predictions': {},
        'classification': 'Unknown',
        'confidence_metrics': {},
        'debug_info': {}
    }

    # Debug container for real-time updates
    debug_container = st.container()

    try:
        with debug_container:
            st.write("🔍 **Debug Information:**")
            debug_info = st.empty()

            # Setup paths
            cic_base_dir = Path("CICFlowMeter/tcpdump_and_cicflowmeter")
            cic_executable = cic_base_dir / "bin" / "CICFlowMeter"
            cic_data_in = cic_base_dir / "data" / "in"
            cic_data_out = cic_base_dir / "data" / "out"

            debug_info.write(f"✅ Paths configured:\n- Base: {cic_base_dir}\n- Executable: {cic_executable}\n- Input: {cic_data_in}\n- Output: {cic_data_out}")

            # Check if CICFlowMeter exists
            if not cic_executable.exists():
                raise FileNotFoundError(f"CICFlowMeter executable not found at {cic_executable}")

            debug_info.write(f"✅ CICFlowMeter executable found\n✅ Paths configured")

            # Prepare CICFlowMeter environment
            for dir_path in [cic_data_in, cic_data_out]:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                dir_path.mkdir(parents=True)

            debug_info.write(f"✅ Directories prepared\n✅ CICFlowMeter executable found\n✅ Paths configured")

            # Set executable permissions
            if not os.access(cic_executable, os.X_OK):
                os.chmod(cic_executable, 0o755)
                debug_info.write(f"✅ Execute permissions set\n✅ Directories prepared\n✅ CICFlowMeter executable found\n✅ Paths configured")

            # Check if input file exists
            pcap_source = Path("data/in") / pcap_filename
            if not pcap_source.exists():
                raise FileNotFoundError(f"Input PCAP file not found: {pcap_source}")

            # Move PCAP file to CICFlowMeter input
            pcap_dest = cic_data_in / pcap_filename
            shutil.copy2(str(pcap_source), str(pcap_dest))  # Use copy instead of move
            debug_info.write(f"✅ PCAP file copied to CICFlowMeter input\n✅ Execute permissions set\n✅ Directories prepared\n✅ CICFlowMeter executable found\n✅ Paths configured")

            # Run CICFlowMeter with detailed output
            debug_info.write(f"🚀 Running CICFlowMeter...\n✅ PCAP file copied to CICFlowMeter input\n✅ Execute permissions set\n✅ Directories prepared\n✅ CICFlowMeter executable found")

            try:
                result = subprocess.run(
                    ["./bin/CICFlowMeter"],
                    cwd=cic_base_dir,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5-minute timeout
                    check=False  # Don't raise exception on non-zero exit
                )

                # Log subprocess results
                results['debug_info']['return_code'] = result.returncode
                results['debug_info']['stdout'] = result.stdout
                results['debug_info']['stderr'] = result.stderr

                debug_info.write(f"✅ CICFlowMeter process completed (return code: {result.returncode})")

                if result.returncode != 0:
                    st.error(f"CICFlowMeter returned error code {result.returncode}")
                    st.write("**STDOUT:**")
                    st.code(result.stdout)
                    st.write("**STDERR:**")
                    st.code(result.stderr)
                    raise subprocess.CalledProcessError(result.returncode, "CICFlowMeter", result.stderr)

                if result.stdout:
                    st.write("**CICFlowMeter Output:**")
                    st.code(result.stdout)

            except subprocess.TimeoutExpired:
                debug_info.write("❌ CICFlowMeter timed out after 5 minutes")
                raise

            # Check output directory contents
            output_files = list(cic_data_out.rglob("*"))
            debug_info.write(f"📁 Files in output directory: {len(output_files)}")

            if output_files:
                st.write("**Output directory contents:**")
                for file_path in output_files:
                    st.write(f"- {file_path}")

            # Find CSV files specifically
            csv_files = list(cic_data_out.rglob("*.csv"))
            debug_info.write(f"📊 CSV files found: {len(csv_files)}")

            if not csv_files:
                # List all files for debugging
                all_files = []
                for root, dirs, files in os.walk(cic_data_out):
                    for file in files:
                        full_path = os.path.join(root, file)
                        all_files.append(full_path)

                st.error(f"No CSV files found in output directory!")
                st.write("**All files in output directory:**")
                for f in all_files:
                    st.write(f"- {f}")

                raise FileNotFoundError("No CSV output generated by CICFlowMeter")

            # Load and analyze the CSV
            feature_csv = csv_files[0]
            st.write(f"📄 Loading CSV: {feature_csv}")

            try:
                df_features = pd.read_csv(feature_csv)
                debug_info.write(f"✅ CSV loaded: {df_features.shape[0]} rows, {df_features.shape[1]} columns")

                # Show first few rows for debugging
                st.write("**First 5 rows of extracted features:**")
                st.dataframe(df_features.head())

                if df_features.empty:
                    raise ValueError("Generated CSV is empty - no flows detected")

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                # Try to read the file as text to see its contents
                try:
                    with open(feature_csv, 'r') as f:
                        content = f.read(1000)  # First 1000 characters
                    st.write("**CSV file contents (first 1000 chars):**")
                    st.code(content)
                except:
                    st.write("Could not read CSV file contents")
                raise

            # Enhanced preprocessing
            debug_info.write(f"🔧 Starting preprocessing...")
            df_processed, preprocessing_stats = preprocess_dataframe(df_features.copy())
            results['preprocessing_stats'] = preprocessing_stats

            debug_info.write(f"✅ Preprocessing complete: {df_processed.shape[0]} rows remaining")

            if df_processed.empty:
                st.error("No valid flows remain after preprocessing!")
                st.write("**Preprocessing statistics:**")
                st.json(preprocessing_stats)
                raise ValueError("No valid flows after preprocessing")

            # Make predictions
            if st.session_state.model is None:
                raise ValueError("ML model not loaded in session state")

            debug_info.write(f"🤖 Making predictions...")
            model = st.session_state.model
            predictions = model.predict(df_processed)

            # Calculate prediction statistics
            prediction_counts = pd.Series(predictions).value_counts().sort_index()
            label_mapping = {i: label for i, label in enumerate(Config.CLASS_LABELS)}

            results['predictions'] = {
                label_mapping.get(idx, f'Unknown_{idx}'): count
                for idx, count in prediction_counts.items()
            }

            # Determine overall classification
            is_malicious = any(idx != 0 for idx in prediction_counts.index)
            results['classification'] = 'Malicious' if is_malicious else 'Benign'

            # Add confidence metrics
            total_flows = len(predictions)
            benign_ratio = prediction_counts.get(0, 0) / total_flows
            results['confidence_metrics'] = {
                'total_flows': total_flows,
                'benign_ratio': benign_ratio,
                'malicious_ratio': 1 - benign_ratio
            }

            debug_info.write(f"✅ Analysis complete: {results['classification']}")

            # Cleanup and move results
            final_csv_path = Path("data/out") / feature_csv.name
            shutil.copy2(str(feature_csv), str(final_csv_path))  # Copy instead of move for debugging

            results['success'] = True
            return results

    except subprocess.TimeoutExpired:
        results['error'] = "CICFlowMeter analysis timed out (>5 minutes)"
        st.error(results['error'])
    except subprocess.CalledProcessError as e:
        results['error'] = f"CICFlowMeter failed with return code {e.returncode}"
        st.error(results['error'])
        if hasattr(e, 'stderr') and e.stderr:
            st.code(e.stderr)
    except FileNotFoundError as e:
        results['error'] = f"File not found: {str(e)}"
        st.error(results['error'])
    except Exception as e:
        results['error'] = f"Analysis error: {str(e)}"
        st.error(results['error'])
        logger.error(f"Traffic analysis failed: {e}")

        # Additional debugging for unexpected errors
        st.write("**Exception details:**")
        import traceback
        st.code(traceback.format_exc())

    return results

# --- Enhanced UI Functions ---
def display_analysis_results(results: Dict):
    """Display comprehensive analysis results"""
    if not results['success']:
        st.error(f"Analysis failed: {results['error']}")
        return

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # Classification result
    classification = results['classification']
    if classification == 'Malicious':
        st.error(f"🔴 **Classification: {classification}**")
        st.balloons()
    else:
        st.success(f"🟢 **Classification: {classification}**")

    # Confidence metrics
    metrics = results['confidence_metrics']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Flows", metrics['total_flows'])
    with col2:
        st.metric("Benign Ratio", f"{metrics['benign_ratio']:.2%}")
    with col3:
        st.metric("Malicious Ratio", f"{metrics['malicious_ratio']:.2%}")

    # Detailed predictions
    with st.expander("📈 Detailed Prediction Breakdown"):
        pred_df = pd.DataFrame(
            list(results['predictions'].items()),
            columns=['Classification', 'Count']
        )
        st.dataframe(pred_df, use_container_width=True)

        # Simple bar chart
        st.bar_chart(pred_df.set_index('Classification'))

    # Preprocessing statistics
    with st.expander("🔧 Preprocessing Statistics"):
        stats = results['preprocessing_stats']
        stats_df = pd.DataFrame([
            ("Original Rows", stats['original_rows']),
            ("Duplicates Removed", stats['duplicates_removed']),
            ("Infinite Values Found", stats['inf_values_found']),
            ("NaN Values Found", stats['nan_values_found']),
            ("Rows After Cleaning", stats['rows_after_cleaning']),
            ("Features Used", stats['features_used'])
        ], columns=['Metric', 'Value'])
        st.dataframe(stats_df, use_container_width=True)

# --- Main Application ---
def main():
    # Page configuration
    st.set_page_config(
        page_title="Malicious .PCAP File Classifier",
        page_icon="🛡️",
        layout="centered"
    )

    # Initialize session state
    init_session_state()

    # Header
    st.markdown("<h1 style='text-align: center; color: #b5213b;'>AI Builders 2025</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Malicious <span style='color: #074ec0;'>.pcap</span> File Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'><span style='color: #1abc9c;'>By: MiN - Vibrant Peacock</span> 🦚</h3>", unsafe_allow_html=True)

    # Setup phase
    if not st.session_state.setup_complete:
        st.markdown("---")
        st.subheader("🚀 Initial Setup Required")
        st.info("Click the button below to download and configure CICFlowMeter and the ML model.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔧 Start Setup", use_container_width=True):
                try:
                    with st.spinner("Setting up application..."):
                        model = perform_enhanced_setup()
                        st.session_state.model = model
                        st.session_state.setup_complete = True
                    st.success("✅ Setup completed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Setup failed: {e}")

    else:
        # Main application
        st.success("✅ Setup complete! Ready for analysis.")

        # File upload section
        st.markdown("---")
        st.subheader("📁 Upload PCAP File")

        uploaded_file = st.file_uploader(
            "Choose a PCAP file",
            type=["pcap"],
            accept_multiple_files=False
        )

        if uploaded_file:
            if uploaded_file.name != st.session_state.last_uploaded_pcap:
                st.info(f"Selected: **{uploaded_file.name}**")

                if st.button("📤 Submit File for Analysis"):
                    # Save uploaded file
                    file_path = os.path.join("data", "in", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Validate file
                    is_valid, msg = validate_pcap_file(file_path)
                    if not is_valid:
                        st.error(f"Invalid PCAP file: {msg}")
                        os.remove(file_path)
                    else:
                        st.session_state.last_uploaded_pcap = uploaded_file.name
                        st.success("✅ File uploaded and validated successfully!")
                        st.rerun()
            else:
                st.success(f"✅ File **{uploaded_file.name}** is ready for analysis.")

        # Analysis section
        if st.session_state.last_uploaded_pcap:
            st.markdown("---")
            st.subheader("🔍 Traffic Analysis")

            if st.button("🚀 Start Analysis", type="primary"):
                with st.spinner("Analyzing network traffic..."):
                    results = perform_traffic_analysis(st.session_state.last_uploaded_pcap)
                    st.session_state.analysis_results = results

                display_analysis_results(results)

        # Display previous results if available
        elif st.session_state.analysis_results:
            display_analysis_results(st.session_state.analysis_results)

if __name__ == "__main__":
    main()