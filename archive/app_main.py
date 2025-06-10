import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import requests
import joblib
import os

from pipeline import *

def map_port(port):
    if port == 21:
        return 1  # FTP
    elif port == 22:
        return 2  # SSH
    elif port == 53:
        return 3  # DNS
    elif port == 80:
        return 4  # HTTP
    elif port == 443:
        return 5  # HTTPS
    else:
        return 6  # Other

def process_dataframe(df):
    # replace space in columns name with underscore
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # drop objects type columns
    columns_to_drop = [
        'Flow_ID','Src_IP','Dst_IP','Src_Port','Protocol','Timestamp','Label'
    ]

    df = df.drop(columns=columns_to_drop)

    # remove rows with missing and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace = True)
    df = df.dropna()
    
    # map destination port to 1-6 numbers
    df['Dst_Port'] = df['Dst_Port'].apply(map_port)
    
    return df    

def display_setup_logs():
    
    # CICFlowMeter setup
    with st.status("Setting up CICFlowMeter-3.0...",expanded=True) as status:
        try:
            # install libpcap-dev library
            st.write(":arrow_down: Installing libpcap-dev...")
            subprocess.run(["sudo","apt-get", "install", "-y", "libpcap-dev"], check=True)
            st.write(":white_check_mark: libpcap-dev installed.")
            
            # CICflowmeter download if not exist
            if not os.path.exists("CICFlowMeter-3.0"):
                # Download CICFlowMeter.zip from codeberg
                st.write(":arrow_down: Downloading CICFlowMeter-3.0.zip...")
                url = "https://codeberg.org/iortega/TCPDUMP_and_CICFlowMeter/archive/master:CICFlowMeters/CICFlowMeter-3.0.zip"
                subprocess.run(["wget", url, "-O", "CICFlowMeter-3.0.zip"], check=True)
                st.write(":white_check_mark: CICFlowMeter-3.0.zip downloaded.")
                
                # Extracting CICFlowMeter from codeberge
                st.write(":open_file_folder: Extracting CICFlowMeter-3.0...")
                subprocess.run(["unzip", "CICFlowMeter-3.0.zip", "-d", "CICFlowMeter-3.0"], check=True)
                st.write(":white_check_mark: CICFlowMeter extracted.")
                
                # Setting executable permission 
                st.write(":wrench: Configuring executable permission...")
                subprocess.run(["chmod", "+x", "CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter"], check=True)
                st.write(":white_check_mark: Permission configured")
                
                # Clearing unused zip file
                st.write(":wastebasket: Clearing .zip file...")
                subprocess.run(["rm","CICFlowMeter-3.0.zip"], check=True)
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
            st.error(f":x: Error during CICFlowMeter setup: {e}")
            status.update(label=":x: CICFlowMeter Setup Failed", state="error", expanded=True)
            
        except Exception as e:
            st.error(f":x: An unexpected error occurred: {e}")
            status.update(label=":x: CICFlowMeter Setup Failed", state="error", expanded=True)
            
    # Classification Model setup
    with st.status("Setting up ML Model...",expanded=True) as status:
        try:            
            # downloading model from hugging face if not exist
            if not os.path.exists("RandomForest400IntPortCIC1718-2.pkl"):
                st.write(":hugging_face: Downloading ML model...")
                model_url = "https://huggingface.co/Mc4minta/RandomForest400IntPortCIC1718/resolve/main/RandomForest400IntPortCIC1718-2.pkl"
                response = requests.get(model_url)
                with open("RandomForest400IntPortCIC1718-2.pkl", "wb") as f:
                    f.write(response.content)
                st.write(":white_check_mark: ML Model downloaded.")

            # import model as using joblib
            st.write(":robot_face: Loading ML model...")
            model = joblib.load('RandomForest400IntPortCIC1718-2.pkl')
            st.session_state.model_state = model
            st.write(":white_check_mark: ML Model loaded successfully.")
            st.info(model)
            
            status.update(label=":white_check_mark: ML Model Setup Complete", state="complete", expanded=False)
        except subprocess.CalledProcessError as e:
            st.error(f":x: Error during ML Model setup: {e}")
            status.update(label=":x: ML Model Setup Failed", state="error", expanded=True)
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            status.update(label=":x: ML Model Setup Failed", state="error", expanded=True)

def clear_uploaded_files():
    # remove pcap files
    for filename in os.listdir("data/in"):
        filepath = os.path.join("data/in", filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            st.error(f"Failed to delete {filename}: {e}")
    # remove csv files
    for filename in os.listdir("data/out"):
        filepath = os.path.join("data/out", filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            st.error(f"Failed to delete {filename}: {e}")

def initial_setup():
    display_setup_logs()
    st.session_state.initial_setup_completed = True
    st.session_state.setup_failed = False
    st.session_state.show_setup_logs = True

def main():
    
    # page config
    st.set_page_config(
        page_title="Malicious .PCAP File Classifier",
        page_icon=":peacock:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # initial state setup
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

    # css style for centered st.button
    st.markdown("""
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Logic for StartSetup Button
    if not st.session_state.initial_setup_completed and not st.session_state.setup_failed:
        if st.button("Start Setup"):
            st.session_state.initial_setup_attempted = True
            try:
                initial_setup()
            except Exception as e:
                st.error(f"❌ Setup failed: {e}")
                st.session_state.setup_failed = True
                st.session_state.initial_setup_completed = False
            st.rerun()

    # Logic handling StartSetup button failing
    elif st.session_state.setup_failed:
        st.warning("Setup failed. Please try again.")
        if st.button("Start Setup"):
            st.session_state.initial_setup_attempted = True
            try:
                initial_setup()
            except Exception as e:
                st.error(f"❌ Setup failed: {e}")
                st.session_state.setup_failed = True
                st.session_state.initial_setup_completed = False
            st.rerun()

    # show success and proceed after setup
    if st.session_state.initial_setup_completed and not st.session_state.proceed_clicked:
        st.success(":tada: Setup Completed")
        if st.button("Proceed"):
            st.session_state.show_setup_logs = False
            st.session_state.proceed_clicked = True
            st.rerun()

    # show result only
    if st.session_state.show_results:
        uploaded_filename = st.session_state.uploaded_filename
        st.info(f"This is your result for {uploaded_filename}")
        
        # convert pcap to csv using cicflowmter
        try: 
            subprocess.run("CICFlowMeter-3.0/tcpdump_and_cicflowmeter/bin/CICFlowMeter",check=True)
            csv_path = "data/out/"
            csv_name = uploaded_filename[:-5] + "_ISCX.csv"
            df = pd.read_csv(csv_path + csv_name)
            df = process_dataframe(df)
            model = st.session_state.model_state
            
            # Initialize session state for display toggles
            if "show_df" not in st.session_state:
                st.session_state.show_df = False
            if "show_proba" not in st.session_state:
                st.session_state.show_proba = False
            if "show_predictions" not in st.session_state:
                st.session_state.show_predictions = True  # default

            try:
                predictions = model.predict(df)
                predictions_proba = model.predict_proba(df)

                predictions_proba_df = pd.DataFrame(predictions_proba, columns=model.classes_)
                predictions_proba_df['Predicted_Label'] = predictions

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Show Raw Features"):
                        st.session_state.show_df = True
                        st.session_state.show_predictions = False
                        st.session_state.show_proba = False

                with col2:
                    if st.button("Show Predictions"):
                        st.session_state.show_df = False
                        st.session_state.show_predictions = True
                        st.session_state.show_proba = False

                with col3:
                    if st.button("Show Prediction Probabilities"):
                        st.session_state.show_df = False
                        st.session_state.show_predictions = False
                        st.session_state.show_proba = True

                # Display according to state
                if st.session_state.show_df:
                    st.dataframe(df, use_container_width=True)

                if st.session_state.show_predictions:
                    prediction_counts = pd.Series(predictions).value_counts().sort_index()
                    pd.Series(predictions).value_counts().sort_index()
                    st.bar_chart(prediction_counts)

                if st.session_state.show_proba:
                    st.dataframe(predictions_proba_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error making prediction: {e}")
        except Exception as e:
            st.error(f"Error converting pcap to csv: {e}")

        
        if st.button("Upload another file"):
            st.session_state.show_df = False
            st.session_state.show_proba = False
            st.session_state.show_predictions = True
            clear_uploaded_files()
            st.session_state.show_results = False
            st.session_state.file_uploaded_successfully = False
            st.session_state.proceed_clicked = True
            st.rerun()

    # Proceed to file upload when proceed is clicked
    elif st.session_state.initial_setup_completed and st.session_state.proceed_clicked:
        try:
            st.info(":file_folder: Browse your file here")

            uploaded_file = st.file_uploader(
                "Choose a PCAP file", accept_multiple_files=False, type=["pcap"]
            )

            # Show Upload button only when a file is selected
            if uploaded_file is not None:
                with st.container():
                    if st.button("Upload File"):
                        bytes_data = uploaded_file.read()
                        try:
                            with open(f"data/in/{uploaded_file.name}", "wb") as f:
                                f.write(bytes_data)
                            mb_size = len(bytes_data) / (1024 * 1024)
                            st.session_state.uploaded_filename = uploaded_file.name
                            st.success(f":file_folder: {uploaded_file.name} size {mb_size:.2f} MB uploaded successfully")
                            st.session_state.file_uploaded_successfully = True
                        except Exception as e:
                            st.error(f"Error uploading file: {e}")
                            st.session_state.file_uploaded_successfully = False

            # Show See Attacks only when upload was successful
            if st.session_state.file_uploaded_successfully:
                if st.button("See Attacks"):
                    st.session_state.show_results = True
                    st.rerun()

        except Exception as e:
            st.error(f"Error : {e}")

if __name__ == "__main__":
    main()