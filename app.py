import streamlit as st
import subprocess
import os
import joblib
import requests
import sklearn

import time

def display_setup_logs():
    
    # CICFlowMeter setup
    with st.status("Setting up CICFlowMeter-3.0...",expanded=True) as status:
        try:
            # install libpcap-dev library
            st.write(":arrow_down: Installing libpcap-dev...")
            subprocess.run(["sudo","apt-get", "install", "-y", "libpcap-dev"], check=True)
            st.write(":white_check_mark: libpcap-dev installed.")
            
            # Download CICFlowMeter.zip from codeberge
            st.write(":arrow_down: Downloading CICFlowMeter-3.0...")
            wget_command = "https://codeberg.org/iortega/TCPDUMP_and_CICFlowMeter/archive/master:CICFlowMeters/CICFlowMeter-3.0.zip"
            subprocess.run(["wget", wget_command], check=True)
            st.write(":white_check_mark: CICFlowMeter-3.0.zip downloaded.")
            
            # Extracting CICFlowMeter from codeberge
            st.write(":open_file_folder: Extracting CICFlowMeter-3.0...")
            subprocess.run(["unzip", "CICFlowMeter-3.0.zip", "-d", "CICFlowMeter-3.0"], check=True)
            st.write(":white_check_mark: CICFlowMeter extracted.")
            
            # Clearing unused zip file
            st.write(":wastebasket: Clearing .zip file...")
            subprocess.run(["rm","CICFlowMeter-3.0.zip"], check=True)
            st.write(":white_check_mark: CICFlowMeter-3.0.zip Cleared")
            
            # Creating data/in data/out directories
            st.write(":file_folder: Creating data/in data/out directories...")
            os.makedirs("data/in", exist_ok=True)
            os.makedirs("data/out", exist_ok=True)
            st.write(":white_check_mark: Directories created.")
            
            # updata status to indicate successful cicflowmeter setup
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
            # downloading model from hugging face
            st.write(":hugging_face: Downloading ML model...")
            model_url = "https://huggingface.co/Mc4minta/RandomForest400IntPortCIC1718/resolve/main/RandomForest400IntPortCIC1718-2.pkl"
            response = requests.get(model_url)
            with open("RandomForest400IntPortCIC1718-2.pkl", "wb") as f:
                f.write(response.content)
            st.write(":white_check_mark: ML Model downloaded.")
            
            # import model as using joblib
            st.write(":robot_face: Loading ML model...")
            model = joblib.load('RandomForest400IntPortCIC1718-2.pkl')
            st.write(":white_check_mark: ML Model loaded successfully.")
            st.info(model)
            
            status.update(label=":white_check_mark: ML Model Setup Complete", state="complete", expanded=False)
        except subprocess.CalledProcessError as e:
            st.error(f":x: Error during ML Model setup: {e}")
            status.update(label=":x: ML Model Setup Failed", state="error", expanded=True)
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            status.update(label=":x: ML Model Setup Failed", state="error", expanded=True)

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
                st.error(f":e: Setup failed: {e}")
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
                st.error(f":e: Setup failed: {e}")
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

    # Proceed to file upload when proceed is clicked
    if st.session_state.initial_setup_completed and st.session_state.proceed_clicked:
        # Upload file logic
        try:
            uploaded_file = st.file_uploader(
                ":file_folder: Choose a .pcap file", 
                accept_multiple_files=False,
                type=["pcap"],
            )
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            st.write(bytes_data)
            
        except Exception as e:
            st.error(f"Error : {e}")

if __name__ == "__main__":
    main()