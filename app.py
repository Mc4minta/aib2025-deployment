import streamlit as st

def initial_setup():
    # CICFlowMeter setup
    with st.status("Setting up CICFlowMeter..."):
        # Download CICFlowMeter.zip from codeberge
        st.write("Downloading CICFlowMeter...")
        # Extracting CICFlowMeter from codeberge
        st.write("Extracting CICFlowMeter...")
        # Creating data/in data/out directories
        st.write("Creating data/in data/out directories...")
    
    # Classification Model setup
    with st.status("Setting up ML Model..."):
        # downloading model from huggin face
        st.write("Downloading ML model from huggingface...")
        # import model as using joblib
        st.write("Loading ML model...")

    st.success(":tada: Setup Completed")
    st.session_state.initial_setup_completed = True
    st.session_state.setup_failed = False

def main():
    st.set_page_config(
        page_title="Malicious .PCAP File Classifier",
        page_icon=":peacock:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    # setup states
    if 'initial_setup_completed' not in st.session_state:
        st.session_state.initial_setup_completed = False
    if 'setup_failed' not in st.session_state:
        st.session_state.setup_failed = False
    
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
    
    # StartSetupButton
    st.markdown("""
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
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
    elif st.session_state.setup_failed:
        st.warning("Setup failed previously. Please try again.")
        if st.button("Start Setup"):
            st.session_state.initial_setup_attempted = True
            try:
                initial_setup()
            except Exception as e:
                st.error(f"❌ Setup failed: {e}")
                st.session_state.setup_failed = True
                st.session_state.initial_setup_completed = False
            st.rerun()

    # ContinueButton
    if st.session_state.initial_setup_completed:
        if st.button("Proceed"):
            try:
                st.info("Browse your file here")
                # Add your file Browse and classification logic here
            except Exception as e:
                st.error(f"Error : {e}")
                
if __name__ == "__main__":
    main()