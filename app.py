import streamlit as st

def display_setup_logs():
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

def initial_setup():
    display_setup_logs()
    st.success(":tada: Setup Completed")
    st.session_state.initial_setup_completed = True
    st.session_state.setup_failed = False
    st.session_state.show_setup_logs = True

def main():
    st.set_page_config(
        page_title="Malicious .PCAP File Classifier",
        page_icon=":peacock:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    if 'initial_setup_completed' not in st.session_state:
        st.session_state.initial_setup_completed = False
    if 'setup_failed' not in st.session_state:
        st.session_state.setup_failed = False
    if 'show_setup_logs' not in st.session_state:
        st.session_state.show_setup_logs = False
    if 'proceed_clicked' not in st.session_state:
        st.session_state.proceed_clicked = False

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

    if st.session_state.show_setup_logs:
        display_setup_logs()
        st.success(":tada: Setup Completed")

    if st.session_state.initial_setup_completed and not st.session_state.proceed_clicked:
        if st.button("Proceed"):
            st.session_state.show_setup_logs = False
            st.session_state.proceed_clicked = True
            st.rerun()

    if st.session_state.initial_setup_completed and st.session_state.proceed_clicked:
        try:
            st.info("Browse your file here")
            # Add your file Browse and classification logic here
        except Exception as e:
            st.error(f"Error : {e}")

if __name__ == "__main__":
    main()