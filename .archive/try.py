import os
import subprocess
import requests
import streamlit as st

def setup_sample_pcap():
    try:
        with st.status("Downloading sample pcap...", expanded=True, state="running") as status:
            # Download sample_pcap.zip if it doesn't exist
            if not os.path.exists("sample_pcap.zip"):
                st.write(":hugging_face: Downloading sample_pcap.zip...")
                zip_url = "https://huggingface.co/Mc4minta/RandomForest400IntPortCIC1718/resolve/main/sample_pcap.zip"
                response = requests.get(zip_url, stream=True)
                response.raise_for_status()
                with open("sample_pcap.zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write(":white_check_mark: sample_pcap.zip Downloaded.")

            # Extract sample_pcap if directory doesn't exist or is empty
            if not os.path.exists("sample_pcap") or not os.listdir("sample_pcap"):
                # Create directory if it doesn't exist
                os.makedirs("sample_pcap", exist_ok=True)
                
                st.write(":open_file_folder: Extracting sample_pcap...")
                # Extract the zip file using subprocess
                subprocess.run(["unzip", "-o", "sample_pcap.zip", "-d", "."], check=True, capture_output=True, text=True)
                st.write(":white_check_mark: sample_pcap extracted.")
                
                # Verify extraction was successful
                pcap_files = [f for f in os.listdir("sample_pcap") if f.endswith('.pcap')]
                if pcap_files:
                    st.write(f":information_source: Found {len(pcap_files)} .pcap files in sample_pcap directory.")
                else:
                    st.warning(":warning: sample_pcap directory appears to be empty after extraction.")
            else:
                pcap_files = [f for f in os.listdir("sample_pcap") if f.endswith('.pcap')]
                st.write(f":information_source: Sample PCAP directory already exists with {len(pcap_files)} .pcap files. Skipping extraction.")

            status.update(label=":white_check_mark: Sample Data Setup Complete", state="complete", expanded=False)
            return True  # Indicate success
            
    except subprocess.CalledProcessError as e:
        st.error(f":x: Error extracting sample data: Command '{e.cmd}' returned {e.returncode}. Output: {e.stdout}\nError: {e.stderr}")
    except requests.exceptions.RequestException as e:
        st.error(f":x: Error downloading sample data: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during sample data setup: {e}")
    
    # If any error occurs, update session state and return False
    st.session_state.setup_failed = True
    st.session_state.initial_setup_completed = False
    return False

# Example usage in Streamlit app
if __name__ == "__main__":
    if setup_sample_pcap():
        st.success("Sample data setup completed successfully!")
    else:
        st.error("Failed to set up sample data. Check logs for details.")
