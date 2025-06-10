import pandas as pd
import google.generativeai as genai
import os

# --- Configuration ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it before running the script (e.g., export GOOGLE_API_KEY='YOUR_API_KEY')")
    exit()

GEMINI_MODEL = "gemini-1.5-flash" # Consider using 1.5-flash for better chat capabilities if 2.0-flash is limiting or deprecated

# Define input and output directories (ensure these directories exist)
data_in_dir = 'data/in/'
data_out_dir = 'data/out/'

# Define base file name
file_name = 'ftpbrute-kali' # Or change this
prompt_file_name = f'{file_name}_prompt.csv'

# Correct path for the input CSV file
# Assuming your input CSV is named 'ftpbrute-kali.csv' and is in 'data/in/'
CSV_FILE_PATH = f'{data_out_dir}{prompt_file_name}'

# --- Function to generate the initial prompt ---
def create_gemini_prompt(df_row):
    prompt = f"""
I have a network flow extracted from a .pcap file using CICFlowMeter.
This flow was predicted by my machine learning model as: {df_row['Label']}.

Here are the flow details:
- Source IP: {df_row['Source_IP']}
- Destination IP: {df_row['Destination_IP']}
- Destination Port: {df_row['Destination_Port']}
- Protocol: {df_row['Protocol']}
- Minimum segment size observed in the forward direction: {df_row['Fwd_Seg_Size_Min']}
- The total number of bytes sent in initial window in the forward direction: {df_row['Init_Fwd_Win_Byts']}
- Number of backward packets per second: {df_row['Bwd_Pkts_s']}
- Maximum time between two packets sent in the flow: {df_row['Flow_IAT_Max']} ms
- Duration of the flow in Microsecond: {df_row['Flow_Duration']} ms
- Mean length of a packet: {df_row['Pkt_Len_Mean']}
- Number of flow packets per second : {df_row['Flow_Pkts_s']}
- Total bytes used for headers in the forward direction: {df_row['Fwd_Header_Len']}
- TotLen Fwd Pkts: {df_row['TotLen_Fwd_Pkts']}
- Average size of packet: {df_row['Pkt_Size_Avg']}
- The total number of bytes sent in initial window in the backward direction: {df_row['Init_Bwd_Win_Byts']}
- Mean time between two packets sent in the flow: {df_row['Flow_IAT_Mean']} ms
- The average number of bytes in a sub flow in the forward direction: {df_row['Subflow_Fwd_Byts']}
- Mean size of packet in backward direction: {df_row['Bwd_Pkt_Len_Mean']}
- Total bytes used for headers in the backward direction: {df_row['Bwd_Header_Len']}
- Average size observed in the backward direction: {df_row['Bwd_Seg_Size_Avg']}
- Number of packets with PUSH: {df_row['PSH_Flag_Cnt']}
- Number of flow bytes per second: {df_row['Flow_Byts_s']}
- Number of forward packets per second : {df_row['Fwd_Pkts_s']}

Please explain:
1. Why might the model classify this flow as {df_row['Label']}?
2. What suspicious behaviors or flow features support this classification?
3. What insights does this prediction provide about the network activity?
4. What parts of the original .pcap file (e.g., packet types, filters) should I examine further to confirm or understand this flow better?
"""
    return prompt.strip()

# --- Main program execution ---
if __name__ == "__main__":
    try:
        # Create directories if they don't exist (good practice)
        os.makedirs(data_in_dir, exist_ok=True)
        os.makedirs(data_out_dir, exist_ok=True)

        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded '{CSV_FILE_PATH}'. Shape: {df.shape}")
        print("Columns found in CSV:", df.columns.tolist())

        required_columns = [
            'Packet_Indices', 'Label', 'Source_IP', 'Destination_IP', 'Destination_Port', 'Protocol',
            'Fwd_Seg_Size_Min', 'Init_Fwd_Win_Byts', 'Bwd_Pkts_s', 'Flow_IAT_Max',
            'Flow_Duration', 'Pkt_Len_Mean', 'Flow_Pkts_s', 'Fwd_Header_Len',
            'TotLen_Fwd_Pkts', 'Pkt_Size_Avg', 'Init_Bwd_Win_Byts', 'Flow_IAT_Mean',
            'Subflow_Fwd_Byts', 'Bwd_Pkt_Len_Mean', 'Bwd_Header_Len',
            'Bwd_Seg_Size_Avg', 'PSH_Flag_Cnt', 'Flow_Byts_s', 'Fwd_Pkts_s'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"\nError: The following required columns are missing from '{CSV_FILE_PATH}':")
            for col in missing_columns:
                print(f"- {col}")
            print("Please ensure your CSV file contains all the specified flow details.")
            exit()
        else:
            print("All required columns found in the CSV.")

        model = genai.GenerativeModel(GEMINI_MODEL)
        print(f"Using Gemini model: {GEMINI_MODEL}")

        # Display non-benign flows for quick reference
        print("\n--- Non-Benign Flows in your data (for reference) ---")
        print(df[(df['Label'] != 'Benign')])
        print("--------------------------------------------------")

        # --- Main loop for user input and flow processing ---
        while True:
            try:
                user_input = input("\nEnter **Packet_Indices** to analyze (or 'all' to process all, 'quit' to exit): ").strip()
                if user_input.lower() == 'quit':
                    print("Exiting program.")
                    break
                elif user_input.lower() == 'all':
                    flows_to_process = df
                    print("Processing all flows... (Note: This will not allow follow-up questions for individual flows.)")
                else:
                    target_index = int(user_input)
                    # Filter the DataFrame to get the row(s) with the specified Packet_Indices
                    flows_to_process = df[df['Packet_Indices'] == target_index]

                    if flows_to_process.empty:
                        print(f"No flow found with Packet_Indices: {target_index}. Please try again.")
                        continue
                    print(f"Processing flow with Packet_Indices: {target_index}...")

                # Process the selected flows
                for index, row in flows_to_process.iterrows():
                    print(f"\n--- Analyzing Flow {row['Packet_Indices']} (Label: {row['Label']}) ---")
                    initial_prompt_text = create_gemini_prompt(row)

                    try:
                        # Start a chat session for this specific flow
                        chat = model.start_chat(history=[])
                        print("\n**Gemini's Initial Explanation:**")
                        response = chat.send_message(initial_prompt_text)
                        print(response.text)

                        # --- Follow-up interaction loop ---
                        if user_input.lower() != 'all': # Only allow follow-up for single flow analysis
                            while True:
                                follow_up_question = input("\nAsk a follow-up question (or type 'done' to analyze another flow): ").strip()
                                if follow_up_question.lower() == 'done':
                                    print("Ending conversation for this flow.")
                                    break
                                elif not follow_up_question:
                                    print("Please enter a question.")
                                    continue

                                try:
                                    follow_up_response = chat.send_message(follow_up_question)
                                    print("\n**Gemini's Response:**")
                                    print(follow_up_response.text)
                                except Exception as e:
                                    print(f"Error getting follow-up response from Gemini: {e}")

                    except Exception as e:
                        print(f"Error starting chat or sending initial request to Gemini for flow {row['Packet_Indices']}: {e}")
                        print("Skipping this flow.")

            except ValueError:
                print("Invalid input. Please enter a number, 'all', or 'quit'.")
            except Exception as e:
                print(f"An unexpected error occurred during selection or processing: {e}")

    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{CSV_FILE_PATH}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Could not parse '{CSV_FILE_PATH}'. Check CSV format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")