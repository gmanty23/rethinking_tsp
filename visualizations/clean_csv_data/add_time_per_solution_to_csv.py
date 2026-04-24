import csv
import os

def add_solution_time_column(input_path, output_path, batch_size=128, time_col_name='Time_Per_Inst'):
    try:
        if not os.path.exists(input_path):
            print(f"Error: File '{input_path}' not found.")
            return

        print(f"Processing {input_path} with batch size {batch_size}...")

        with open(input_path, mode='r', newline='', encoding='utf-8') as f_in, \
             open(output_path, mode='w', newline='', encoding='utf-8') as f_out:
            
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)

            # --- 1. Process Header ---
            try:
                headers = next(reader)
            except StopIteration:
                print("Error: The CSV file is empty.")
                return

            # Find where the time column is located
            try:
                time_col_index = headers.index(time_col_name)
            except ValueError:
                print(f"Error: Column '{time_col_name}' not found in the CSV headers.")
                print(f"Available columns are: {headers}")
                return

            # Create new header list with the new column inserted right after the old one
            new_col_name = 'Time_Per_Solution'
            headers.insert(time_col_index + 1, new_col_name)
            writer.writerow(headers)

            # --- 2. Process Rows ---
            row_count = 0
            for row in reader:
                # Get the original time value
                try:
                    # We grab the value at the found index
                    original_time_str = row[time_col_index]
                    
                    # specific check for empty values to avoid crashing
                    if original_time_str.strip() == '':
                         calculated_time = ''
                    else:
                        original_time = float(original_time_str)
                        # THE MATH: Divide by batch size
                        calculated_time = original_time / batch_size
                        # Format it to avoid massive scientific notation (optional, removes extra precision)
                        calculated_time = f"{calculated_time:.8f}" 

                except ValueError:
                    # If conversion fails (e.g. if the row has text instead of numbers), leave blank or copy
                    calculated_time = "N/A"

                # Insert the new value into the row at the same position as the header
                row.insert(time_col_index + 1, calculated_time)
                
                writer.writerow(row)
                row_count += 1

        print(f"Success! Processed {row_count} rows.")
        print(f"New file saved as: {output_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Configuration ---
input_csv = 'results/gnn100.csv'    # The file from the previous step
output_csv = '/home/pfc/gms/code/rethinking_tsp/results/evaluations/09_gnn100_time_per_solution.csv'         # The new file to create
BATCH_SIZE = 128
TARGET_COLUMN = 'Time_Per_Inst'       

if __name__ == "__main__":
    add_solution_time_column(input_csv, output_csv, BATCH_SIZE, TARGET_COLUMN)