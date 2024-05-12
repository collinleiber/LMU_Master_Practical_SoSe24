import pandas as pd
import os
import io

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'example_files', 'BPI2016_Clicks_Logged_In.csv')

data = pd.read_csv(file_path, sep=';')
# data = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)

# print(data.columns)

data = data[['SessionID', 'TIMESTAMP', 'PAGE_NAME']]

data.rename(columns={'SessionID': 'case_id', 'TIMESTAMP': 'timestamp', 'PAGE_NAME': 'activity'}, inplace=True)

data.sort_values(by=['case_id', 'timestamp'], inplace=True)

data_grouped = data.groupby('case_id').apply(lambda x: x.iloc[max(0, x.index[0] - 10000): min(x.index[-1] + 1001, len(x))])

data_grouped.reset_index(drop=True, inplace=True)

output_file_path = os.path.join(current_dir, 'sorted_session_data1.csv')
data_grouped.to_csv(output_file_path, sep=';', index=False)

print("Data was already cleaned and saved in:", output_file_path)