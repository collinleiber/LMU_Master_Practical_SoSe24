import xml.etree.ElementTree as ET
import csv
import os

input_file_path = '../shared/example_files/DomesticDeclarations.xes'
output_file_path = '../shared/example_files/DomesticDeclarations_afterDC.csv'

tree = ET.parse(input_file_path)
root = tree.getroot()

with open(output_file_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    # Write the CSV file header
    writer.writerow(['case_id', 'timestamp', 'activity'])

    # Iterate over each trace
    for trace in root.findall('trace'):
        case_id = trace.find('string[@key="id"]').get('value')

        # Iterate over each event
        for event in trace.findall('event'):
            timestamp = event.find('date[@key="time:timestamp"]').get('value')
            activity = event.find('string[@key="concept:name"]').get('value')

            # Write to CSV file
            writer.writerow([case_id, timestamp, activity])

print("Done")