import os
import re
from datetime import datetime


def format_for_excel(accuracies):
    # First, transform the data into a single dictionary for easier handling
    combined_accuracies = {}
    for split_method in ['act', 'sen']:
        for feature_method, accuracy in accuracies[split_method]:
            if feature_method not in combined_accuracies:
                combined_accuracies[feature_method] = {}
            combined_accuracies[feature_method][split_method] = accuracy

    # Now format the output
    output_lines = ["mfcc_logf\tACT\tSEN"]
    for feature_method, accuracy_dict in combined_accuracies.items():
        line = f"{feature_method}\t{accuracy_dict.get('act', '')}\t{accuracy_dict.get('sen', '')}"
        output_lines.append(line)

    return "\n".join(output_lines)


# Directory containing the log files
log_dir = r'D:\Projects\emotion_in_speech\CREMA-D\mat\log'

# Regex to match log file names and extract key components
log_pattern = re.compile(r'log_train_(.*?)_(act|sen)_(\d{8}-\d{6})\.log$')

# Store the latest log file for each category and split method
latest_logs = {}

# List all files in the log directory
for filename in os.listdir(log_dir):
    match = log_pattern.match(filename)
    if match:
        feature_method, split_method, timestamp = match.groups()
        # Create a unique key for this category and split method, excluding timestamp
        key = (feature_method, split_method)
        timestamp = datetime.strptime(timestamp, '%Y%m%d-%H%M%S')

        # Update the latest log file for this category and split method
        if key not in latest_logs or timestamp > latest_logs[key][1]:
            latest_logs[key] = (filename, timestamp)

# Dicts to store final accuracies
accuracies = {'act': [], 'sen': []}

# Regex to find the test accuracy in a file
accuracy_pattern = re.compile(r'Test accuracy: ([0-9.]+)')

# Read each latest log file and extract the test accuracy
for (feature_method, split_method), (filename, _) in latest_logs.items():
    with open(os.path.join(log_dir, filename), 'r') as file:
        for line in file:
            accuracy_match = accuracy_pattern.search(line)
            if accuracy_match:
                accuracy = round(float(accuracy_match.group(1)), 4)
                accuracies[split_method].append((feature_method, accuracy))
                break

# Print the results
for split_method in ['act', 'sen']:
    print(f"Results for {split_method}:")
    for feature_method, accuracy in sorted(accuracies[split_method]):
        print(f"{feature_method}: {accuracy}")
    print("\n")

formatted_string = format_for_excel(accuracies)
print(formatted_string)
