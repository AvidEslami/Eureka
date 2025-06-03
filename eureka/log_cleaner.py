# Opens a .txt file
# Removes every line that starts with "Observations:"
# Saves the cleaned content to a file with "_cleaned" appended to the original filename

import os
def clean_log_file(file_path):
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return

    cleaned_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith("Observations:"):
                cleaned_lines.append(line)

    cleaned_file_path = file_path.replace('.txt', '_cleaned.txt')
    with open(cleaned_file_path, 'w') as cleaned_file:
        cleaned_file.writelines(cleaned_lines)

    print(f"Cleaned log saved to {cleaned_file_path}")

if __name__ == "__main__":
    # log_file_path = input("Enter the path to the log file: ")
    # log_file_path = "/home/avidavid/Eureka/eureka/loaded_untuned_20k_epochs.txt"
    log_file_path = "/home/avidavid/Eureka/eureka/15k_epoch_tuned.txt"
    clean_log_file(log_file_path)