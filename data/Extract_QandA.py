import csv
import re
import glob
import os 


def clean_dataset_to_CSV():
    # Specify the input data path and file pattern
    data_path = r'./conversations_lihao'
    file_pattern = os.path.join(data_path, '*.txt')

    # Find all files matching the pattern
    file_list = glob.glob(file_pattern)

    # Specify the output CSV file name and header row
    output_file = "output.csv"
    fieldnames = ["Question", "Answer"]

    # Write the header row to the output CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop over each input file
        for data in file_list:
            # Read the lines from the input file
            with open(data,'r',encoding='utf-8') as f:
                lines = f.readlines()

            # Clean and filter the lines
            lines = [line.strip() for line in lines if line.strip() != '']

            # Loop over each line in the input file
            for i in range(len(lines)):
                line = lines[i].strip()

                # If the line is a customer question, extract and clean the question text
                if i != len(lines) - 1 and re.findall('Customer:', line):
                    question = line.replace('Customer:', '').strip()
                    clean_question = re.sub(r'[^\w\s\d,]', '', question)

                # If the line is a hotel agent answer, extract and clean the answer text
                elif i != 0 and re.findall('Hotel Agent:', line):
                    answer = line.replace('Hotel Agent:', '').strip()
                    clean_answer = re.sub(r'[^\w\s\d,]', '', answer)

                    # Write the cleaned question and answer to the output CSV file
                    with open(output_file, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({'Question': clean_question, 'Answer': clean_answer})

# Call the function to run the cleaning and writing process
if  __name__ == '__main__':
    clean_dataset_to_CSV()
