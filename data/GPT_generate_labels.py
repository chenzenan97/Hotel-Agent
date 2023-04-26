 
import pandas as pd
import re
from tqdm import tqdm
import openai

def generate_qanda_label_file(input_file_path,openai_key):
    def label_customer_questions(file_path, openai_key):
        openai.api_key = openai_key
        file = pd.read_csv(file_path)
        file['Question class'] = None

        for index, row in tqdm(file.iterrows(), total=file.shape[0]):
            question = row['Question']
            class_prompt = "Label this sentence as {1: Reservation, 2: Payment, 3: Check out, 4: Another} return number " + question

            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=class_prompt,
                max_tokens=3000,
                temperature=0.1,
            )
            question_class = response.choices[0].text.strip()

            clean_class = re.sub(r'\D', '', question_class)
            file.loc[index, 'Question class'] = clean_class

        updated_file_path = 'QANDA_label.csv'
        file.to_csv(updated_file_path, index=False)
        return f"Updated file saved to {updated_file_path}"

    result = label_customer_questions(input_file_path, openai_key)
    return result
