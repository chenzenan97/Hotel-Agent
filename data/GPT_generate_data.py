import openai
import os
import openai

def GTP_generate_conversations(openai_key, hotel_information):
    openai.api_key = openai_key
    file_path ='./DialogueText/'
    file_list = [file.split('.')[0] for file in os.listdir(file_path)]
    file_num = 0
    while file_num < 5000:
        file_num += 1
        file_name = str(file_num)
        if file_name not in file_list:
            try:
                file_list.append(file_name)
                prompt = "create conversation between hotel agent and customer, Use the data given to you below, 500 words \n\n" + str(hotel_information)
                answer = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=3500,
                    temperature=0.1,
                ).choices[0].text
            except openai.Error as e:
                print(f"Error generating conversation for {file_name}: {e}")
                continue
            with open(file_path + file_name + '.txt', 'w') as f:
                f.write(answer)

                
if __main__ == '__name__':
    api_response(key,hotel_information)
