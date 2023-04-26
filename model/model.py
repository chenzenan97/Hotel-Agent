from transformers import AutoTokenizer, AutoModelForCausalLM


import torch 
output_dir = "./model/hotel_agent"
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir)

def user_AskQuestion(input_sentence):
    new_user_input_ids = tokenizer.encode(f'User:{input_sentence}'+"Hotel Agent:", return_tensors='pt')
    chat_history_ids = None
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, temperature=0.1, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response
print(user_AskQuestion("I want to book a room") )