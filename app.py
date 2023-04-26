import streamlit as st
import openai
from model.model import user_AskQuestion
import re 

st.set_page_config(page_title="客服助手", layout="centered")

# Add an image at the top
st.image("./imgs/hotel_01.png", width=600)

st.title("Welcome to LiHao Hotel")

# Display the author's contact information
st.markdown('<div style="text-align: left; font-weight: bold;">Created by:Zenan Chen</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: left;">Email: chenzenan99@gmail.com</div>', unsafe_allow_html=True)

def display_chat(q, r):
    st.markdown(f'<div style="text-align: left; font-weight: bold;">USer:</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: left; background-color: #DCF8C6; border-radius: 15px; padding: 10px; display: inline-block;">{q}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: left; font-weight: bold;">Agent：</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: left; background-color: #EDEDED; border-radius: 15px; padding: 10px; display: inline-block;">{r}</div>', unsafe_allow_html=True)

# Initialize the dialogue history if not already in session_state
if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []

st.markdown("---")

# Create a scrollable container for the chat
with st.container():
    st.write("This is the Lihao customer service version 1.0, and in the future it will be updated with more powerful features.")
    st.write("When 'Exit' is entered, the dialog will be terminated.")
    for q, r in st.session_state.dialogue_history:
        display_chat(q, r)

st.markdown("---")

# Set up the OpenAI API
openai.api_key = ""

def classify_input(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Label this sentence as {1: Reservation, 2: Payment, 3: Check out, 4: Another}return is one of number"+ prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.1,
    )


    label = response.choices[0].text.strip()
    label_int = int(re.search(r'\d+', label).group())
    return label_int

# Create input and submit sections at the bottom
input_container = st.container()
with input_container:
    col1, col2 = st.columns(2)
    with col1:
        question = st.text_input("Message：")
    with col2:
        if st.button("SEND"):
            if question.lower() == "exit":
                st.session_state.dialogue_history = []
            else:
                classification = classify_input(question)
                #classification=re.sub(r'\D', '', classification)
                
                if classification == 1:
                    response = "We are currently unable to handle reservations, If you would like to make a reservation, you can contact us by calling our phone number 1234567789"
                else:
                    response = user_AskQuestion(question)

                # Update the dialogue history with the new question and response
                if question and response:
                    st.session_state.dialogue_history.append((question, response))

                    # Refresh the dialogue display
                    st.experimental_rerun()
                   
