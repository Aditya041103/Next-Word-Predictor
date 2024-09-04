import streamlit as st
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load your trained model from the pickle file
with open('train_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your tokenizer from a pickle file (if required)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define max_len (as per your model's requirements)
max_len = 20  # You can adjust this as per your model

def predict_next_words(query, num_words=10):
    for i in range(num_words):
        # Convert the input text to sequences
        token_text = tokenizer.texts_to_sequences([query])[0]
        
        # Pad the sequences
        padded_token_text = pad_sequences([token_text], maxlen=max_len, padding='pre')
        
        # Predict the next word
        pos = np.argmax(model.predict(padded_token_text))
        
        # Find the corresponding word from the tokenizer
        for word, index in tokenizer.word_index.items():
            if index == pos:
                query = query + " " + word
                break  # Break the loop once the word is found
    
    return query

# Streamlit UI
st.title("Next Word Predictor")

# Input field for user to enter the first few words
input_text = st.text_input("Enter the first few words:")

if st.button("Predict"):
    if input_text:
        # Predict the next 10 words
        predicted_text = predict_next_words(input_text, num_words=10)
        
        # Display the predicted text
        st.write("Predicted Text: ", predicted_text)
    else:
        st.write("Please enter some words to get predictions.")
