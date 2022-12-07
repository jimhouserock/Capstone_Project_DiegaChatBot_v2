import streamlit as st
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle
from streamlit_chat import message as st_message
from PIL import Image


with open("intents.json") as file:
    data = json.load(file)

model = keras.models.load_model('chat_model')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# parameters
max_len = 20

image = Image.open('images.png')
st.image(image, width=150)
st.title("Diega, Le Wagon Web Assistant")



#tag = lbl_encoder.inverse_transform([np.argmax(result)])
if "history" not in st.session_state:
    st.session_state.history = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def generate_answer():

    user_message = st.session_state.input_text

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_message]),
                                             truncating='post', maxlen=max_len))


    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
            if i['tag'] == tag:
                out_message=np.random.choice(i['responses'])

                #st.session_state.history.append({"message": out_message, "is_user": False})
                #st.session_state.history.append({"message": user_message, "is_user": True})
                st.session_state.past.append(user_message)
                st.session_state.generated.append(out_message)
                #print (np.random.choice(i['responses']))
                st.session_state["input_text"] = ""



st.text_input("Type your questions below: ", key="input_text", on_change=generate_answer)



if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st_message(st.session_state["generated"][i], is_user=False, avatar_style="bottts",seed="11", key=str(i))
        st_message(st.session_state['past'][i], is_user=True,avatar_style="adventurer-neutral",seed="2", key=str(i) + '_user')

#for chat in st.session_state.history:
#    st_message(**chat)  # unpacking
