import streamlit as st
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle
from streamlit_chat import message as st_message
from PIL import Image
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from twilio.rest import Client


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


lemmatizer = WordNetLemmatizer()

# loading the files we made previously
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)
                      for word in sentence_words]
    return sentence_words

def bagw(sentence):

	# separate out words from the input sentence
	sentence_words = clean_up_sentences(sentence)
	bag = [0]*len(words)
	for w in sentence_words:
		for i, word in enumerate(words):

			# check whether the word
			# is present in the input as well
			if word == w:

				# as the list of words
				# created earlier.
				bag[i] = 1

	# return a numpy array
	return np.array(bag)

def predict_class(sentence):
	bow = bagw(sentence)
	res = model.predict(np.array([bow]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i, r] for i, r in enumerate(res)
			if r > ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent': classes[r[0]],
							'probability': str(r[1])})
		return return_list

def get_response(intents_list, intents_json):
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']
	result = ""
	for i in list_of_intents:
		if i['tag'] == tag:

			# prints a random response
			result = random.choice(i['responses'])
			break
	return result


image = Image.open('images.png')
st.image(image, width=150)
st.title("Diega, Le Wagon Web Assistant v2")



if "history" not in st.session_state:
    st.session_state.history = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []



def generate_answer():
    user_message = st.session_state.input_text

    if (user_message).upper()=="CALL DIEGO":
        # Download the helper library from https://www.twilio.com/docs/python/install
        account_sid = st.secrets["account_sid"]
        auth_token = st.secrets["auth_token"]

        client = Client(account_sid, auth_token)

        call = client.calls.create(

                                to='+529871182931',
                                from_='+17208066079'
                            )
        st.session_state.past.append(user_message)
        st.session_state.generated.append("Sure, let me call my favourite person for you!")
    else:
        ints = predict_class(user_message)
        out_message = get_response(ints, intents)
        st.session_state.past.append(user_message)
        st.session_state.generated.append(out_message)

    st.session_state["input_text"] = ""

st.text_input("Type your questions below: (Type 'Call Diego' if you want to call our Admission Manager, Diego, right away)", key="input_text", on_change=generate_answer)



if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st_message(st.session_state["generated"][i], is_user=False, avatar_style="bottts",seed="11", key=str(i))
        st_message(st.session_state['past'][i], is_user=True,avatar_style="adventurer-neutral",seed="2", key=str(i) + '_user')
