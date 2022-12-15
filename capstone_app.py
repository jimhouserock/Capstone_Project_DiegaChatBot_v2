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
import re
import webbrowser
import time
from streamlit_extras.let_it_rain import rain
from PIL import Image


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


#Page Icon
st.set_page_config(page_title="Diega is here!", page_icon=":robot:")

# Images are shown: Le Wagon and Diega
col1, col2 = st.columns([2,1])
with col1:
    image = Image.open('images.png')
    st.image(image, caption='',width=200)
with col2:
    st.image("https://avatars.dicebear.com/api/bottts/11.svg", width=50)


# Page Title
st.title("Diega, Le Wagon Web Assistant")
# st.image("https://avatars.dicebear.com/api/bottts/11.svg", width=50)

# Welcome message
# st.write(
#        """
#        **Welcome to Le Wagon Mexico. My name is Diega. I am here to assist you.**

#        ℹ️ Please be patient, I am still learning. Thank you! ❤️
#        """
#    )

# Balloons falling
# st.balloons()

# Rains Mexican flags
rain(
    emoji="🎅",
    font_size=50,
    falling_speed=5,
    animation_length=1,
)

#Hide Main Menu and Footer
hide_menu_style = """
<style>
#MainMenu {visibility: hidden; }
footer {visibility: hidden;}
</style
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)


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
                                url='http://demo.twilio.com/docs/voice.xml',
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

        # Search if the bot response contains "address" and create a clikable button for the page link
        search_tag = re.search(r"address",out_message)
        if search_tag:
            url = 'https://www.tinyurl.com/Le-Wagon-CDMX'
            st.markdown(f'''
            <a href={url}><button style="text-align: justify;position:fixed;font-size:25px;background-color:RedWhite;">🏫 Click to see Le Wagon 🇲🇽 on the map</button></a>
            ''',unsafe_allow_html=True)

    st.session_state["input_text"] = ""

st.text_input("Type your questions below: (Type 'Call Diego' if you want to call our Admission Manager, Diego, right away)", key="input_text", on_change=generate_answer)



if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st_message(st.session_state["generated"][i], is_user=False, avatar_style="bottts",seed="11", key=str(i))
        st_message(st.session_state['past'][i], is_user=True,avatar_style="adventurer-neutral",seed="2", key=str(i) + '_user')


# Widget "Loaded successfully" appears each input
with st.spinner("Loading..."):
    time.sleep(0.25)
st.success("Loaded successfully")

# Inserts GitHub badge and is linked to our project repository
st.write('''
         [[![GitHub watchers](https://img.shields.io/github/watchers/jimhouserock/Capstone_Project_DiegaChatBot_v2?label=github&style=social)](https://github.com/jimhouserock/Capstone_Project_DiegaChatBot_v2)]
         ''')
