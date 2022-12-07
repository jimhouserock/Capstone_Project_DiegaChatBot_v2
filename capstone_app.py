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



#tag = lbl_encoder.inverse_transform([np.argmax(result)])
if "history" not in st.session_state:
    st.session_state.history = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


# message = input("")
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)


def generate_answer():
    user_message = st.session_state.input_text
    ints = predict_class(user_message)
    out_message = get_response(ints, intents)
    st.session_state.past.append(user_message)
    st.session_state.generated.append(out_message)
    st.session_state["input_text"] = ""

# def generate_answer():

#     user_message = st.session_state.input_text

#     result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_message]),
#                                              truncating='post', maxlen=max_len))


    # tag = lbl_encoder.inverse_transform([np.argmax(result)])

    # for i in data['intents']:
    #         if i['tag'] == tag:
    #             out_message=np.random.choice(i['responses'])

    #             #st.session_state.history.append({"message": out_message, "is_user": False})
    #             #st.session_state.history.append({"message": user_message, "is_user": True})
    #             st.session_state.past.append(user_message)
    #             st.session_state.generated.append(out_message)
    #             #print (np.random.choice(i['responses']))
    #             st.session_state["input_text"] = ""



st.text_input("Type your questions below: ", key="input_text", on_change=generate_answer)



if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st_message(st.session_state["generated"][i], is_user=False, avatar_style="bottts",seed="11", key=str(i))
        st_message(st.session_state['past'][i], is_user=True,avatar_style="adventurer-neutral",seed="2", key=str(i) + '_user')

#for chat in st.session_state.history:
#    st_message(**chat)  # unpacking
