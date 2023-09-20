import os
import tensorflow as tf
import numpy as np
import pandas as pd
import transformers
from summarizer import Summarizer, TransformerSummarizer
import streamlit as st
import joblib

#Defining bert model
model_bert = Summarizer()

#helper  function
def summarize(model, text):
    print("Length of text: ",len(text))
    summary = ''.join(model(text, min_length=40))
    st.success(f'{summary}')
    print("Length: ",len(summary))

st.title('Welcome to Shortcut')
image = 'summary.png'  
st.image(image,width=500, caption='Image from https://storyset.com/illustration/notes/bro#default&hide=&hide=simple')
name = st.text_input('Enter your name', '')

if(name):
    st.warning(f'''**Welcome {name}! Please note that this is text summarization tool which uses bert model. Please use only for text. Also, if bullets or numbering are included your text, it won't be quite accurate!**''', icon="⚠️")
    text = st.text_area('Text to analyze', '''Remove this text and add text that you want to summarize!''')
    button_clicked = st.button("Summarize")
    if button_clicked:
        summarize(model_bert, text)

    

#Defining gpt2 model
#model_gpt2 = TransformerSummarizer(transformer_type = "GPT2", transformer_model_key="gpt2-medium")

#Defining xlnet model
#model_xlnet = TransformerSummarizer(transformer_type = "XLNet", transformer_model_key = "xlnet-base-cased")
 
#summarize(model_bert, text)
#summarize(model_gpt2, text)
#summarize(model_xlnet, text)

