import streamlit as st
from transformers import AutoTokenizer

st.title("AI Tokenizer Visualizer")

text = st.text_area("Enter text")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

if st.button("Tokenize"):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    st.subheader("Tokens")
    colored_tokens = " ".join([f":blue[{token}]" for token in tokens])
    st.markdown(colored_tokens)

    st.subheader("Token IDs")
    st.write(ids)

    st.subheader("Token Count")
    st.write(len(tokens))