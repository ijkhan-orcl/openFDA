import streamlit as st
import langCode
import RAG

st.title("openFDA Bot")
langCode.upload_pdf()

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])
                    
prompt=st.chat_input("Enter question")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append(
        {
        "role":"user",
        "content":prompt
        }
    )

    response=langCode.chat(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )