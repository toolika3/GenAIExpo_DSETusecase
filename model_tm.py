import boto3  # AWS SDK for Python(integrate AWS services for Python)
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import BedrockEmbeddings # To access Bedrock's Titan Text embedding model.BedrockEmbeddings uses AWS Bedrock API to generate embeddings for given text for saving to vectordb 
from langchain_community.vectorstores.faiss import FAISS
from langchain_aws import ChatBedrock
from langchain.chains import ConversationalRetrievalChain  # Langchain fuction to create chatbot that can remember past interactions
import streamlit as st
from streamlit_chat import message


DB_FAISS_PATH = 'vectorstore/db_faiss'

# Initialise boto session
boto_session = boto3.Session()
aws_region = boto_session.region_name
# runtime client to call Bedrock models
br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)

#Loading the model
def load_llm():
    llm = ChatBedrock(
                      model_id='anthropic.claude-3-sonnet-20240229-v1:0',
                      model_kwargs={"temperature": 0.12, "max_tokens": 2000},
                      client=br_runtime
        )
    return llm


custom_prompt_template = """
    You are a Data Discovery AI assistance created by Fannie Mae.Your goal is to help users with data related questions. 
    You should maintain a friendly customer service tone.

    Here are some important rules for the interaction:
    - Use English language for answering questions.
    - Always stay in charater
    - Look across all datasets before answering the question.
    - Provide top 20 records on questions with multiple records
    - If someone asks irrelevant question, say, I am Data Discovery AI assistant and can't help with above question.
    - Be specific and don't add extra information in the response that is not asked for.

    How do you respond to user's question?

    Think about your answer first before you respond.You will look around all datasets and will use the context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Chat History: {chat_history}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question', 'chat_history'])
    return prompt


embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=br_runtime)
db = FAISS.load_local(DB_FAISS_PATH, embeddings)
llm = load_llm()
prompt = set_custom_prompt()

qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                 chain_type='stuff',
                                                 retriever=db.as_retriever(search_kwargs={'k':30}),
                                                 verbose=True,
                                                 #retriever=db.as_retriever(search_type="mmr"),
                                                 #retriever=db.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold':0.1}),
                                                 combine_docs_chain_kwargs={'prompt': prompt},
                                                 return_source_documents=True,
                                                 output_key='answer'
                                                )

def main():
    st.markdown(
        """
        <link rel="stylesheet" type="text/css" href="/styles.css" />
        """,
        unsafe_allow_html=True)
    st.markdown("""
                # Data Discovery Chatbot
                """)

    def conversational_chat(user_input):
        result = qa_chain({"question": user_input, "chat_history": st.session_state['history']})
        #st.write(result)
        st.session_state['history'].append((user_input, result["answer"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    #Container for chat history
    response_container = st.container()
    # container for user text
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):

            user_input = st.text_input("Query:", placeholder="Enter your question", key='input')
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = conversational_chat(user_input)
            # store output
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="croodles")
                message(st.session_state["generated"][i], key=str(i), avatar_style="micah")


if __name__ == "__main__":
    main()

