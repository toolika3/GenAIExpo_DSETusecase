import streamlit as st
from streamlit_chat import message
from langchain_experimental.agents import create_csv_agent
import boto3  # AWS SDK for Python(integrate AWS services for Python)
from langchain_aws import ChatBedrock
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import pandas as pd

# Initialise boto session
boto_session = boto3.Session()
aws_region = boto_session.region_name
# runtime client to call Bedrock models
br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)

llm = ChatBedrock(
                      model_id='anthropic.claude-3-sonnet-20240229-v1:0',
                      model_kwargs={"temperature": 0.12, "max_tokens": 2000},
                      client=br_runtime
        )

agent = create_csv_agent(llm,
                         'data/consolidated_dset_file.csv',
                         verbose=True,
                         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
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
        result = agent.run(user_input)
        #st.write(result)
        st.session_state['history'].append((user_input, result))
        return result

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


if __name__=="__main__":
    main()

