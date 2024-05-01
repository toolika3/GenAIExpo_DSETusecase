import boto3  # AWS SDK for Python(integrate AWS services for Python)
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import BedrockEmbeddings # To access Bedrock's Titan Text embedding model.BedrockEmbeddings uses AWS Bedrock API to generate embeddings for given text for saving to vectordb 
from langchain_community.vectorstores.faiss import FAISS
from langchain_aws import ChatBedrock
from langchain.chains import ConversationalRetrievalChain  # Langchain fuction to create chatbot that can remember past interactions
import streamlit as st
from streamlit_chat import message
import pandas as pd  # to create a dataframe for displaying all models from Bedrock
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.app_logo import add_logo

st.set_page_config(
    layout="wide"
)

@st.cache_resource
def initialize_model():

    DB_FAISS_PATH = 'vectorstore/db_faiss'

    # Initialise boto session
    boto_session = boto3.Session()
    aws_region = boto_session.region_name
    # runtime client to call Bedrock models
    br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)
    embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=br_runtime)

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
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k':30}),
        #retriever=db.as_retriever(search_type="mmr"),
        #retriever=db.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold':0.1}),
        combine_docs_chain_kwargs={'prompt': prompt},
        return_source_documents=True,
        output_key='answer'
    )
    return(qa_chain)

qa_chain = initialize_model()

def main():

    # Build function to get the response given a question
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

    # Create layout for different components
    col1, gap, col3 = st.columns([50, 1, 30])

    # Static component
    with col1:

        # Build the icon + title
        c1, mid, c2 = st.columns([1, 3, 40])
        with c1:
            st.image("images.png", width = 80)
        with c2:
            st.title("FNMA AI Super Search Dictionary")
    
        # Application description
        st.write("View a wide range of data details and explore the depths of Fannie Mae data with AI powered super search technology!")
    
        add_vertical_space(2)
        
        col1.header("Data Dictionary")
        tab1, tab2, tab3 = st.tabs(["Datasets", "Columns", "Profile"])

        tab1.subheader("Dataset Summary")
        df = pd.read_csv("data/dataset_metadata.csv")
        df = df[['Dataset Name', 'Table Name', 'Dataset Reg ID', 'Dataset Description']]
        df["show"] = False
        
        table = tab1.data_editor(
            df,
            column_config = {
                "show": st.column_config.CheckboxColumn(
                    "Show Columns",
                    help = "Select to show the columns in this table",
                    default = False
                )
            },
            disabled = ['Dataset Name', 'Table Name', 'Dataset Reg ID', 'Dataset Description'],
            hide_index= True
        )
        
        selected_tables = table.loc[table['show'] == True]
        
        tab2.subheader("Column Summary")
        df2 = pd.read_csv("data/attribute_metadata.csv")
        df2 = df2[['Physical Attribute Name', 'Logical Attribute Name', 'Description', 'Dataset Reg ID', 'DataType']]
        
        col_df = selected_tables.merge(df2, on='Dataset Reg ID', how = 'left')
        col_df = col_df[['Table Name', 'Physical Attribute Name', 'Logical Attribute Name', 'Description', 'DataType']]
        col_df['show'] = False
        
        table2 = tab2.data_editor(
            col_df,
            column_config = {
                "show": st.column_config.CheckboxColumn(
                    "Show Profile",
                    help = "Select to show the profile for this column",
                    default = False
                )
            },
            disabled = ['Table Name', 'Physical Attribute Name', 'Logical Attribute Name', 'Description', 'DataType'],
            hide_index= True
        )
        
        tab3.subheader("Profile Summary")
        prof_tables = table2.loc[table2['show'] == True]
        
        if prof_tables.shape[0] > 0:
            selected = prof_tables.iloc[0]['Physical Attribute Name']
            tab3.text(f"Showing profile information for column: {selected}")
            tab3.text("")
            tab3.text("Attribute Type: Continuous Number")
            tab3.text("Minimum Value: 0")
            tab3.text("Median Value: 500")
            tab3.text("Maximum Value: 1000")    
            
    # Chatbot Component
    with col3:
        col3.header("AI SuperSearch")
        overall_container = st.container(border=True)
        with overall_container:

            # Container for chat history
            response_container = st.container(height = 450)

            # container for user text
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):

                    user_input = st.text_input("Query:", placeholder="Ask questions here", key='input')
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
