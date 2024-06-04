import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from handlers import assessment_message, survey_message, checks_message
from utils import load_environment_variables

# Load environment variables
load_environment_variables()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None
if 'selected_category' not in st.session_state:
    st.session_state['selected_category'] = None

# Input fields for the form
userid = st.text_input('User ID')
chatid = st.text_input('Chat ID')
generalInfo = st.text_area('General Info')
bussinessInfo = st.text_area('Business Info')

# Define categories and their options
categories = ['Q&A Assessments', 'Surveys', 'Checks']
qa_assessment_options = ['Change vision/case for change', 'Change approach/strategy', 'Change impact assessment', 'ADKAR assessment', 'Training assessment', 'What’s changing and what is not summary', 'Readiness assessment', 'Managing Resistance']
survey_options = ['Champions survey', 'Users survey', 'Training feedback survey', 'Post-Go Live Survey']
check_options = ['Communications plan', 'Engagement plan', 'Key messages by stakeholder group', 'Briefing messages', 'Benefits/Adoption KPIs', 'Change KPIs/user adoption statistics', 'Communications messages', 'FAQ’s', 'Health check']

# Display categories
selected_category = st.selectbox('Select a Category', categories)

# Display options based on the selected category
if selected_category == 'Q&A Assessments':
    selected_option = st.selectbox('Select an Assessment Type', qa_assessment_options)
elif selected_category == 'Surveys':
    selected_option = st.selectbox('Select a Survey Type', survey_options)
elif selected_category == 'Checks':
    selected_option = st.selectbox('Select a Check Type', check_options)

# Set the selected option in session state when the category and option are selected
if st.button('Select and Start'):
    st.session_state['selected_option'] = selected_option
    st.session_state['selected_category'] = selected_category
    st.session_state['chat_history'].append(f"Selected Option: {selected_option}")

# Function to handle the chat logic
def handle_chat(userid, chatid, selected_option, selected_category, embeddings, user_question, chat_history, generalInfo, bussinessInfo):
    if selected_category == 'Q&A Assessments':
        return assessment_message(userid, chatid, user_question, chat_history, embeddings, generalInfo, bussinessInfo, selected_option)
    elif selected_category == 'Surveys':
        return survey_message(userid, chatid, user_question, chat_history, embeddings, generalInfo, bussinessInfo, selected_option)
    elif selected_category == 'Checks':
        return checks_message(userid, chatid, user_question, chat_history, embeddings, generalInfo, bussinessInfo, selected_option)

# Display the chat history
st.subheader("Chat History")
for entry in st.session_state['chat_history']:
    st.write(entry)

# User input for the conversation
chat_message = st.chat_input("Say something")

# Generate response based on the selected category and option
if chat_message:
    st.chat_message("user").markdown(chat_message)
    res_area = st.chat_message("assistant").markdown("...")

    embeddings = OpenAIEmbeddings()
    chat_history = st.session_state['chat_history']
    response = handle_chat(userid, chatid, st.session_state['selected_option'], st.session_state['selected_category'], embeddings, chat_message, chat_history, generalInfo, bussinessInfo)
    
    # Update chat history
    st.session_state['chat_history'].append(f"User: {chat_message}")
    st.session_state['chat_history'].append(f"AI: {response}")
    res_area.markdown(response)

def generate_report(userid, chatid, chat_history, generalInfo, bussinessInfo, selected_option, selected_category):
    model = ChatOpenAI(model_name='gpt-4', temperature=0.7, max_tokens=4096)
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name='quickstart', embedding=embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY"), namespace=chatid)
    retriever = vectorstore.as_retriever()

    template = """
    You are an expert in change management. You are given the following general information about an organization:
    General Info: {general_info}
    Business Info: {bussiness_info}
    Chat History: {chat_history}

    Based on the information provided and the selected option {selected_option} in category {selected_category}, generate a comprehensive report.
    The report should cover all relevant aspects of the selected category and option, ensuring it adheres to international standards.
    """

    prompt = ChatPromptTemplate.from_template(template)

    context = {
        "general_info": generalInfo,
        "bussiness_info": bussinessInfo,
        "chat_history": "\n".join(chat_history),
        "selected_option": selected_option,
        "selected_category": selected_category
    }

    # Prepare the final prompt with the context
    final_prompt = prompt.format(**context)

    # Invoke the model with the final prompt
    response = model.predict(final_prompt)

    return response

# Button to generate the final report
if st.button('Generate Report'):
    report = generate_report(
        userid,
        chatid,
        st.session_state['chat_history'],
        generalInfo,
        bussinessInfo,
        st.session_state['selected_option'],
        st.session_state['selected_category']
    )
    st.subheader("Generated Report")
    st.write(report)
