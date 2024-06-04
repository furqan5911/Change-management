import os
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

def assessment_message(userid, chatid, user_question, formated_history, embeddings, generalInfo, bussinessInfo, assessmentType):
    model = ChatOpenAI(model_name='gpt-4o', temperature=0.9, max_tokens=4096)
    vectorstore = PineconeVectorStore(index_name='quickstart', embedding=embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY"), namespace=chatid)
    retriever = vectorstore.as_retriever()

    template = """You are an expert in change management. You are given the following general information about an organization:
    Given background Change Management information collected from Form {general_info} and User info at the time of registration {bussiness_info} consider this as a part of what we already know about the user and firm so do not use anything asked or taken before even the headings for these forms.
    Right now user wants to perform a {assessment_type} Assessment. 

    Ask main and most necessary questions for {assessment_type} Assessment and make sure to do it one by one based on user giving response for each proceeding the questions. 
    
    Asked {assessment_type} assessment questions should gather the required information for a Outcome evaluation report generation for {assessment_type} assessment which would not only be professional but covers the knowledge of how Change management is being done. 
    
    Outcome ready to handover to the user in the end. 
    every time when user answers any questions ask if he is interested to get the {assessment_type} assessment report generated right now or not 
    The generated detailed report should cover aspect in the Following categories.

    Based on the provided information, generate a comprehensive {assessment_type} assessment report covering all relevant aspects for this particular assessment keeping international standards in mind. 
    Ensure the report is tailored to the type of assessment and The output report format should be according to the {assessment_type} output design contain client name company type company name and some suggestion in the end and that's it.

    Start by asking the user the necessary questions to gather all the information needed for the {assessment_type} assessment.
    {context}
    User responses based on your queries: {question}
    This is all previous chat history: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
            "general_info": itemgetter("general_info"),
            "bussiness_info": itemgetter("bussiness_info"),
            "assessment_type": itemgetter("assessment_type"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_question, 
        "chat_history": formated_history, 
        "general_info": generalInfo, 
        "bussiness_info": bussinessInfo, 
        "assessment_type": assessmentType
    })

def survey_message(userid, chatid, user_question, formated_history, embeddings, generalInfo, bussinessInfo, surveyType):
    model = ChatOpenAI(model_name='gpt-4o', temperature=0.9, max_tokens=4096)
    vectorstore = PineconeVectorStore(index_name='quickstart', embedding=embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY"), namespace=chatid)
    retriever = vectorstore.as_retriever()

    template = """You are an expert in change management. You are given the following general information about an organization:
    Given background Change Management information collected from Form {general_info} and User info at the time of registration {bussiness_info} consider this as a part of what we already know about the user and firm so do not use anything asked or taken before even the headings for these forms.
    Right now user wants to perform a {survey_type} asssessment in the light of change management process.

    Ask main and most necessary questions for {survey_type}  asssessment and make sure ask all at once from user. Asked {survey_type} survey questions in form of survey with options as realistic and based on prior info feeded and in light of change management {survey_type} so you should gather the required information for a Outcome evaluation report generation for {survey_type} asssessment which would not only be professional but covers the knowledge of how Change management is being done.
    After the responses have been given now its time to make a detailed comprehensive report in the light of Change management Assessment {survey_type}.The generated detailed report should cover aspect in the Following things.
    {survey_type} output design contain international standards which personalized in a way that it contain client name company type company name and some suggestion in the end and that's it.and no need to show the survey questions and respective response on this outcome just the report generated.
    {context}
    User responses based on your queries: {question}
    This is all previous chat history: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
            "general_info": itemgetter("general_info"),
            "bussiness_info": itemgetter("bussiness_info"),
            "survey_type": itemgetter("survey_type"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_question, 
        "chat_history": formated_history, 
        "general_info": generalInfo, 
        "bussiness_info": bussinessInfo, 
        "survey_type": surveyType
    })

def checks_message(userid, chatid, user_question, formated_history, embeddings, generalInfo, bussinessInfo, checkType):
    model = ChatOpenAI(model_name='gpt-4o', temperature=0.9, max_tokens=4096)
    vectorstore = PineconeVectorStore(index_name='quickstart', embedding=embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY"), namespace=chatid)
    retriever = vectorstore.as_retriever()

    template = """You are an expert in change management. You are given the following general information about an organization:
    Given background Change Management information collected from Form {general_info} and User info at the time of registration {bussiness_info} consider this as a part of what we already know about the user and firm so do not use anything asked or taken before even the headings for these forms.
    Right now user wants to perform a {check_type} asssessment in the light of change management process.

    Ask main and most necessary questions for {check_type}  asssessment and make sure ask all at once from user. Asked {check_type} survey questions in form of checks that would be useful for this with options as yes or no based on the Change mamangement sops for the particular assessment and keeping knowledge of prior info feeded and in light of change management {check_type} so you should gather the required information for a Outcome evaluation report generation for {check_type} asssessment which would not only be professional but covers the knowledge of how Change management is being done.
    After the responses have been given now its time to make a detailed comprehensive report in the light of Change management Assessment {check_type}.The generated detailed report should cover aspect in the Following things.
    {check_type} output design contain international standards which personalized in a way that it contain client name company type company name and some suggestion in the end and that's it.and no need to show the survey questions and respective response on this outcome just the report generated. and it should not contain the responses the user gave but the assessment being made on this in the light of change management and international Standards of {check_type} and report credit is to ChangeAI
    {context}
    User responses based on your queries: {question}
    This is all previous chat history: {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
            "general_info": itemgetter("general_info"),
            "bussiness_info": itemgetter("bussiness_info"),
            "check_type": itemgetter("check_type"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_question, 
        "chat_history": formated_history, 
        "general_info": generalInfo, 
        "bussiness_info": bussinessInfo, 
        "check_type": checkType
    })
