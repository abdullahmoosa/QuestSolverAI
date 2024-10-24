import nltk
import time
import os
import openai
from backend.services.answer_generation import AnswerGeneration

# nltk.download("punkt")
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Library for Reading Template file
import docx
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def docx_to_markdown(docx_path):
    doc = docx.Document(docx_path)
    markdown_content = []

    def convert_table_to_markdown(table):
        markdown_table = []
        max_col_widths = [0] * len(table.rows[0].cells)

        # First pass: determine maximum width for each column
        for row in table.rows:
            for i, cell in enumerate(row.cells):
                max_col_widths[i] = max(max_col_widths[i], len(cell.text.strip()))

        # Second pass: create markdown table
        for i, row in enumerate(table.rows):
            markdown_row = []
            for j, cell in enumerate(row.cells):
                cell_content = cell.text.strip().ljust(max_col_widths[j])
                markdown_row.append(cell_content)
            markdown_table.append("| " + " | ".join(markdown_row) + " |")

            # Add header separator after first row
            if i == 0:
                separator = (
                    "|"
                    + "|".join(["-" * (width + 2) for width in max_col_widths])
                    + "|"
                )
                markdown_table.append(separator)

        return "\n".join(markdown_table)

    for element in doc.element.body:
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, doc)
            if paragraph.text.strip():  # Only add non-empty paragraphs
                markdown_content.append(paragraph.text)
        elif isinstance(element, CT_Tbl):
            table = Table(element, doc)
            markdown_content.append(convert_table_to_markdown(table))
            markdown_content.append("")  # Add an empty line after the table

    # Handle text boxes (shapes)
    for shape in doc.inline_shapes:
        if shape.has_text_frame:
            for paragraph in shape.text_frame.paragraphs:
                if paragraph.text.strip():  # Only add non-empty paragraphs
                    markdown_content.append(f"[Text Box] {paragraph.text}")
    return "\n\n".join(markdown_content)


def load_docx(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=2500)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        collection_name="local-project-rag",
    )
    return vector_db


def setup_llm():
    # local_model = "llama3.1:latest"
    # llm = ChatOpenAI(temperature=0.00001)
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0,)
    return llm


def setup_activity_llm():
    llm = ChatOpenAI(temperature=0.2)
    return llm


def project_info_extraction_pipeline(vector_store, llm):
    # query = """
    # [INST] Based on the content of the document, extract the scenario and list down the activities with their requirements. For your help, document is structured like that:
    # Scenario: {scenario}

    # Make sure you didn't change any information
    # [/INST]
    # """
    query = """
    You are a professional document analyzer. I am giving you a document that contains specific instructions and tasks. 
    Please extract and organize the following information in a structured format:

    Scenario: Extract the project scenario with all the information without modification

    Be concise and to the point.
    """
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})

    return result["result"]

def project_activity_extraction_pipeline(vector_store, llm):
    # query = """
    # [INST] Based on the content of the document, extract the scenario and list down the activities with their requirements. For your help, document is structured like that:
    # Scenario: {scenario}

    # Make sure you didn't change any information
    # [/INST]
    # """
    query = """
    You are a professional document analyzer. I am giving you a document that contains specific activities.
    Please extract and organize the following information in a structured format:

    Activities: Extract the activities listed under Activities headline. Summarize to include only the vital information.
    Activities: [Provide a list of activities]
    Be concise and to the point.
    """
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})

    return result["result"]

# def generate_project_output(llm, project_info):
#     prompt = f"""
#     [INST] You are a creative scenario generator for project-based learning.Your task is to create or assume a detailed, elaborated, engaging scenario based on the provided project information and only include the scenario details. After generating the scenario, you have to complete the tasks.
#     Project Information: {project_info}

#     Include a detailed description of the event including who it is aimed at, the format of the day and where it will be held, as well as approximately how many event staff and participants there will be.
#     List at least 10 hazards. The hazards you list must include at least one actual or foreseeable hazard from the following list:
#         Physical environment
#         Plant/equipment
#         Work practice
#         Security issue
#     Describe each one and include the risk rating and a suggested risk control. Identify who is responsible. Ensure the risk rating is scored as per the risk legend included with this plan.Make sure you use markdown tabular format for the output. you need participate in a brief meeting with your assessor to discuss Hazard Identification and Risk Assessment Tool. Provide meeting minutes for the meeting with the assesor.


#     [/INST]
#     """
#     response = llm.invoke(prompt)
#     return response.content


def generate_project_output(llm, project_info, template_content):
    # prompt = f"""
    # [INST] You are a creative scenario generator for project-based learning.
    # Your task is to create or assume a detailed, elaborated, engaging scenario based on the provided project information and only include the scenario details. 
    # After generating the scenario, you have to complete the tasks by following the template structure. 
    # Project Information: {project_info}
    # Template Structure: {template_content}
    

    # [/INST]
    # """
    prompt = f"""
    [INST] You are a creative scenario generator for project-based learning.
    Your task is to analyze the Scenario and then answer the activities mentioned the Project Information according to the scenario. 
    If template is provided then use the template to answer the activities if it is suitable.
    Project Information: {project_info}
    Template Structure: {template_content}
    [/INST]
    """
    response = llm.invoke(prompt)
    return response.content

def generate_project_output_per_activities(llm, project_info, activities,template_content):
    # prompt = f"""
    # [INST] You are a creative scenario generator for project-based learning.
    # Your task is to create or assume a detailed, elaborated, engaging scenario based on the provided project information and only include the scenario details. 
    # After generating the scenario, you have to complete the tasks by following the template structure. 
    # Project Information: {project_info}
    # Template Structure: {template_content}
    

    # [/INST]
    # """
    prompt = f"""
    [INST] You are a creative scenario generator for project-based learning.
    Your task is to answer the activities mentioned the Activities base on the extractted scenario in 'Project Information'. 
    First understand the Project Information, Create a scenario. 
    Think you are the organizer of the event. Now, analyze the scenario. 
    Understand the conceptual meaning of each activities. If the activities ask you to discuss, access. Think of the activities as tasks assigned to you
    Then think yourself as a owner discussing with others, get feed back from others hypothetically, include those in the answer of the activities. Answer chronologically and in details as possible according to each activity.
    Use template.


    Project Information: {project_info}
    Activities : {activities}
    template : {template_content}
   
    [/INST]
    """
    response = llm.invoke(prompt)
    return response.content


def generate_project_output_with_guidelines_template(llm, project_info, template_content,guidelines):
    # prompt = f"""
    # [INST] You are a creative scenario generator for project-based learning.
    # Your task is to create or assume a detailed, elaborated, engaging scenario based on the provided project information and only include the scenario details. 
    # After generating the scenario, you have to complete the tasks by following the template structure. 
    # You will be given a sample extracted guidelines. You will answer similar to the guidelines but not exactly same.
    # Project Information: {project_info}
    # Template Structure: {template_content}
    # Guidelines : {guidelines}

    # [/INST]
    # """

    prompt = f"""
    [INST] You are a creative scenario generator for project-based learning.
    Your task is to analyze the Scenario and then answer the activities mentioned the Project Information according to the scenario. 
    Answer using the provided template.
    You will be given a sample extracted guidelines. You will answer similar to the guidelines but not exactly same.
    Project Information: {project_info}
    Template Structure: {template_content}
    Guidelines : {guidelines}

    output format :
    Scenario: [Write the scenario]
    Meeting Minutes : If asked to participate in a meeting with your assessor, provide draft meeting minutes for the meeting with the assesor.
    Template: [Answer activities according to template]
    
    [/INST]
    """
    response = llm.invoke(prompt)
    return response.content

def generate_project_output_per_activities_with_guidelines_template(llm, project_info, activities,template_content, guidelines):
    # prompt = f"""
    # [INST] You are a creative scenario generator for project-based learning.
    # Your task is to create or assume a detailed, elaborated, engaging scenario based on the provided project information and only include the scenario details. 
    # After generating the scenario, you have to complete the tasks by following the template structure. 
    # Project Information: {project_info}
    # Template Structure: {template_content}
    

    # [/INST]
    # """
    prompt = f"""
    [INST] You are a creative scenario generator for project-based learning.
    Your task is to answer the activities mentioned the Activities base on the extractted scenario in Project Information. 
    Provide detailed, elaborated, engaging scenario based on the provided project information per Activities and only include the scenario details. 
    If, template is provided then use the template to answer the activities if it is suitable.
    You will be given a sample extracted guidelines. You will answer similar to the guidelines but not exactly same.
    Project Information: {project_info}
    Activities : {activities}
    Template Structure: {template_content}
    Guidelines : {guidelines}
    Output format :
    Scenario: [Write the scenario]
    Activities: [Write the activities and answer each of them]
    [/INST]
    """
    response = llm.invoke(prompt)
    return response.content

def main():
    answer_generation = AnswerGeneration()
    project_tasks_file = "docs/SITXWHS006/SITXWHS006 Student Assessment Tasks 2 (Project) Short.docx"
    # relevant_docs_file = "../data/project/SITXWHS006 WHS Plan (Notes).docx"
    template_file = "docs/SITXWHS006/SITXWHS006 WHS Plan (Template).docx"
    documents = load_docx(project_tasks_file)
    chunks = split_documents(documents)
    vector_db = create_vector_store(chunks)

    llm = setup_llm()
    start_time = time.time()
    project_info = project_info_extraction_pipeline(vector_db, llm)
    project_activities = project_activity_extraction_pipeline(vector_db, llm)
    markdown_content = docx_to_markdown(template_file)
    print(project_info)
    print("-" * 40)
    print(project_activities)
    print("-" * 40)

    qa_chain_project_guidelines = answer_generation.create_question_extraction_pipeline(
        answer_generation.project_guidelines,
        llm
    )
    extracted_guidelines = answer_generation.extract_guidelines_from_template(
        qa_chain= qa_chain_project_guidelines, 
        template=markdown_content)
    with open("extracted_guidelines.md", "w") as file:
        file.write(extracted_guidelines)
    activity_llm = setup_activity_llm()
    # answer = generate_project_output(activity_llm, project_info, markdown_content)
    # answer = generate_project_output_with_guidelines_template(activity_llm, project_info, markdown_content, extracted_guidelines)
    answer = generate_project_output_per_activities(activity_llm, project_info, project_activities, markdown_content)
    # answer = generate_project_output_per_activities_with_guidelines_template(
    #     llm=activity_llm,
    #     project_info=project_info,
    #     activities=project_activities,
    #     template_content=markdown_content,
    #     guidelines=extracted_guidelines
    # )
    with open("output_18_oct_2024.md", "w") as file:
        file.write(answer)
    end_time = time.time()
    print(f"Time spent: {(end_time-start_time)/60} minutes")


if __name__ == "__main__":
    main()
