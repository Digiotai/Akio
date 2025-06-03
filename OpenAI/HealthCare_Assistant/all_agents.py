import os
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import sys
from dotenv import load_dotenv
import re

load_dotenv()

CONFIG = {
    "output_dir": "OpenAI/HealthCare_Assistant/scraping_output",
    "faiss_index_name": "combined_faiss_index",
    "embedding_model": "text-embedding-3-small",  # OpenAI embedding model
    "validation_threshold": 0.8  # Confidence threshold for validation
}

print("[DEBUG] Configuration loaded:", CONFIG)

# Load the FAISS index from the specified path
def load_faiss_index() -> FAISS:
    print("[DEBUG] Loading FAISS index...")
    index_path = os.path.join(CONFIG["output_dir"], CONFIG["faiss_index_name"])
    print(f"[DEBUG] Looking for index at: {index_path}")
    if os.path.exists(index_path):
        print("[DEBUG] Index found, loading OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(model=CONFIG["embedding_model"], openai_api_key=os.getenv("OPENAI_API_KEY"))
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    raise FileNotFoundError("FAISS index not found.")

# Initialize the LLM with OpenAI
def initialize_llm():
    print("[DEBUG] Initializing OpenAI Chat LLM...")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# Function to search for content in the FAISS index
def content_search(query: str, faiss_index: FAISS, content_type: str = None) -> List[Dict[str, Any]]:
    print(f"[DEBUG] Performing content search for query: '{query}'")
    if not faiss_index:
        raise ValueError("FAISS index is not loaded.")

    results = faiss_index.similarity_search_with_score(query, k=5)
    print(f"[DEBUG] Found {len(results)} results from FAISS index")
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "type": doc.metadata.get("type"),
            "score": float(score)
        }
        for doc, score in results
        if content_type is None or doc.metadata.get("type") == content_type
    ]



# Function to initialize the research agent
def Research_agent(faiss_index: FAISS):
    print("[DEBUG] Initializing Research Agent...")
    llm = initialize_llm()
    tool = Tool(
        name="Combined_Content_Search",
        func=lambda query: content_search(query, faiss_index, None),
        description="Use this tool to retrieve relevant content (PDFs and URLs) to answer the user's query. The documents returned should be used to reason and formulate a final answer."
    )
    return initialize_agent(
        [tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )


# Function to initialize the validation agent
def Validation_agent(faiss_index: FAISS):
    print("[DEBUG] Initializing Validation Agent...")
    llm = initialize_llm()

    tools = [
        Tool(
            name="Fact_Verification_Search",
            func=lambda query: content_search(query, faiss_index, None),
            description="Use this tool to verify facts in a given statement by searching for supporting evidence."
        ),
        Tool(
            name="Source_Check",
            func=lambda query: content_search(query, faiss_index, None),
            description="Use this tool to check if sources cited in a response are accurate and relevant."
        )
    ]

    return initialize_agent(
        tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )


# Function to validate the response from the research agent
def validate_response(response: str, query: str, validation_agent, max_attempts: int = 3) -> Dict[str, Any]:
    global validation_result
    print("\n=== VALIDATION PHASE ===")
    print(f"[DEBUG] Validating response for query: '{query}'")

    validation_prompt = f"""
    Please validate the following response to the query. Check for:
    1. Accuracy of facts
    2. Completeness of information
    3. Relevance to the original query
    4. Presence of supporting evidence

    Original Query: {query}
    Response to Validate: {response}

    Provide your validation as a detailed analysis. If you find issues, suggest improvements.
    """

    for attempt in range(1, max_attempts + 1):
        print(f"[DEBUG] Validation attempt {attempt}/{max_attempts}")
        validation_result = validation_agent.run(validation_prompt)
        print("\n[DEBUG] Validation result:", validation_result)

        if "approve" in validation_result.lower() or "accurate" in validation_result.lower():
            print("[DEBUG] Validation successful - response approved")
            return {
                "status": "approved",
                "validated_response": response,
                "validation_notes": validation_result
            }

        print(f"[DEBUG] Validation issues found:\n{validation_result}")

        improvement_prompt = f"""
        The following response to the query was found to have issues during validation.
        Please generate an improved version addressing these concerns:

        Validation Feedback: {validation_result}
        Original Query: {query}
        Original Response: {response}

        Generate an improved response that addresses all validation concerns.
        """

        improved_response = validation_agent.run(improvement_prompt)
        response = improved_response
        print(f"[DEBUG] Improved response generated: {response[:200]}...")  # Truncate preview

    print("[DEBUG] Maximum validation attempts reached - returning with issues noted")
    return {
        "status": "needs_review",
        "validated_response": response,
        "validation_notes": validation_result
    }


# Orchestrator function to handle the complete research and validation flow
def research_with_validation(query: str, faiss_index: FAISS):
    print("\n=== RESEARCH PHASE ===")
    print(f"[DEBUG] Processing query: '{query}'")

    print("[DEBUG] Initializing agents...")
    research_agent = Research_agent(faiss_index)
    validation_agent = Validation_agent(faiss_index)

    print("\n[DEBUG] Researching information...")
    research_response = research_agent.run(query)
    print("\n[DEBUG] Initial research response:")
    print(research_response)

    print("[DEBUG] Starting validation process...")
    validation_result = validate_response(research_response, query, validation_agent)

    if validation_result["status"] == "approved":
        print("\n=== FINAL VALIDATED RESPONSE ===")
        print(validation_result["validated_response"])
        return validation_result["validated_response"]
    else:
        print("\n=== FINAL RESPONSE (NEEDS REVIEW) ===")
        print("[DEBUG] The system generated a response but has some concerns about its accuracy:")
        print(validation_result["validated_response"])
        print("\n[DEBUG] Validation notes:")
        print(validation_result["validation_notes"])
        return validation_result["validated_response"]



# Reuse the LLM initialization
llm = initialize_llm()
def create_browser_tool() -> Tool:
    """
    Create the web search tool (currently just LLM-based).
    This tool simulates a browser by using the LLM to generate informed answers.
    """
    return Tool(
        name="Research Assistant",
        func=lambda query: llm.invoke(query),
        description="A tool that provides detailed research answers using an LLM. Use this when external knowledge or context is needed."
    )
#Resource agent code
def Resource_agent():
    """
    Initializes the Resource Agent that simulates real-time web search using the LLM.
    """
    print("[DEBUG] Initializing Resource Agent...")
    llm = initialize_llm()

    tools = [create_browser_tool()]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

#Data Agent(Text to sql):
# Data Agent(Text-to-sql agent)
# --- Prompt Tool ---
def prompt_tool_func(query: str) -> str:
    # --- Load CSV and metadata ---
    print("[DEBUG] Loading CSV data...")
    df = pd.read_csv('doctor_availability.csv')
    csv_metadata = {"columns": df.columns.tolist()}
    metadata_str = ", ".join(csv_metadata["columns"])
    print(f"[DEBUG] Loaded CSV with columns: {metadata_str}")

    prompt_eng = f"""
    You are a Python expert focused on answering user queries about data preprocessing. Always strictly adhere to the following rules:

    1. Generic Queries:
        If the user's query is generic and not related to data, respond with a concise and appropriate print statement.
        Example:
        Query: "What is AI?"
        Response: print("Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.")

    2. Data-Related Queries:
        The data file is 'doctor_availability.csv' with these columns: {metadata_str}.
        The Metadata includes:{csv_metadata}
        Respond with Python code only (no explanations) that:
        - Starts with: df = pd.read_csv('doctor_availability.csv')
        - Directly addresses the query
        - Includes comments for key steps
        - Ends with print() of the result
        Example:
        Query: "Filter rows where 'Available' is True"
        Response:
        # Load the dataset
        df = pd.read_csv('doctor_availability.csv')

        # Filter available doctors
        available_doctors = df[df['Available'] == True]

        # Show results
        print(available_doctors)

    3. Theoretical Concepts:
        Provide a brief explanation as a print statement.
        Example:
        Query: "What is normalization?"
        Response: print("Normalization scales numeric data to a specific range (typically [0,1]) to ensure equal feature contribution.")

    Never reply with confirmations - respond directly to the query.
    Current query: {query}
    Respond with Python code only for data-related queries.
    """

    # Code Generation using OpenAI
    generated_code = code_generation_func(prompt_eng)

    # Execute the generated code
    result = execute_py_code(generated_code, df)

    return result


# --- Code Generation Tool ---
def code_generation_func(prompt: str) -> str:
    print("[DEBUG] Generating code from prompt...")
    llm = initialize_llm()
    response = llm.invoke(prompt)
    content = response.content

    # Extract code block if present
    if "```python" in content:
        code_start = content.find("```python") + 9
        code_end = content.find("```", code_start)
        content = content[code_start:code_end].strip()

    return content


# --- Execution Tool ---
def execute_py_code(code: str, df: pd.DataFrame):
    buffer = io.StringIO()
    sys.stdout = buffer

    local_vars = {'df': df}

    try:
        exec(code, globals(), local_vars)
        output = buffer.getvalue().strip()

        # If no output, try last line
        if not output:
            last_line = code.strip().split('\n')[-1]
            if not last_line.startswith(('print', 'return')):
                output = eval(last_line, globals(), local_vars)
                print(output)
    except Exception as e:
        output = f"Error executing code: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__

    return str(output)


# --- Define Tools ---
print("[DEBUG] Creating Data Agent tools...")
prep_tool = Tool(name="DataProcessingTool", func=prompt_tool_func,
                 description="Tool for data processing with pandas.")  # Langchain Feature

# --- Data Agent with SQL---
# Define the agent initialization
def Data_agent():
    print("[DEBUG] Initializing Data Agent...")
    llm = initialize_llm()

    # Assuming prep_tool is defined globally or imported
    tools = [prep_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent


#Action agent Initialisation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from typing import Optional

def Action_agent(
    query: str,
    content_response: Optional[str] = "",
    browser_response: Optional[str] = "",
    data_response: Optional[str] = "",
    llm=None
) -> str:
    """
    Aggregates and formats responses from different agents into a final answer.
    """
    print("\n=== ACTION AGENT STARTING ===")
    if llm is None:
        llm = initialize_llm()

    system_prompt = """
    You are a medical AI assistant responsible for summarizing and formatting responses.
    The user asked a query, and different subsystems provided partial answers.

    Instructions:
    - Prioritize accuracy and relevance.
    - Use the validated RAG content as the main response if available.
    - Augment with browser search result if RAG is empty or needs more context.
    - Use Data Agent's output for queries regarding scheduling or tabular data.
    - Format the final response cleanly and clearly.

    Format:
    Final Response:
    <Clean, precise, and human-readable answer based on available data>

    Include which agents were used in the answer.
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        Query: {query}

        RAG Content:
        {content_response}

        Browser Content:
        {browser_response}

        Data Agent Content:
        {data_response}
        """)
    ])

    chain: Runnable = prompt_template | llm | StrOutputParser()

    final_response = chain.invoke({
        "query": query,
        "content_response": content_response or "None",
        "browser_response": browser_response or "None",
        "data_response": data_response or "None"
    })

    print("[DEBUG] Final formatted response ready.")
    return final_response.strip()

#Main agent logic orchestration
def classify_query(query: str, llm) -> str:
    """
    Classifies the user query to determine appropriate agent routing.
    Categories:
    - medical
    - appointment
    - non-medical
    """
    print(f"[DEBUG] Classifying query: '{query}'")

    system_prompt = """
    You are a classifier that determines the type of user query. Classify it into ONLY one of the following categories:

    1. medical – Any query about health, symptoms, conditions, diseases, diagnosis, medications, or treatments.
       Examples:
       - What are the symptoms of dengue?
       - How is diabetes treated?
       - What medicine should I take for high fever?

    2. appointment – Any query related to scheduling, finding doctors, doctor/hospital availability, hospital timings, or lists of medical professionals.
       Examples:
       - Book an appointment with a skin specialist.
       - List dermatologists in Mumbai.
       - What time does Apollo Hospital open?

    3. non-medical – General knowledge questions or anything unrelated to health or scheduling.
       Examples:
       - Who won the 2024 elections?
       - How does a rocket work?
       - Tell me about the Eiffel Tower.

    IMPORTANT:
    - Return ONLY ONE WORD: medical, appointment, or non-medical.
    - Be strict: do not invent new categories.
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Query: {query}")
    ])
    chain: Runnable = chat_prompt | llm | StrOutputParser()
    classification = chain.invoke({"query": query}).strip().lower()

    print(f"[DEBUG] Query classified as: {classification}")
    return classification


#Main agent Design.
def Main_agent(query: str, faiss_index) -> str:
    print("\n=== MAIN AGENT STARTING ===")
    llm = initialize_llm()
    classification = classify_query(query, llm)
    print(f"\n[Main Agent] Classified Query As: {classification}")

    content_response = browser_response = data_response = ""

    if classification == "medical":
        print("\n→ Route: RAG (with validation), fallback to Browser")
        content_response = research_with_validation(query, faiss_index)
        if not content_response.strip():
            print("\n→ No content found in RAG, switching to Browser Agent")
            browser_agent = Resource_agent()
            browser_response = browser_agent.run(query)

    elif classification == "appointment":
        print("\n→ Route: Data Agent (text-to-SQL)")
        data_agent = Data_agent()
        data_response = data_agent.run(query)

    elif classification == "non-medical":
        print("\n→ Route: General Knowledge (Browser Agent)")
        browser_agent = Resource_agent()
        browser_response = browser_agent.run(query)

    else:
        print("[DEBUG] Unable to classify query")
        return "Unable to classify query. Please rephrase."

    print("[DEBUG] All agent responses collected, formatting final output...")
    full_response = Action_agent(
        query=query,
        content_response=content_response,
        browser_response=browser_response,
        data_response=data_response,
        llm=llm
    )

    # # Extract final response only (remove labels)
    # final_resp_match = re.search(r"Final Response:\s*(.*?)(?:\n\n|$)", full_response, re.DOTALL)
    # final_output = final_resp_match.group(1).strip() if final_resp_match else full_response.strip()

    return full_response
