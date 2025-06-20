import os
import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Tuple
import re
from langchain.tools import StructuredTool
from typing import Dict

from pandas import DataFrame
# For PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image, Table, TableStyle, Frame, KeepInFrame
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import requests
from typing import List, Dict, Optional, Tuple
import re
import json
from PIL import Image as PILImage
from sklearn.utils import resample


def initialize_llm():
    print("[DEBUG] Initializing OpenAI LLM...")
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=500
    )


# -----------------------------------------------------------------------------------------------------------------
# For DataScout with Excel Generation
# Data Extraction Tools
def parse_prompt_with_semantic_ai(prompt: str) -> Tuple[Optional[int], List[str]]:
    llm = initialize_llm()

    system_instruction = (
        """You are a prompt parser for synthetic data generation. 
        Given a user input prompt, extract:
        1. The number of rows requested (as an integer).
        2. The list of column/field names (as a list of lowercase snake_case strings).

        Respond with a JSON object strictly in the form:
        {"num_rows": <int or null>, "columns": ["col1", "col2", ...]}.
        Do not include any commentary.
        """
    )

    response = llm.invoke(f"{system_instruction}\n\nPrompt: {prompt}")
    content = response.content if hasattr(response, 'content') else str(response)

    try:
        parsed = eval(content) if isinstance(content, str) else content
        num_rows = parsed.get("num_rows")
        columns = parsed.get("columns", [])
        return num_rows, columns
    except Exception:
        return None, []


def extrapolate_from_seed(seed_df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    factor = (target_rows + len(seed_df) - 1) // len(seed_df)
    repeated_df = pd.concat([seed_df] * factor, ignore_index=True)
    extrapolated_df = resample(repeated_df, n_samples=target_rows, random_state=42)
    extrapolated_df.reset_index(drop=True, inplace=True)
    return extrapolated_df


def generate_data_from_text(text_sample: str, column_names: List[str], num_rows: int = 10, seed_limit: int = 150) -> \
        Tuple[str, pd.DataFrame]:
    llm = initialize_llm()

    sysp = """You are a data generator. Follow these rules:
    1. Generate only the requested data format
    2. No additional commentary
    3. No markdown or code fences
    4. No null values generation
    5. Strictly follow the output format"""

    column_names_str = ", ".join(column_names)
    generated_rows = []

    prompt = (
        f"{sysp}\n\n"
        f"Description: '{text_sample}'\n"
        f"Generate {min(seed_limit, num_rows)} rows of synthetic data with columns: {column_names_str}.\n"
        f"Tilde-separated only. No column headers or extra text."
    )

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)

    for line in content.strip().split('\n'):
        parts = [cell.strip() for cell in line.split('~')]
        if len(parts) == len(column_names):
            generated_rows.append(parts)

    df_seed = pd.DataFrame(generated_rows, columns=column_names)

    if num_rows <= seed_limit:
        df = df_seed
    else:
        df = extrapolate_from_seed(df_seed, num_rows)

    file_path = "data_output.xlsx"
    df.to_excel(file_path, index=False)
    return file_path, df


# ----------------------------------------------------------------------------------------------------------------
# For DataScout with PDF Generation
# Data Extraction Tools for PDF
import re
from typing import Optional


# --- Prompt Parsing for PDF (LLM-based) ---
def extract_pdf_prompt_semantics(user_prompt: str) -> Tuple[Optional[int], List[str]]:
    llm = initialize_llm()

    system_instruction = (
        """You are a document request parser. Given a user prompt, extract:
        1. The number of pages requested (as an integer).
        2. The list of section names (as a list of title-case strings).

        Respond with JSON only in this format:
        {"num_pages": <int or null>, "sections": ["Section 1", "Section 2", ...]}.
        No commentary or markdown.
        """
    )

    response = llm.invoke(f"{system_instruction}\n\nPrompt: {user_prompt}")
    content = response.content if hasattr(response, 'content') else str(response)

    try:
        parsed = json.loads(content)
        return parsed.get("num_pages"), parsed.get("sections", [])
    except Exception:
        return None, []

def parse_llm_json_response(response_text: str) -> Optional[dict]:
    if not response_text.strip():
        return None
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE)
        return json.loads(cleaned)
    except Exception as e:
        print(f"[LLM Parse Error] {e} | Raw: {response_text}")
        return None


# ----------------------------------------
# PDF Generator Core

def generate_structured_content(text_sample: str, sections: List[str], num_pages: int = 15) -> Dict[str, Any]:
    assert isinstance(text_sample, str), "text_sample must be a string"
    assert isinstance(sections, list) and all(
        isinstance(s, str) for s in sections), "sections must be a list of strings"

    llm = initialize_llm()

    try:
        joined_sections = ', '.join(sections)
        analysis_prompt = (
            f"Analyze this document request and return JSON:\n"
            f"{{\n"
            f"    \"title\": \"Document title\",\n"
            f"    \"style\": \"professional/academic\",\n"
            f"    \"sections\": [\n"
            f"        {{\"name\": \"Section name\", \"content_type\": \"text/mixed\", \"needs_visuals\": false}}\n"
            f"    ]\n"
            f"}}\n"
            f"Request: Create a {num_pages}-page document about '{text_sample}' with sections: [{joined_sections}]"
        )

        analysis = llm.invoke(analysis_prompt)
        structure = parse_llm_json_response(analysis.content)

        if not structure:
            print("Analysis failed, using defaults.")
            structure = {
                "title": "Generated Document",
                "style": "professional",
                "sections": [{"name": s, "content_type": "text"} for s in sections]
            }
    except Exception as e:
        print(f"Analysis exception, using defaults: {e}")
        structure = {
            "title": "Generated Document",
            "style": "professional",
            "sections": [{"name": s, "content_type": "text"} for s in sections]
        }

    output = {
        "title": structure["title"],
        "sections": []
    }

    for section in structure["sections"]:
        content_prompt = (
            f"Write professional content for the section: '{section['name']}'\n"
            f"Topic: {text_sample}\n"
            f"Format: Use main heading, and 2-3 subheadings, each with 1-2 detailed paragraphs\n"
            f"Tone: Professional and well-structured\n"
            f"Return as JSON: {{\"heading\": \"...\", \"subsections\": [{{\"subheading\": \"...\", \"content\": \"...\"}}]}}"
        )

        response = llm.invoke(content_prompt)
        structured_section = parse_llm_json_response(response.content)

        if structured_section:
            output["sections"].append(structured_section)
        else:
            print(f"Failed to parse section '{section['name']}'")
            print(f"Raw response: {response.content}")

    return output


# ----------------------------------------------------------------------------------------------------------------
# Tool Wrapping for LangChain For PDF Generation
def pdf_generator_tool(prompt: str, sections: List[str] = None, number_of_pages: int = None) -> Dict[str, Any]:
    if not sections or not isinstance(sections, list):
        sections = ["Introduction", "Content", "Conclusion"]
    if not number_of_pages or number_of_pages <= 0:
        number_of_pages = 1
    return generate_structured_content(prompt, sections, number_of_pages)

def extract_sections_tool(prompt: str) -> List[str]:
    _, sections = extract_pdf_prompt_semantics(prompt)
    return sections

def extract_num_pages_tool(prompt: str) -> int:
    num_pages, _ = extract_pdf_prompt_semantics(prompt)
    return num_pages if num_pages else 1

# -----------------------------------------------------------------------------------------------------------------
def excel_generator_tool(prompt: str) -> tuple[str, DataFrame]:
    num_rows, columns = parse_prompt_with_semantic_ai(prompt)
    if not columns:
        raise ValueError("Could not extract column names from prompt.")
    if not num_rows or num_rows <= 0:
        raise ValueError("Could not extract valid number of rows from prompt.")

    return generate_data_from_text(prompt, columns, num_rows)



# -----------------------------------------------------------------------------------------------------------------
# Agent Setup
def DataScout_agent():
    llm = initialize_llm()
    tools = [
        Tool(
            func=excel_generator_tool,
            name="GenerateExcelFromPrompt",
            description="Generate an Excel file from a free-text prompt. The prompt should include both the number of records and fields.",
            return_direct=True
        )

    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent


# -----------------------------------------------------------------------------------------------------------------
# Agent Setup for PDF Generation

def DataScout_agent_with_pdf():
    llm = initialize_llm()
    tools = [
        Tool(
            name="ExtractPageCount",
            func=extract_num_pages_tool,
            description="Extracts number of pages from the user's prompt."
        ),
        Tool(
            name="ExtractSectionNames",
            func=extract_sections_tool,
            description="Extracts section names from the user's prompt."
        ),
        StructuredTool.from_function(
            func=pdf_generator_tool,
            name="GeneratePDFFromPrompt",
            description="Generates structured PDF content from the prompt, section list, and number of pages.",
            return_direct=True
        )
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

# pip install langchain-openai pandas openpyxl reportlab


# Testing the pipeline
# if __name__ == "__main__":
#     test_prompt = "Generate a table with 25 rows of realistic synthetic data with field names: product_id, product_name, category, current_stock_quantity, supplier_name, supplier_contact,units_sold,Date"
#     try:
#         agent = DataScout_agent()
#         response = agent.invoke(test_prompt)
#         print(f"Response: {response}")
#     except Exception as e:
#         print(f"❌ Error: {e}")

# if __name__ == "__main__":
#     test_prompt = "Create a pdf with 3 pages about Artificial Intelligence with sections: Introduction, Methodology, Conclusion in 500 words per each page"
#     try:
#         agent = DataScout_agent_with_pdf()
#         response = agent.invoke(test_prompt)
#         print(f"Response: {response}")
#     except Exception as e:
#         print(f"❌ Error: {e}")
