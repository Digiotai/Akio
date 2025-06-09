import os
import openai
from typing import Dict, List
import re
from langchain.tools import Tool, StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
import requests
from PIL import Image
from io import BytesIO
import uuid
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
# Set your OpenAI API key (replace with your actual key)
api_key = os.getenv("OPENAI_API_KEY")


from openai import OpenAI
client = OpenAI(api_key=api_key)

# -----------------------------------------------------------------------------------------------------------------
# Image Generation Tools

def extract_image_style_from_prompt(user_prompt: str) -> Dict[str, str]:
    """Extracts image style preferences from the user prompt."""
    styles = {
        "realistic": re.search(r'realistic|photo|photograph', user_prompt, re.IGNORECASE),
        "cartoon": re.search(r'cartoon|animated|drawing', user_prompt, re.IGNORECASE),
        "painting": re.search(r'painting|oil|watercolor', user_prompt, re.IGNORECASE),
        "3d": re.search(r'3d|three dimensional|render', user_prompt, re.IGNORECASE),
        "minimalist": re.search(r'minimal|simple|clean', user_prompt, re.IGNORECASE),
        "digital art": re.search(r'digital art|digital painting', user_prompt, re.IGNORECASE)
    }

    # Default to realistic if no style detected
    detected_style = next((style for style, match in styles.items() if match), "realistic")

    return {
        "style": detected_style,
        "description": user_prompt
    }


def extract_image_count_from_prompt(user_prompt: str) -> int:
    """Extracts number of images requested from the prompt."""
    match = re.search(r'(\d+)\s+(images|pictures|illustrations)', user_prompt, re.IGNORECASE)
    if match:
        return min(int(match.group(1)), 4)  # DALL-E 2 max is 4 images per request
    return 1  # Default to 1 image


def generate_images_from_prompt(prompt: str, style: str, num_images: int = 1) -> List[str]:
    """Generates images using DALL·E 3 (new API v1+)."""
    try:
        # Enhance prompt with character limit
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=api_key
        )

        sysp = """You are a professional prompt engineer for image generation.
        Enhance the image generation prompt by:
        - Adding visual and stylistic details (e.g., colors, lighting, environment, composition)
        - Using descriptive language
        The final prompt MUST NOT exceed 950 characters. Return ONLY the final enhanced prompt as plain text."""

        enhanced_prompt = llm.invoke(f"{sysp}\n\nOriginal prompt: {prompt}\nStyle: {style}").content.strip()

        # Final safety check
        if len(enhanced_prompt) > 1000:
            enhanced_prompt = enhanced_prompt[:1000]  # truncate to safe limit

        print(f"Enhanced prompt (length {len(enhanced_prompt)}): {enhanced_prompt}")

        # Call OpenAI image generation using DALL·E 3
        response = client.images.generate(
            model="dall-e-2",  # DALL·E 3 only allows 1 image per request
            prompt=str(enhanced_prompt),
            n=num_images,
            size="1024x1024"
        )

        image_url = response.data[0].url
        image_paths = []

        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            filename = f"OpenAI/Predective_maintenence/images/dalle2_generated_{uuid.uuid4().hex[:8]}.png"
            with open(filename, 'wb') as f:
                f.write(image_response.content)
            image_paths.append(filename)

            img = Image.open(BytesIO(image_response.content))
            img.show()
            print(f"Saved at: {filename}")
        else:
            print("Image download failed.")

        return image_paths

    except Exception as e:
        print(f"Error generating image: {e}")
        raise ValueError("Failed to generate image.")

def create_thumbnail(image_path: str, size: tuple = (256, 256)) -> str:
    """Creates a thumbnail version of the generated image."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size)
            thumb_path = f"thumb_{image_path}"
            img.save(thumb_path)
            return thumb_path
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        return None


# -----------------------------------------------------------------------------------------------------------------
# Tool Wrapping for LangChain for Image Generation

def image_generator_tool(prompt: str, style: str, num_images: int) -> List[str]:
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if num_images <= 0 or num_images > 4:
        raise ValueError("Number of images must be between 1 and 4.")
    return generate_images_from_prompt(prompt, style, num_images)


def extract_image_style_tool(prompt: str) -> Dict[str, str]:
    return extract_image_style_from_prompt(prompt)


def extract_image_count_tool(prompt: str) -> int:
    return extract_image_count_from_prompt(prompt)


# -----------------------------------------------------------------------------------------------------------------
# Agent Setup

def ImageGen_agent():
    # Initialize ChatOpenAI instead of using initialize_llm
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai.api_key
    )

    tools = [
        Tool(
            name="ExtractImageCount",
            func=extract_image_count_tool,
            description="Extracts number of images requested from the user's prompt."
        ),
        Tool(
            name="ExtractImageStyle",
            func=extract_image_style_tool,
            description="Extracts image style preferences from the user's prompt."
        ),
        StructuredTool.from_function(
            func=image_generator_tool,
            name="GenerateImagesFromPrompt",
            description="Generates images using DALL-E 2. Args: prompt (str), style (str), num_images (int)",
            return_direct=True
        )
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = ImageGen_agent()

    # Example prompt
    result = agent.run("Generate 2 realistic images of a futuristic city at night")
    print(f"Generated images: {result}")