from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
import uvicorn
import os
import tempfile
import docx
import fitz  # PyMuPDF for PDF files
from typing import Optional, List, Dict, Any, Union
import re
import random
import json
import requests
import numpy as np
import base64

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API configuration
# Load Gemini API configuration from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY environment variable is not set!")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    true_false = "true_false"
    fill_in_the_blank = "fill_in_the_blank"
    short_answer = "short_answer"
    matching = "matching"

class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"

class TextQuizRequest(BaseModel):
    text: str
    num_questions: int = Field(default=5, gt=0, le=20)
    num_options: int = Field(default=4, gt=1, le=6)
    question_type: Union[QuestionType, List[QuestionType]] = QuestionType.multiple_choice
    difficulty: DifficultyLevel = DifficultyLevel.medium

class QuizQuestion(BaseModel):
    question: str
    options: Optional[List[str]] = None
    correct_answer: Union[str, List[str]]
    explanation: str
    question_type: QuestionType

class QuizResponse(BaseModel):
    questions: List[QuizQuestion]
    source_text_summary: str

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def encode_image_to_base64(image_path):
    """Reads an image file and encodes it to Base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_gemini_api(parts: List[Dict[str, Any]]):
    """Call the Gemini API with the given parts (text or inline_data for images)"""
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": parts
        }]
    }
    
    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.text}")
    
    return response.json()

def extract_text_from_image(base64_image: str, mime_type: str) -> str:
    """Use Gemini to extract text or describe the image."""
    parts = [
        {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64_image
            }
        },
        {
            "text": "Extract any text from the image or provide a detailed description of the image."
        }
    ]
    
    response = call_gemini_api(parts)
    extracted_text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    return extracted_text

def generate_quiz_with_gemini(content: Union[str, List[Dict[str, Any]]], num_questions: int, num_options: int, question_types: Union[str, List[str]], difficulty: str):
    """Generate a complete quiz using Gemini"""
    
    # Determine question type string
    question_type_str = ""
    if isinstance(question_types, list):
        question_type_str = ", ".join(question_types)
    else:
        question_type_str = question_types
    
    # If content is a list (likely from image), extract text first
    if isinstance(content, list) and content and "inline_data" in content[0]:
        base64_image = content[0]["inline_data"]["data"]
        mime_type = content[0]["inline_data"]["mime_type"]
        text = extract_text_from_image(base64_image, mime_type)
    else:
        text = content[0]["text"] if isinstance(content, list) and "text" in content[0] else content

    # Improved prompt that ensures questions are directly based on the text
    prompt = f"""
Generate {num_questions} quiz questions based solely on the following text. Each question must:
- Be directly answerable from the text.
- Contain exactly {num_options} answer options.
- Include a clear, correct answer (identified by its corresponding letter) and a brief explanation of why that answer is correct.
- Not reference meta-information such as page numbers or formatting details.

Text:
{text}

Format your response as follows:

Question 1:
[Question text]

a) [Option 1]
b) [Option 2]
c) [Option 3]
d) [Option 4]

Correct answer: [letter of correct option]
Explanation: [explanation]

Repeat for each question.
    """
    
    response = call_gemini_api([{"text": prompt}])
    result_text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    
    # Parse the result
    questions = []
    
    # Get a summary of the text
    summary_prompt = f"Generate a brief summary (3-5 sentences) of the following text:\n\n{text}"
    summary_response = call_gemini_api([{"text": summary_prompt}])
    summary = summary_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    
    # Extract questions using regex
    question_blocks = re.split(r'\n\s*Question \d+:', result_text)
    if len(question_blocks) > 1:  # Skip the first element which is usually intro text
        for block in question_blocks[1:]:
            if not block.strip():
                continue
                
            try:
                # Extract question text
                question_text = block.strip().split("\n\n")[0].strip()
                
                # Extract options
                options = []
                option_matches = re.findall(r'[a-d]\) (.*?)(?:\n|$)', block)
                for option in option_matches:
                    options.append(option.strip())
                
                # Extract correct answer
                correct_answer_match = re.search(r'Correct answer: ([a-d])', block)
                if correct_answer_match:
                    correct_letter = correct_answer_match.group(1)
                    correct_index = ord(correct_letter) - ord('a')
                    if 0 <= correct_index < len(options):
                        correct_answer = options[correct_index]
                    else:
                        correct_answer = "Unable to determine correct answer"
                else:
                    correct_answer = "Unable to determine correct answer"
                
                # Extract explanation
                explanation_match = re.search(r'Explanation: (.*?)(?:\n\n|$)', block, re.DOTALL)
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
                
                # Determine question type
                if len(options) == 2 and "True" in options and "False" in options:
                    q_type = QuestionType.true_false
                elif "fill in the blank" in question_text.lower() or "_____" in question_text:
                    q_type = QuestionType.fill_in_the_blank
                elif not options:
                    q_type = QuestionType.short_answer
                else:
                    q_type = QuestionType.multiple_choice
                
                question = QuizQuestion(
                    question=question_text,
                    options=options if options else None,
                    correct_answer=correct_answer,
                    explanation=explanation,
                    question_type=q_type
                )
                questions.append(question)
            except Exception as e:
                print(f"Error parsing question block: {e}")
    
    return QuizResponse(
        questions=questions[:num_questions],
        source_text_summary=summary
    )

@app.post("/generate-text-quiz", response_model=QuizResponse)
async def generate_quiz_from_text(request: TextQuizRequest):
    """Generate a quiz from provided text input"""
    try:
        # Extract parameters from the request
        text = request.text
        num_questions = request.num_questions
        num_options = request.num_options
        question_types = request.question_type
        difficulty = request.difficulty

        # Convert question_type to string or list of strings
        question_types_str = question_types if isinstance(question_types, list) else [question_types]

        # Generate the quiz
        return generate_quiz_with_gemini(
            text,
            num_questions,
            num_options,
            question_types_str,
            difficulty
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {str(e)}")

@app.post("/generate-file-quiz", response_model=QuizResponse)
async def generate_quiz_from_file(
    file: UploadFile = File(...),
    num_questions: int = Form(5),
    num_options: int = Form(4),
    question_type: str = Form(QuestionType.multiple_choice),
    difficulty: str = Form(DifficultyLevel.medium)
):
    """Generate a quiz from an uploaded file"""
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name
    
    try:
        # Extract text based on file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension == ".docx":
            text = extract_text_from_docx(temp_path)
        elif file_extension == ".pdf":
            text = extract_text_from_pdf(temp_path)
        elif file_extension == ".txt":
            text = extract_text_from_txt(temp_path)
        elif file_extension in [".png", ".jpeg", ".jpg"]:
            base64_image = encode_image_to_base64(temp_path)
            content = [{"inline_data": {"mime_type": f"image/{file_extension[1:]}", "data": base64_image}}]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate quiz
        question_types = question_type.split(",") if "," in question_type else question_type
        
        return generate_quiz_with_gemini(
            content if file_extension in [".png", ".jpeg", ".jpg"] else text,
            num_questions,
            num_options,
            question_types,
            difficulty
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
