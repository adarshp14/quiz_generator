from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
import uvicorn
import os
import tempfile
import docx
import fitz
from typing import Optional, List, Dict, Any, Union
import re
import random
import requests
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY environment variable is not set!")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"  # Replace with actual URL

class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    true_false = "true_false"
    fill_in_the_blank = "fill_in_the_blank"
    short_answer = "short_answer"
    matching = "matching"
    mixed = "mixed"

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
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_pdf(file_path):
    with fitz.open(file_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_gemini_api(parts: List[Dict[str, Any]]):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": parts}]}
    response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.text}")
    return response.json()

def extract_text_from_image(base64_image: str, mime_type: str) -> str:
    parts = [
        {"inline_data": {"mime_type": mime_type, "data": base64_image}},
        {"text": "Extract any text from the image or provide a detailed description of the image."}
    ]
    response = call_gemini_api(parts)
    return response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

def get_question_prompt(text: str, question_type: str, num_questions: int, num_options: int, difficulty: str) -> str:
    base_prompt = f"Generate {num_questions} quiz questions of difficulty '{difficulty}' based solely on the following text:\n\n{text}\n\nEach question must be directly answerable from the text and include a correct answer and a brief explanation. Format your response as:\n\nQuestion [number]:\n[Question text]\n[Options or matching pairs if applicable]\nCorrect answer: [answer]\nExplanation: [explanation]\n\n"
    if question_type == "multiple_choice":
        return base_prompt + f"Questions must be multiple-choice with exactly {num_options} unique options labeled a), b), c), etc. Do not generate more than {num_options} options, and ensure no options are repeated."
    elif question_type == "true_false":
        return base_prompt + "Questions must be true/false with options 'True' and 'False'."
    elif question_type == "fill_in_the_blank":
        return base_prompt + "Questions must be fill-in-the-blank with a single blank (_____) and no options."
    elif question_type == "short_answer":
        return base_prompt + "Questions must be short-answer with no options, requiring a brief text response."
    elif question_type == "matching":
        return base_prompt + f"Questions must be matching type with {num_options} pairs of items to match (e.g., terms and definitions), labeled a), b), etc. for one list and 1), 2), etc. for the other."
    elif question_type == "mixed":
        return base_prompt + f"Generate a mix of multiple-choice (with exactly {num_options} unique options), true/false, fill-in-the-blank, short-answer, and matching questions, ensuring variety. For multiple-choice questions, do not generate more than {num_options} options and ensure no options are repeated."
    return base_prompt

def parse_gemini_response(result_text: str, num_questions: int, requested_question_type: Union[str, List[str]], num_options: int) -> List[QuizQuestion]:
    questions = []
    question_blocks = re.split(r'\n\s*Question \d+:', result_text)
    is_mixed = isinstance(requested_question_type, str) and requested_question_type == "mixed" or isinstance(requested_question_type, list) and "mixed" in requested_question_type
    requested_types = requested_question_type if isinstance(requested_question_type, list) else [requested_question_type]

    if len(question_blocks) > 1:
        for idx, block in enumerate(question_blocks[1:], start=1):
            if not block.strip():
                continue
            try:
                lines = block.strip().split("\n")
                question_text = lines[0].strip()
                options = []
                correct_answer = None
                explanation = "No explanation provided"

                # Extract options or matching pairs
                option_matches = re.findall(r'[a-z]\) (.*?)(?:\n|$)', block)
                if option_matches:
                    # Remove duplicates and trim to num_options
                    unique_options = list(dict.fromkeys(opt.strip() for opt in option_matches))  # Preserve order, remove duplicates
                    options = unique_options[:num_options]  # Take only requested number of options

                matching_pairs = re.findall(r'[a-d]\) (.*?) - \d+\) (.*?)(?:\n|$)', block)
                if matching_pairs:
                    options = [f"{left} - {right}" for left, right in matching_pairs]
                    correct_answer = [f"{left} - {right}" for left, right in matching_pairs]

                # Extract correct answer and explanation
                correct_match = re.search(r'Correct answer: (.*?)(?:\n|$)', block, re.DOTALL)
                if correct_match:
                    correct_answer = correct_match.group(1).strip()
                exp_match = re.search(r'Explanation: (.*?)(?:\n\n|$)', block, re.DOTALL)
                if exp_match:
                    explanation = exp_match.group(1).strip()

                # Determine question type
                if is_mixed and len(requested_types) > 1:
                    if matching_pairs:
                        q_type = QuestionType.matching
                    elif len(options) == 2 and "True" in options and "False" in options:
                        q_type = QuestionType.true_false
                    elif "_____" in question_text or "fill in the blank" in question_text.lower():
                        q_type = QuestionType.fill_in_the_blank
                    elif not options and not matching_pairs:
                        q_type = QuestionType.short_answer
                    else:
                        q_type = QuestionType.multiple_choice
                else:
                    q_type = QuestionType(requested_types[0])
                    if matching_pairs and q_type != QuestionType.matching:
                        q_type = QuestionType.matching
                    elif "_____" in question_text and q_type not in [QuestionType.fill_in_the_blank, QuestionType.short_answer]:
                        q_type = QuestionType.fill_in_the_blank
                    elif len(options) == 2 and "True" in options and "False" in options and q_type != QuestionType.true_false:
                        q_type = QuestionType.true_false

                # Adjust correct answer for multiple choice if needed
                if q_type == QuestionType.multiple_choice:
                    if correct_answer in "abcdefghijklmnopqrstuvwxyz"[:num_options]:
                        correct_index = ord(correct_answer.lower()) - ord('a')
                        correct_answer = options[correct_index] if 0 <= correct_index < len(options) else "Invalid answer"
                    # Ensure correct answer is in options; if not, adjust
                    if correct_answer not in options and options:
                        correct_answer = options[0]  # Fallback to first option if mismatch

                # Validate multiple-choice options count
                if q_type == QuestionType.multiple_choice:
                    if len(options) != num_options:
                        print(f"Warning: Question {idx} has {len(options)} options instead of {num_options}")
                        if len(options) < num_options:
                            # Pad with generic unique options if too few
                            existing = set(options)
                            i = len(options)
                            while len(options) < num_options:
                                new_option = f"Option {chr(97 + i)}"
                                if new_option not in existing:
                                    options.append(new_option)
                                    existing.add(new_option)
                                    i += 1

                questions.append(QuizQuestion(
                    question=question_text,
                    options=options if options and q_type not in [QuestionType.fill_in_the_blank, QuestionType.short_answer] else None,
                    correct_answer=correct_answer if not matching_pairs else correct_answer,
                    explanation=explanation,
                    question_type=q_type
                ))
            except Exception as e:
                print(f"Error parsing question block {idx}: {e}")
    return questions[:num_questions]

def generate_quiz_with_gemini(content: Union[str, List[Dict[str, Any]]], num_questions: int, num_options: int, question_types: Union[str, List[str]], difficulty: str):
    if isinstance(content, list) and "inline_data" in content[0]:
        text = extract_text_from_image(content[0]["inline_data"]["data"], content[0]["inline_data"]["mime_type"])
    else:
        text = content[0]["text"] if isinstance(content, list) and "text" in content[0] else content

    question_type_list = question_types if isinstance(question_types, list) else [question_types]
    all_questions = []

    if "mixed" in question_type_list:
        types = [t for t in QuestionType if t != QuestionType.mixed]
        questions_per_type = max(1, num_questions // len(types))
        remainder = num_questions % len(types)
        for q_type in types:
            n = questions_per_type + (1 if remainder > 0 else 0)
            remainder -= 1 if remainder > 0 else 0
            prompt = get_question_prompt(text, q_type, n, num_options, difficulty)
            response = call_gemini_api([{"text": prompt}])
            result_text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            all_questions.extend(parse_gemini_response(result_text, n, q_type, num_options))
    else:
        for q_type in question_type_list:
            prompt = get_question_prompt(text, q_type, num_questions, num_options, difficulty)
            response = call_gemini_api([{"text": prompt}])
            result_text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            all_questions.extend(parse_gemini_response(result_text, num_questions, q_type, num_options))

    summary_prompt = f"Generate a brief summary (3-5 sentences) of the following text:\n\n{text}"
    summary_response = call_gemini_api([{"text": summary_prompt}])
    summary = summary_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    random.shuffle(all_questions)
    return QuizResponse(
        questions=all_questions[:num_questions],
        source_text_summary=summary
    )

@app.post("/generate-text-quiz", response_model=QuizResponse)
async def generate_quiz_from_text(request: TextQuizRequest):
    try:
        return generate_quiz_with_gemini(
            request.text,
            request.num_questions,
            request.num_options,
            request.question_type,
            request.difficulty
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
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp_path = temp.name
    try:
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
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
