# <PROJECT NAME>

<Short description of your project. Example: "A FastAPI application that generates quiz questions from text or files using the Google Gemini API.">

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Environment Variables](#environment-variables)
6. [Running the Application](#running-the-application)
7. [API Endpoints](#api-endpoints)
8. [Deployment](#deployment)
9. [Security and .gitignore](#security-and-gitignore)
10. [License](#license)

---

## Features

- **Quiz Generation from Text**  
  Automatically generate quiz questions based on a text snippet.
- **File Upload Support**  
  Upload DOCX, PDF, TXT, or image files (PNG/JPG/JPEG) to extract text and create quizzes.
- **Summary Generation**  
  Provide a concise summary of the source text or extracted text from images.
- **Configurable**  
  Specify number of questions, difficulty level, and question types (multiple-choice, true/false, etc.).
- **CORS Enabled**  
  Can be accessed from different domains.

---

## Prerequisites

1. **Python 3.9+** (Recommended)
2. **Google Gemini API Key**  
   - A valid API key with access to [Gemini API](https://generativelanguage.googleapis.com/).
3. **Virtual Environment** (Optional but recommended)
   - Helps keep dependencies isolated.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone <YOUR_REPO_URL>
   cd <YOUR_PROJECT_FOLDER>
