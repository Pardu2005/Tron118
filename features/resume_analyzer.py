# resume_analyzer.py
import os
import PyPDF2
from docx import Document  # Corrected import to use docx
from typing import Optional
import logging
from tkinter import filedialog
import tkinter as tk

logger = logging.getLogger(__name__)

LOADED_RESUME = None

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text() or ""
                text += extracted + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error reading TXT file: {e}")
            return ""
    except Exception as e:
        logger.error(f"Error reading TXT file: {e}")
        return ""

def extract_text_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return ""
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        return f"Unsupported file format: {file_extension}"

def open_file_manager() -> Optional[str]:
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_types = [
            ("All Supported", "*.pdf;*.docx;*.txt"),
            ("PDF files", "*.pdf"),
            ("Word documents", "*.docx"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            title="Select Resume File",
            filetypes=file_types,
            initialdir=os.getcwd()
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        logger.error(f"Error opening file manager: {e}")
        return None

def load_resume() -> str:
    global LOADED_RESUME
    print("\n" + "=" * 50)
    print("RESUME ANALYZER - LOAD RESUME")
    print("=" * 50)
    print("Opening file manager to select your resume...")
    print("Supported formats: PDF, DOCX, TXT")
    print("=" * 50)
    file_path = open_file_manager()
    if not file_path:
        return "No file selected. Resume loading cancelled."
    print(f"Loading file: {os.path.basename(file_path)}")
    resume_text = extract_text_from_file(file_path)
    if not resume_text:
        return "Could not extract text from the file. Please check the file format and try again."
    LOADED_RESUME = resume_text
    char_count = len(resume_text)
    print(f"✅ Resume loaded successfully! ({char_count} characters)")
    return f"Resume loaded successfully! ({char_count} characters)\n\nYou can now:\n- Ask specific questions about your resume\n- Type 'analyze' for full detailed analysis\n- Type 'load' to upload a different resume"

def analyze_resume_content(chat, resume_text: str, user_query: str = None) -> str:
    if user_query and user_query.strip().lower() != 'analyze':
        prompt = f"""
        Based on the following resume content, please answer the user's specific question:
        RESUME CONTENT:
        {resume_text}
        USER QUESTION: {user_query}
        Please provide a detailed and specific answer based on the resume content.
        """
    else:
        prompt = f"""
        Analyze the following resume comprehensively and provide detailed feedback:
        RESUME CONTENT:
        {resume_text}
        Please provide analysis in the following structure:
        1. OVERALL ASSESSMENT (Score out of 10):
        - Brief summary of strengths and weaknesses
        2. STRUCTURE & FORMATTING:
        - Layout and organization
        - Readability and visual appeal
        - Section organization
        3. CONTENT ANALYSIS:
        - Professional summary/objective
        - Work experience descriptions
        - Skills section
        - Education section
        - Achievements and quantifiable results
        4. ATS COMPATIBILITY:
        - Keyword optimization
        - Format compatibility
        - Section headers
        5. SPECIFIC IMPROVEMENTS:
        - 3-5 actionable recommendations
        - Specific examples of what to change
        6. MISSING ELEMENTS:
        - What important sections or information are missing
        7. INDUSTRY ALIGNMENT:
        - How well does this resume match modern standards
        - Industry-specific suggestions
        Please be constructive and specific in your feedback.
        """
    try:
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        return "Sorry, there was an error analyzing your resume. Please try again."

def analyze_resume(chat, user_input: str) -> str:
    global LOADED_RESUME
    if user_input.lower() in ['load', 'upload', 'new file', 'new resume', 'load resume']:
        return load_resume()
    if user_input.lower() in ['template', 'sample resume', 'fresher resume']:
        return """
        Sample Fresher Resume Template:
        [Name]
        [Contact Info]
        Objective: A motivated [field] graduate seeking to apply [skills] in a [role] position.
        Education: [Degree], [University], [Year], [GPA]
        Skills: [List relevant skills]
        Projects: [Project name, description, technologies used]
        Certifications: [Certification name, issuer, year]
        """
    if LOADED_RESUME is None:
        return "❌ No resume loaded yet!\n\nPlease type 'load' to select and upload your resume file first, or type 'template' for a sample resume."
    print(f"🔍 Analyzing resume for: {user_input}")
    return analyze_resume_content(chat, LOADED_RESUME, user_input)

def clear_loaded_resume():
    global LOADED_RESUME
    LOADED_RESUME = None
    return "Resume cleared from memory."