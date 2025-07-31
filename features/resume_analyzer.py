# resume_analyzer.py - Web deployment compatible version
import os
import PyPDF2
from docx import Document
from typing import Optional
import logging

# Remove tkinter imports completely for web deployment
# from tkinter import filedialog  # <-- REMOVED
# import tkinter as tk  # <-- REMOVED

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

# REMOVED: open_file_manager() function - not needed for web deployment
# File uploads are handled through the web interface

def load_resume_from_text(resume_text: str) -> str:
    """Load resume from text content (for web deployment)"""
    global LOADED_RESUME
    if not resume_text or not resume_text.strip():
        return "No resume content provided."
    
    LOADED_RESUME = resume_text.strip()
    char_count = len(LOADED_RESUME)
    logger.info(f"Resume loaded successfully! ({char_count} characters)")
    return f"Resume loaded successfully! ({char_count} characters)\n\nYou can now ask specific questions about your resume or request a full analysis."

def analyze_resume_content(chat, resume_text: str, user_query: str = None) -> str:
    if user_query and user_query.strip().lower() not in ['analyze', 'analysis', 'full analysis']:
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
    
    # Handle template requests
    if user_input.lower() in ['template', 'sample resume', 'fresher resume']:
        return """
        📄 Sample Fresher Resume Template:
        
        [Your Full Name]
        [Email Address] | [Phone Number] | [LinkedIn Profile] | [Location]
        
        OBJECTIVE:
        A motivated [field] graduate seeking to apply [key skills] in a [target role] position to contribute to organizational growth while developing professional expertise.
        
        EDUCATION:
        [Degree Name], [University Name], [Graduation Year]
        GPA: [X.X/4.0] (if above 3.5)
        Relevant Coursework: [List 3-4 relevant courses]
        
        TECHNICAL SKILLS:
        • Programming Languages: [List languages]
        • Tools & Technologies: [List tools]
        • Databases: [List database systems]
        • Other: [Additional technical skills]
        
        PROJECTS:
        [Project Name 1] | [Duration]
        • Brief description of what the project does
        • Technologies used: [List technologies]
        • Key achievements or results
        
        [Project Name 2] | [Duration]
        • Brief description of what the project does
        • Technologies used: [List technologies]
        • Key achievements or results
        
        CERTIFICATIONS:
        • [Certification Name] - [Issuing Organization] ([Year])
        • [Certification Name] - [Issuing Organization] ([Year])
        
        ACHIEVEMENTS:
        • [Academic/Professional achievement]
        • [Competition results or recognition]
        • [Leadership roles or volunteer work]
        """
    
    # Check if resume is loaded
    if LOADED_RESUME is None:
        return """
        ❌ No resume loaded yet!
        
        To analyze your resume:
        1. Upload your resume file using the file upload button above, OR
        2. Paste your resume content in the text area and submit
        3. Type 'template' for a sample resume format
        
        Supported file formats: PDF, DOCX, TXT
        """
    
    logger.info(f"🔍 Analyzing resume for query: {user_input[:50]}...")
    return analyze_resume_content(chat, LOADED_RESUME, user_input)

def clear_loaded_resume():
    """Clear the loaded resume from memory"""
    global LOADED_RESUME
    LOADED_RESUME = None
    logger.info("Resume cleared from memory.")
    return "Resume cleared from memory."

def get_resume_status():
    """Get the current status of loaded resume"""
    global LOADED_RESUME
    if LOADED_RESUME:
        char_count = len(LOADED_RESUME)
        return f"Resume loaded ({char_count} characters)"
    return "No resume loaded"