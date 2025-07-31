import os
import logging
import logging.handlers
import google.generativeai as genai
from dataclasses import dataclass
import sys
import re
import time
import csv
import threading
from typing import Optional
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import json
from datetime import datetime
from unittest.mock import MagicMock

# Import features
from features import doubt_solver, quiz_generator, code_debugger, resume_analyzer

# Configure logging with rotation
log_handler = logging.handlers.RotatingFileHandler('tron_server.log', maxBytes=10_000_000, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log_handler,
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Use environment variable for API key in production
GOOGLE_API_KEY = "AIzaSyBfNHzwcjsHLOP4Y1z8zBRSGgeLjI4la3s"
HISTORY_FILE = "tron_history.csv"
MAX_HISTORY_ENTRIES = 1000  # Limit history file size

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create Flask app - THIS IS CRITICAL FOR GUNICORN
app = Flask(__name__, static_folder='.')
CORS(app, origins=["*"])  # Allow all origins for deployment, restrict in production

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

class MockTkinter:
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        return MockTkinter()
    
    def __call__(self, *args, **kwargs):
        return MockTkinter()

# Create mock modules
mock_tk = MockTkinter()
mock_tk.Tk = MockTkinter
mock_tk.filedialog = MockTkinter()
mock_tk.filedialog.askopenfilename = lambda **kwargs: None
mock_tk.messagebox = MockTkinter()

# Mock all tkinter-related modules
sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.filedialog'] = mock_tk.filedialog
sys.modules['tkinter.messagebox'] = mock_tk.messagebox
sys.modules['tkinter.ttk'] = mock_tk
sys.modules['tkinter.font'] = mock_tk
sys.modules['tkinter.constants'] = mock_tk
sys.modules['_tkinter'] = mock_tk

print("Tkinter modules mocked for web deployment")


@dataclass
class Config:
    input_mode: str = "text"  # Default to text-only for web deployment
    max_retries: int = 3
    retry_delay: int = 2

    def __post_init__(self):
        try:
            if not GOOGLE_API_KEY:
                logger.error("GOOGLE_API_KEY environment variable not set")
                raise ValueError("GOOGLE_API_KEY is required")

            genai.configure(api_key=GOOGLE_API_KEY)
            # Test API key validity
            test_model = genai.GenerativeModel('gemini-1.5-flash')
            test_model.generate_content("Test")  # Simple test call
            logger.info("Google Generative AI configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Google Generative AI: {e}", exc_info=True)
            # Don't call sys.exit() here - let the application handle the error gracefully
            raise e


class AudioHandler:
    """Audio handler that works in web deployment (audio features disabled)"""

    def __init__(self, config: Config):
        self.config = config
        # Audio features disabled for web deployment
        self.audio_available = False
        logger.info("Audio features disabled for web deployment.")

    def speak(self, text: str):
        """Placeholder for speech - handled by frontend"""
        logger.info(f"Speech request received (handled by frontend): {text[:50]}...")
        return "Speech handled by frontend"

    def stop_speech(self):
        """Placeholder for stop speech - handled by frontend"""
        logger.info("Stop speech request received (handled by frontend).")
        return "Stop speech handled by frontend"

    def listen(self) -> Optional[str]:
        """Placeholder for listening - handled by frontend"""
        logger.info("Listen request received (handled by frontend).")
        return None


class ResponseHandler:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.chat = self.model.start_chat(history=[])
            logger.info("Gemini GenerativeModel and chat initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini GenerativeModel: {e}", exc_info=True)
            self.model = None
            self.chat = None

    def get_response(self, prompt: str) -> str:
        if not self.model or not self.chat:
            logger.error("Generative model not initialized.")
            return "Error: AI service is unavailable. Please check server configuration."
        try:
            logger.debug(f"Sending prompt to Gemini: {prompt[:150]}...")
            response = self.chat.send_message(prompt)
            response_text = getattr(response, 'text', '').strip()
            logger.debug(f"Received raw response from Gemini (first 150 chars): {response_text[:150]}...")
            return response_text if response_text else "No meaningful response generated."
        except Exception as e:
            logger.error(f"Error getting response from Gemini: {e}", exc_info=True)
            return f"Error: Failed to generate response: {str(e)}"


class TRONAssistant:
    def __init__(self):
        try:
            self.config = Config()
            self.audio_handler = AudioHandler(self.config)
            self.response_handler = ResponseHandler()
            self.speech_thread: Optional[threading.Thread] = None
        except Exception as e:
            logger.error(f"Failed to initialize TRONAssistant: {e}", exc_info=True)
            # Create a fallback configuration
            self.config = None
            self.audio_handler = None
            self.response_handler = ResponseHandler()  # Try to at least initialize this
            self.speech_thread = None

    def save_history(self, mode: str, query: str, response: str):
        try:
            # Check history size and truncate if necessary
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    lines = list(csv.reader(f))
                    if len(lines) > MAX_HISTORY_ENTRIES:
                        with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(['timestamp', 'mode', 'query', 'response'])
                            writer.writerows(lines[-MAX_HISTORY_ENTRIES + 1:])

            with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode, query, response])

            project_dir = "projects"
            if not os.path.exists(project_dir):
                os.makedirs(project_dir)
            output_file = os.path.join(project_dir, "TRON_project_conversation.txt")
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mode}\n")
                f.write(f"Query: {query}\n")
                f.write(f"Response: {response}\n\n")
            logger.info("History saved successfully.")
        except IOError as e:
            logger.error(f"Failed to write to history file: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving history: {e}", exc_info=True)

    def format_response_for_web(self, response: str, mode: str) -> str:
        if mode.lower() == 'quiz':
            formatted = f"""
            <div class="space-y-4">
                <div class="border-l-4 border-cyan-400 pl-4">
                    <h4 class="text-cyan-400 font-semibold mb-2">🧠 Quiz Generated Successfully</h4>
                </div>
                <div class="bg-gray-800 p-4 rounded border border-cyan-600">
                    <div class="prose prose-invert max-w-none">
                        {self._format_quiz_content(response)}
                    </div>
                </div>
                <div class="mt-4 p-3 bg-cyan-900 bg-opacity-30 rounded border border-cyan-600">
                    <strong class="text-cyan-300">✅ Quiz Ready for Practice!</strong>
                </div>
            </div>
            """
        elif mode.lower() == 'doubt':
            formatted = f"""
            <div class="space-y-4">
                <div class="border-l-4 border-purple-400 pl-4">
                    <h4 class="text-purple-400 font-semibold mb-2">🤔 Doubt Resolved</h4>
                </div>
                <div class="bg-gray-800 p-4 rounded border border-purple-600">
                    <div class="prose prose-invert max-w-none">
                        {self._format_explanation_content(response)}
                    </div>
                </div>
                <div class="mt-4 p-3 bg-purple-900 bg-opacity-30 rounded border border-purple-600">
                    <strong class="text-purple-300">💡 Hope this clarifies your doubt!</strong>
                </div>
            </div>
            """
        elif mode.lower() == 'code':
            formatted = f"""
            <div class="space-y-4">
                <div class="border-l-4 border-red-400 pl-4">
                    <h4 class="text-red-400 font-semibold mb-2">🐛 Code Analysis Complete</h4>
                </div>
                <div class="bg-gray-800 p-4 rounded border border-red-600">
                    <div class="prose prose-invert max-w-none">
                        {self._format_code_content(response)}
                    </div>
                </div>
                <div class="mt-4 p-3 bg-red-900 bg-opacity-30 rounded border border-red-600">
                    <strong class="text-red-300">🔧 Code debugging recommendations provided</strong>
                </div>
            </div>
            """
        elif mode.lower() == 'resume':
            formatted = f"""
            <div class="space-y-4">
                <div class="border-l-4 border-yellow-400 pl-4">
                    <h4 class="text-yellow-400 font-semibold mb-2">📄 Resume Analysis Complete</h4>
                </div>
                <div class="bg-gray-800 p-4 rounded border border-yellow-600">
                    <div class="prose prose-invert max-w-none">
                        {self._format_resume_content(response)}
                    </div>
                </div>
                <div class="mt-4 p-3 bg-yellow-900 bg-opacity-30 rounded border border-yellow-600">
                    <strong class="text-yellow-300">📊 Professional analysis and recommendations provided</strong>
                </div>
            </div>
            """
        else:
            safe_response = response.replace('<', '&lt;').replace('>', '&gt;').replace(chr(10), '<br>')
            formatted = f"""
            <div class="space-y-4">
                <div class="bg-gray-800 p-4 rounded border border-gray-600">
                    <div class="prose prose-invert max-w-none">
                        <p class="text-gray-200">{safe_response}</p>
                    </div>
                </div>
            </div>
            """
        return formatted

    def _format_quiz_content(self, content: str) -> str:
        lines = content.split('\n')
        formatted_lines = []
        question_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(marker in line.lower() for marker in ['question', 'q:', 'q.']) and not line.startswith(
                    ('a)', 'b)', 'c)', 'd)')):
                question_count += 1
                if question_count > 1:
                    formatted_lines.append('</div>')  # Close previous question block
                    formatted_lines.append(
                        '<div class="question-block mt-6 p-3 bg-gray-700 rounded border border-cyan-500">')
                else:
                    formatted_lines.append(
                        '<div class="question-block mt-4 p-3 bg-gray-700 rounded border border-cyan-500">')
                formatted_lines.append(f'<strong class="text-cyan-300">Question {question_count}:</strong>')
                formatted_lines.append(f'<p class="mt-2 text-gray-200">{line}</p>')
            elif line.startswith(('a)', 'b)', 'c)', 'd)', 'A)', 'B)', 'C)', 'D)')):
                formatted_lines.append(f'<div class="option ml-4 mt-1 text-gray-300">• {line}</div>')
            elif 'answer' in line.lower() and not line.startswith(('a)', 'b)', 'c)', 'd)')):
                formatted_lines.append(
                    '<div class="answer mt-2 p-2 bg-green-800 bg-opacity-50 rounded border border-green-500">')
                formatted_lines.append(f'<strong class="text-green-300">{line}</strong>')
                formatted_lines.append('</div>')
            else:
                formatted_lines.append(f'<p class="text-gray-200">{line}</p>')
        if question_count > 0:
            formatted_lines.append('</div>')  # Close last question block
        return ''.join(formatted_lines)

    def _format_explanation_content(self, content: str) -> str:
        try:
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong class="text-purple-300">\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em class="text-purple-200">\1</em>', content)
            content = re.sub(r'`(.*?)`', r'<code class="bg-gray-700 px-2 py-1 rounded text-purple-200">\1</code>',
                             content)
            content = re.sub(r'^- (.+)$', r'<li><p>\1</p></li>', content, flags=re.MULTILINE)
            if '<li>' in content:
                content = f'<ul class="list-disc list-inside mt-3 text-gray-200">{content}</ul>'
            content_parts = content.split('\n\n')
            paragraphs = []
            for p in content_parts:
                if p.strip():
                    safe_p = p.replace("\n", "<br>")
                    paragraphs.append(f'<p class="mt-3 text-gray-200">{safe_p}</p>')
            return ''.join(paragraphs)
        except Exception as e:
            logger.error(f"Error formatting explanation content: {e}", exc_info=True)
            return content.replace('\n', '<br>')

    def _format_code_content(self, content: str) -> str:
        lines = content.split('\n')
        formatted_lines = []
        in_code_block = False

        for line in lines:
            line_strip = line.strip()
            if line_strip.startswith('```'):
                if in_code_block:
                    formatted_lines.append('</code></pre>')
                    in_code_block = False
                else:
                    lang = line_strip[3:].strip()
                    lang_class = f'language-{lang}' if lang else ''
                    formatted_lines.append(
                        f'<pre class="bg-gray-900 p-3 rounded border border-red-500 mt-3 overflow-auto"><code class="text-red-200 {lang_class}">')
                    in_code_block = True
                continue
            if in_code_block:
                formatted_lines.append(f'{line}\n')
            else:
                if any(keyword in line.lower() for keyword in ['error', 'bug', 'issue', 'problem', 'fix']):
                    formatted_lines.append(
                        '<div class="error-solution-highlight p-2 bg-red-800 bg-opacity-50 rounded border border-red-500 mt-2">')
                    formatted_lines.append(f'<strong class="text-red-300">🚨 {line}</strong>')
                    formatted_lines.append('</div>')
                elif any(keyword in line.lower() for keyword in ['solution', 'correct', 'optim', 'recommend']):
                    formatted_lines.append(
                        '<div class="solution-highlight p-2 bg-green-800 bg-opacity-50 rounded border border-green-500 mt-2">')
                    formatted_lines.append(f'<strong class="text-green-300">✅ {line}</strong>')
                    formatted_lines.append('</div>')
                else:
                    formatted_lines.append(f'<p class="text-gray-200 mt-2">{line}</p>')
        if in_code_block:
            formatted_lines.append('</code></pre>')
        return ''.join(formatted_lines)

    def _format_resume_content(self, content: str) -> str:
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(keyword in line.lower() for keyword in ['score', 'rating', '/10', '%']):
                formatted_lines.append(
                    '<div class="score-section p-3 bg-yellow-800 bg-opacity-50 rounded border border-yellow-500 mt-3">')
                formatted_lines.append(f'<strong class="text-yellow-300">📊 {line}</strong>')
                formatted_lines.append('</div>')
            elif any(keyword in line.lower() for keyword in ['strength', 'good', 'excellent', 'strong']):
                formatted_lines.append(
                    '<div class="strength-item p-2 bg-green-800 bg-opacity-30 rounded border border-green-600 mt-2">')
                formatted_lines.append(f'<span class="text-green-300">✅ {line}</span>')
                formatted_lines.append('</div>')
            elif any(keyword in line.lower() for keyword in ['improve', 'weak', 'lacking', 'missing']):
                formatted_lines.append(
                    '<div class="improvement-item p-2 bg-orange-800 bg-opacity-30 rounded border border-orange-600 mt-2">')
                formatted_lines.append(f'<span class="text-orange-300">📈 {line}</span>')
                formatted_lines.append('</div>')
            else:
                formatted_lines.append(f'<p class="text-gray-200 mt-2">{line}</p>')
        return ''.join(formatted_lines)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_history_file():
    """Initialize history file with header if it doesn't exist"""
    if not os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'mode', 'query', 'response'])
            logger.info(f"History file '{HISTORY_FILE}' created with header.")
        except IOError as e:
            logger.error(f"Failed to create history file '{HISTORY_FILE}': {e}", exc_info=True)


# Initialize required files and directories
def setup_app():
    """Setup function called at module level"""
    try:
        initialize_history_file()
        if not os.path.exists("projects"):
            os.makedirs("projects", exist_ok=True)
            logger.info("Created 'projects' directory.")
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            logger.info("Created uploads directory.")
    except Exception as e:
        logger.error(f"Error during app setup: {e}", exc_info=True)


# Setup app immediately when module is imported
setup_app()

# Initialize the TRON instance with error handling
try:
    tron_instance = TRONAssistant()
    logger.info("TRON Assistant initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize TRON Assistant: {e}", exc_info=True)
    # Create a minimal fallback instance
    tron_instance = None


@app.route('/')
def serve_index():
    html_files = ['index.html', 'tron_assistant_colored.html', 'Index.html']
    for html_file in html_files:
        if os.path.exists(html_file):
            logger.info(f"Serving {html_file}")
            return send_from_directory('.', html_file)
    logger.error("No HTML file found in project directory to serve.")
    return abort(404, description="HTML file not found in project directory")


@app.route('/favicon.ico')
def serve_favicon():
    if os.path.exists('favicon.ico'):
        return send_from_directory('.', 'favicon.ico', mimetype='image/x-icon')
    logger.debug("favicon.ico not found, returning 204 No Content.")
    return '', 204


# Health check endpoint for Render - CRITICAL FOR PORT DETECTION
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'port': os.environ.get('PORT', 'unknown'),
        'message': 'TRON Assistant is running',
        'tron_status': 'initialized' if tron_instance else 'failed'
    })


# Add a simple test endpoint that responds quickly
@app.route('/ping')
def ping():
    return jsonify({'status': 'pong', 'timestamp': datetime.now().isoformat()})


@app.route('/api/process', methods=['POST'])
def process_query():
    try:
        if not tron_instance:
            logger.error("TRON instance not available.")
            return jsonify({'error': 'Service temporarily unavailable. Please try again later.', 'status': 'error'}), 503

        data = request.get_json()
        if not data:
            logger.warning("Received /api/process request with no JSON data.")
            return jsonify({'error': 'No JSON data received', 'status': 'error'}), 400

        mode = data.get('mode')
        query = data.get('query', '')
        input_mode = data.get('inputMode', 'text')

        logger.info(f"Processing request - Mode: {mode}, Query: {query[:100]}..., Input Mode: {input_mode}")

        mode_map = {
            'quiz': ("Quiz Generator", quiz_generator.generate_quiz),
            'doubt': ("Doubt Solver", doubt_solver.solve_doubt),
            'code': ("Code Debugger", lambda chat, code_query: code_debugger.explain_or_debug_code(chat, code_query)),
            'resume': ("Resume Analyzer", resume_analyzer.analyze_resume)
        }

        if mode not in mode_map:
            logger.error(f"Invalid mode specified: {mode}")
            return jsonify({'error': f'Invalid mode: {mode}', 'status': 'error'}), 400

        if not query and mode != 'resume':
            logger.warning("Query is empty for non-resume mode.")
            return jsonify({'error': 'Query is required for selected module.', 'status': 'error'}), 400

        if mode == 'resume' and not query:
            if not getattr(resume_analyzer, 'LOADED_RESUME', None):
                logger.warning("Resume mode selected, but no query provided and no resume file loaded.")
                return jsonify(
                    {'error': 'Please upload a resume file or paste resume content.', 'status': 'error'}), 400
            query = resume_analyzer.LOADED_RESUME

        mode_name, handler = mode_map[mode]
        logger.debug(f"Calling {mode_name} handler with query length {len(query)}.")

        raw_response = ""
        try:
            raw_response = handler(tron_instance.response_handler.chat, query)
            if not raw_response or not raw_response.strip():
                logger.warning("AI raw_response is empty or only whitespace.")
                raw_response = "No meaningful response generated by the AI. Please try a different query."
        except Exception as ai_e:
            logger.error(f"Error during AI handler call for mode {mode}: {ai_e}", exc_info=True)
            raw_response = f"AI processing error: {str(ai_e)}. Please try again or contact support."

        formatted_response = tron_instance.format_response_for_web(raw_response, mode)
        tron_instance.save_history(mode_name, query, raw_response)

        response_data = {
            'response': formatted_response,
            'raw_response': raw_response,
            'mode': mode_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        logger.info(f"Successfully processed {mode_name} request. Final response length: {len(raw_response)}.")
        return jsonify(response_data)

    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from request.")
        return jsonify({'error': 'Invalid JSON format', 'status': 'error'}), 400
    except Exception as e:
        logger.error(f"Unhandled error in /api/process: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}', 'status': 'error'}), 500


@app.route('/api/speech/listen', methods=['POST'])
def speech_listen():
    try:
        logger.info("Speech recognition request received - handled by frontend.")
        return jsonify({
            'error': 'Speech recognition handled by frontend browser APIs',
            'status': 'info',
            'message': 'Use browser Web Speech API for audio input'
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/speech/listen: {str(e)}", exc_info=True)
        return jsonify({'error': f'Speech recognition error: {str(e)}', 'status': 'error'}), 500


@app.route('/api/speech/speak', methods=['POST'])
def speech_speak():
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON data received for speech speak request.")
            return jsonify({'error': 'No JSON data received', 'status': 'error'}), 400
            
        text = data.get('text', '')

        if not text:
            logger.warning("Received empty text for speech generation.")
            return jsonify({'error': 'No text provided for speech.', 'status': 'error'}), 400

        logger.info(f"Received text to speak (handled by frontend): {text[:100]}...")

        return jsonify({
            'status': 'success',
            'message': 'Speech handled by frontend browser APIs',
            'text': text
        })

    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from speech speak request.", exc_info=True)
        return jsonify({'error': 'Invalid JSON format', 'status': 'error'}), 400
    except Exception as e:
        logger.error(f"Error in /api/speech/speak: {str(e)}", exc_info=True)
        return jsonify({'error': f'Text-to-speech error: {str(e)}', 'status': 'error'}), 500


@app.route('/api/speech/stop', methods=['POST'])
def speech_stop():
    try:
        logger.info("Stop speech request received (handled by frontend).")
        return jsonify({
            'status': 'success',
            'message': 'Speech stop handled by frontend browser APIs'
        })
    except Exception as e:
        logger.error(f"Error in /api/speech/stop: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error stopping speech: {str(e)}', 'status': 'error'}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        history_data = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 4:
                        history_data.append({
                            'timestamp': row[0],
                            'mode': row[1],
                            'query': row[2][:100] + '...' if len(row[2]) > 100 else row[2],
                            'response': row[3][:200] + '...' if len(row[3]) > 200 else row[3]
                        })
        history_data.reverse()
        history_data = history_data[:20]
        logger.info(f"History loaded: {len(history_data)} entries.")
        return jsonify({'history': history_data, 'status': 'success'})
    except IOError as e:
        logger.error(f"Failed to read history file: {e}", exc_info=True)
        return jsonify({'error': 'Failed to load history', 'status': 'error'}), 500
    except Exception as e:
        logger.error(f"Unhandled error in /api/history: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}', 'status': 'error'}), 500


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            initialize_history_file()
            logger.info("History file cleared and re-initialized.")
        else:
            logger.info("History file did not exist, no action needed for clear.")
        return jsonify({'status': 'success', 'message': 'History cleared.'})
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error clearing history: {str(e)}', 'status': 'error'}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.warning("No file part in upload request.")
            return jsonify({'error': 'No file part', 'status': 'error'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file in upload request.")
            return jsonify({'error': 'No selected file', 'status': 'error'}), 400

        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            logger.info(f"File uploaded successfully: {filename}")

            if filename.lower().endswith(('.pdf', '.doc', '.docx')):
                try:
                    extracted_text = resume_analyzer.extract_text_from_file(file_path)
                    resume_analyzer.LOADED_RESUME = extracted_text
                    logger.info(f"Text extracted from resume file: {filename}")
                    return jsonify({
                        'status': 'success',
                        'message': f'File {filename} uploaded and text extracted successfully.',
                        'filename': filename
                    })
                except Exception as e:
                    logger.error(f"Failed to extract text from {filename}: {e}", exc_info=True)
                    return jsonify({
                        'status': 'error',
                        'message': f'File uploaded, but failed to extract text: {str(e)}',
                        'filename': filename
                    }), 400
            else:
                logger.info(f"Uploaded file {filename} is not a supported resume format. Text extraction skipped.")
                return jsonify({
                    'status': 'success',
                    'message': f'File {filename} uploaded successfully, but not processed as a resume.',
                    'filename': filename
                })
        else:
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            return jsonify({'error': 'Invalid file type. Allowed: pdf, doc, docx.', 'status': 'error'}), 400

    except Exception as e:
        logger.error(f"Error in /api/upload: {str(e)}", exc_info=True)
        return jsonify({'error': f'File upload error: {str(e)}', 'status': 'error'}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    ai_status = 'available' if tron_instance and tron_instance.response_handler.model else 'unavailable'
    logger.debug(f"Current status - AI: {ai_status}")
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'speech_recognition': 'frontend_handled',
            'text_to_speech': 'frontend_handled',
            'ai_processing': ai_status
        }
    })


@app.errorhandler(400)
def bad_request(error):
    logger.error(f"400 Bad Request: {error.description if hasattr(error, 'description') else str(error)}",
                 exc_info=True)
    return jsonify({
        'error': error.description if hasattr(error, 'description') else 'Bad request',
        'status': 'error'
    }), 400


@app.errorhandler(404)
def not_found(error):
    logger.error(f"404 Not Found: {error.description if hasattr(error, 'description') else str(error)}", exc_info=True)
    return jsonify({
        'error': error.description if hasattr(error, 'description') else 'Resource not found',
        'status': 'error'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Internal Server Error: {error.description if hasattr(error, 'description') else str(error)}",
                    exc_info=True)
    return jsonify({
        'error': 'Internal server error. Please try again later.',
        'status': 'error'
    }), 500


# Application factory pattern for better deployment
def create_app():
    """Application factory pattern for better deployment"""
    return app


# WSGI callable - CRITICAL for Gunicorn
application = app

# Ensure app is available at module level for Render
app = app

if __name__ == "__main__":
    try:
        # Enhanced port configuration for Render deployment
        port = int(os.environ.get('PORT', 10000))
        
        logger.info("Starting TRON Assistant Flask server...")
        logger.info(f"Server will be accessible on port: {port}")
        
        # Always run the app directly for better port binding
        app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

    except KeyboardInterrupt:
        logger.info("Server interrupted by user (KeyboardInterrupt).")
        print("\nServer interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal server startup error: {e}", exc_info=True)
        print(f"Server error occurred: {e}")
        # Only exit when running directly, not when imported by Gunicorn
        if __name__ == "__main__":
            sys.exit(1)