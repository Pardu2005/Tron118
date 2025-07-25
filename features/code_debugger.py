def explain_or_debug_code(chat, code: str) -> str:
    prompt = f"Analyze the following code. Explain what it does and fix any issues:\n\n{code}"
    response = chat.send_message(prompt)
    return response.text.strip()
