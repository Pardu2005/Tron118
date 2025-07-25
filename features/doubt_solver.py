def solve_doubt(chat, question: str) -> str:
    prompt = f"Explain the following in simple terms for a student:\n{question}"
    response = chat.send_message(prompt)
    return response.text.strip()
