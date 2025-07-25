def generate_quiz(chat, topic: str) -> str:
    prompt = f"Generate 5 multiple-choice questions (with answers) on the topic: {topic}."
    response = chat.send_message(prompt)
    return response.text.strip()
