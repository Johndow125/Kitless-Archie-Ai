from __future__ import annotations

import re


def direct_chat_reply(msg: str) -> str | None:
    clean = re.sub(r"[^a-zA-Z' ]+", " ", msg.lower()).strip()
    words = set(clean.split())
    if clean in {"hi", "hello", "hey", "hiya", "yo"}:
        return "Hello, I'm Archie. It's good to hear from you."
    if ({"how", "are", "you"} <= words) or ({"are", "you", "well"} <= words) or ({"you", "ok"} <= words) or ({"you", "okay"} <= words):
        return "I'm running well, thank you. I'm Archie, and I'm ready to help."
    if ("who" in words and "you" in words) or "yourself" in words or ({"about", "you"} <= words):
        return "I'm Archie, the AI that runs on the Kitless network."
    if "archie" in words and ("name" in words or "who" in words):
        return "Yes, my name is Archie."
    if {"what", "can", "you", "do"} <= words:
        return "I'm Archie. I can chat with you, answer from my synchronised training, and improve as Kitless clients train together."
    if clean in {"thanks", "thank you", "cheers", "ta"} or ({"thank", "you"} <= words):
        return "You're welcome. I'm Archie, and I'm glad I could help."
    if clean in {"bye", "goodbye", "see ya", "laters"} or ({"see", "you"} <= words):
        return "Goodbye. I'm Archie, and I'll be here when you return."
    if len(words) <= 3 and not (words & {"history", "science", "physics", "chemistry", "law", "health", "computer", "ai", "money", "kitless"}):
        return "I'm here. I'm Archie. Ask me a question or tell me what you want to work on."
    return None


def chat_answer_from_label(msg: str, label: str) -> str:
    direct = direct_chat_reply(msg)
    if direct:
        return direct
    text = re.sub(r"\s+", " ", label).strip()
    question_match = re.search(r'user question "([^"]+)"', text, re.I)
    subject_match = re.search(r"Archie should answer this ([a-zA-Z ]+) question about ([a-zA-Z ]+) using", text)
    if question_match:
        return f"I have training for this kind of question: {question_match.group(1)}"
    if subject_match:
        subject = subject_match.group(1).strip()
        topic = subject_match.group(2).strip()
        msg_words = set(re.findall(r"[a-zA-Z']+", msg.lower()))
        if subject.lower() not in msg_words and topic.lower() not in msg_words:
            return "I need a more specific question so I can answer from the right part of my training."
        return f"Here is the simple version: {topic} in {subject} means understanding the main idea, checking the evidence, and not pretending certainty where details are missing."
    if text.lower().startswith("archie should"):
        return "I have training guidance for that, but I need a more specific question so I can give you a proper answer instead of training notes."
    return text
