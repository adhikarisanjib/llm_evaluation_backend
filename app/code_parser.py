import re


def extract_code(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
