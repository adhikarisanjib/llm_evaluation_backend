import re


def extract_code(text: str) -> str:
    text = text.strip()
    match = re.search(
        r"```(?:python|js|javascript)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip(" \t\n\r`")
    return text.strip(" \t\n\r`")
