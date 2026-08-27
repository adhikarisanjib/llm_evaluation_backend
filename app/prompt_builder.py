from app.models import Task


def build_prompt(task: Task) -> str:
    starter = task.starter_code or "(none)"
    language = task.programming_language

    if task.task_type == "debugging":
        return f"""You are a coding assistant. Fix the {language} code below while preserving existing correct behaviour.\n\nQuestion:\n{task.question}\n\nBuggy starter code:\n{starter}\n\nReturn only the complete corrected {language} code. Do not use Markdown fences or explanations."""

    return f"""You are a coding assistant. Implement the {language} task below.\n\nQuestion:\n{task.question}\n\nStarter code:\n{starter}\n\nReturn only the complete {language} code. Do not use Markdown fences or explanations."""
