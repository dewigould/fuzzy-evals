from pathlib import Path

RUBRIC = (Path(__file__).parent / "../rubrics/weird_questions_rubric.md").read_text()


def build_grading_prompt(question:str, answer: str) -> str:
    """Build the prompt for grading."""
    return f"""You are grading an answer for quality of thought, research, effort, and exposition.

## Question

{question}

## Answer to Grade

{answer}

## Grading Rubric

{RUBRIC}"""
