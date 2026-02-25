from dataclasses import dataclass


@dataclass
class FuzzyQuestion:
    dataset: str
    question: str
    full_prompt: str = None

    def __post_init__(self):
        if self.full_prompt is None:
            self.full_prompt = f"Answer the question: \n\n {self.question}"
