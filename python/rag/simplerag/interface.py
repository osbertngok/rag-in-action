from typing import Protocol

class RAGWrapperInterface(Protocol):

    def get_answer(self, question: str) -> str:
        ...
