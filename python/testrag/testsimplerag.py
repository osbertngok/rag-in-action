import pytest
from rag.simplerag.llamaindex import LlamaIndexWrapper
import logging

log = logging.getLogger(__name__)


class TestSimpleRAG:

    def test_llama_index_with_gemini(self) -> None:

        document_path: str = "90-文档-Data/黑悟空/设定.txt"
        question: str = "黑神话悟空中有哪些战斗工具?"

        llama_index_wrapper: LlamaIndexWrapper = LlamaIndexWrapper(document_path=document_path)
        answer: str = llama_index_wrapper.get_answer(question=question)
        log.warn(answer)
        assert answer is not None
        assert answer != ""