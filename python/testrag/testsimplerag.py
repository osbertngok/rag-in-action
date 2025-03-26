import pytest
from rag.simplerag.llamaindex import LlamaIndexWrapper
from rag.simplerag.langchain import LangChainWrapper
import logging

log = logging.getLogger(__name__)


class TestSimpleRAG:

    @pytest.mark.skip(reason="skip llama-index")
    def test_llama_index_with_gemini(self) -> None:

        document_path: str = "90-文档-Data/黑悟空/设定.txt"
        question: str = "黑神话悟空中有哪些战斗工具?"

        llama_index_wrapper: LlamaIndexWrapper = LlamaIndexWrapper(document_path=document_path)
        answer: str = llama_index_wrapper.get_answer(question=question)
        log.warn(answer)
        assert answer is not None
        assert answer != ""

    def test_langchain_with_gemini(self) -> None:
        url: str = "https://zh.wikipedia.org/wiki/黑神话：悟空"
        question: str = "黑神话悟空有哪些游戏场景?"

        langchain_wrapper: LangChainWrapper = LangChainWrapper(url=url)
        answer: str = langchain_wrapper.get_answer(question=question)
        log.warn(answer)
        assert answer is not None
        assert answer != ""