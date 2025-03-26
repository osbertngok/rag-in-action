from typing import Optional, List
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.llms.google_genai import GoogleGenAI
from rag.simplerag.interface import RAGWrapperInterface


class LlamaIndexWrapper(RAGWrapperInterface):

    def __init__(self, document_path: str) -> None:
        self.document_path = document_path

    def get_answer(self, question: str) -> str:
        ROOT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")

        # 加载环境变量
        from dotenv import load_dotenv
        # 加载 .env 文件中的环境变量
        load_dotenv()

        # 创建 Gemini LLM
        llm: GoogleGenAI  = GoogleGenAI(
            model="models/gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )

        # Updated embedding model initialization
        embed_model: HuggingFaceEmbedding = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-zh"  # Changed from model to model_name
        )

        # 加载数据
        documents: List[Document] = SimpleDirectoryReader(input_files=[os.path.join(ROOT_DIR, self.document_path)]).load_data() 

        # 构建索引
        index: VectorStoreIndex = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            # llm=llm  # 设置构建索引时的语言模型（一般不需要）
        )

        # 创建问答引擎
        query_engine: BaseQueryEngine = index.as_query_engine(
            llm=llm  # 设置生成模型
            )

        # 开始问答
        return str(query_engine.query(question))
