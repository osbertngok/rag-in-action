import os
from typing import List


def ex_00_01_03() -> None:
    ROOT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    # 导入相关的库
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Document
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.llms.google_genai import GoogleGenAI

    from llama_index.core import Settings # 可以看看有哪些Setting
    # https://docs.llamaindex.ai/en/stable/examples/llm/deepseek/
    # Settings.llm = DeepSeek(model="deepseek-chat")
    # Settings.embed_model = HuggingFaceEmbedding(model="BAAI/bge-small-zh")
    # Settings.llm = OpenAI(model="gpt-3.5-turbo")
    # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

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
    documents: List[Document] = SimpleDirectoryReader(input_files=[os.path.join(ROOT_DIR, "90-文档-Data/黑悟空/设定.txt")]).load_data() 

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
    print(query_engine.query("黑神话悟空中有哪些战斗工具?"))


def main() -> None:
    ex_00_01_03()

if __name__ == "__main__":
    main()