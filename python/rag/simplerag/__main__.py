import os

def ex_00_01_01():
    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        # 第一行代码：导入相关的库
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader 
        # 第二行代码：加载数据
        documents = SimpleDirectoryReader(input_files=[os.path.join(ROOT_DIR, "90-文档-Data/黑悟空/设定.txt")]).load_data() 
        # 第三行代码：构建索引
        index = VectorStoreIndex.from_documents(documents)
        # 第四行代码：创建问答引擎
        query_engine = index.as_query_engine()
        # 第五行代码: 开始问答
        print(query_engine.query("黑神话悟空中有哪些战斗工具?"))
    else:
        print("OPENAI_API_KEY is not set, cannot proceed.")

def ex_00_01_03():
    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    # 导入相关的库
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
    llm = GoogleGenAI(
        model="models/gemini-1.5-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )
    embed_model = HuggingFaceEmbedding(
        model="BAAI/bge-small-zh"
    )

    # 加载数据
    documents = SimpleDirectoryReader(input_files=[os.path.join(ROOT_DIR, "90-文档-Data/黑悟空/设定.txt")]).load_data() 

    # 构建索引
    index = VectorStoreIndex.from_documents(
        documents,
        # llm=llm  # 设置构建索引时的语言模型（一般不需要）
    )

    # 创建问答引擎
    query_engine = index.as_query_engine(
        llm=llm  # 设置生成模型
        )

    # 开始问答
    print(query_engine.query("黑神话悟空中有哪些战斗工具?"))


def main():
    ex_00_01_03()

if __name__ == "__main__":
    main()