import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.simplerag.interface import RAGWrapperInterface

class LangChainWrapper(RAGWrapperInterface):

    def __init__(self, url: str) -> None:
        self.url = url

        # 1. 加载文档
        import os
        from dotenv import load_dotenv
        # 加载环境变量
        load_dotenv()

        loader = WebBaseLoader(
            web_paths=("https://zh.wikipedia.org/wiki/黑神话：悟空",)
        )
        docs = loader.load()

        # 2. 文档分块

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # 3. 设置嵌入模型


        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 4. 创建向量存储
        self.vector_store = InMemoryVectorStore(embeddings)
        self.vector_store.add_documents(all_splits)

    def get_answer(self, question: str) -> str:
        
        # 5. 构建用户查询
        question = "黑悟空有哪些游戏场景？"

        # 6. 在向量存储中搜索相关文档，并准备上下文内容
        retrieved_docs = self.vector_store.similarity_search(question, k=3)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 7. 构建提示模板


        prompt = ChatPromptTemplate.from_template("""
                        基于以下上下文，回答问题。如果上下文中没有相关信息，
                        请说"我无法从提供的上下文中找到相关信息"。
                        上下文: {context}
                        问题: {question}
                        回答:"""
                                                )

        # 8. 使用大语言模型生成答案
        llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,        # 控制输出的随机性(0-1之间,越大越随机)
            max_tokens=2048,        # 最大输出长度
            top_p=0.95,            # 控制输出的多样性(0-1之间)
            top_k=50,              # 控制每次选择的候选token数量
            presence_penalty=0.0,   # 重复惩罚系数(-2.0到2.0之间)
            frequency_penalty=0.0,  # 频率惩罚系数(-2.0到2.0之间)
            google_api_key=os.getenv("GOOGLE_API_KEY"),  # 从环境变量加载API key
            transport="rest"
        )

        answer = llm.invoke(prompt.format(question=question, context=docs_content))
        return str(answer)


