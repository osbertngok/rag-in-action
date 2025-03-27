import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.simplerag.interface import RAGWrapperInterface

from typing import Dict, List, Any

class LangChainWrapper(RAGWrapperInterface):

    def __init__(self, url: str) -> None:
        self.url = url

        # 1. 加载文档
        import os
        from dotenv import load_dotenv
        # 加载环境变量
        load_dotenv()

        loader = WebBaseLoader(
            web_paths=(self.url,)
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

        """
        https://smith.langchain.com/hub/rlm/rag-prompt

        human

        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.

        Question: {question} 
        Context: {context} 
        Answer:

        """
        from langchain import hub
        prompt: ChatPromptTemplate = hub.pull("rlm/rag-prompt")

        # Or, 
        #
        # prompt = ChatPromptTemplate.from_template("""
        #                 基于以下上下文，回答问题。如果上下文中没有相关信息，
        #                 请说"我无法从提供的上下文中找到相关信息"。
        #                 上下文: {context}
        #                 问题: {question}
        #                 回答:"""
        #

        # 6. 定义应用状态
        from typing import List
        from typing_extensions import TypedDict
        from langchain_core.documents import Document
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        # 7. 定义检索步骤
        def retrieve(state: State) -> Dict[str, List[Document]]:
            retrieved_docs = self.vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        # 8. 定义生成步骤
        def generate(state: State) -> Dict[str, str | List[str | Dict[Any, Any]]]:
            llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,        # 控制输出的随机性(0-1之间,越大越随机)
                max_tokens=2048,        # 最大输出长度
                top_p=0.95,            # 控制输出的多样性(0-1之间)
                top_k=50,              # 控制每次选择的候选token数量
                transport="rest" # 默认是grpc
            )
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = llm.invoke(messages)
            return {"answer": response.content}

        # 9. 构建和编译应用
        from langgraph.graph import START, StateGraph # pip install langgraph
        from langgraph.graph.state import CompiledStateGraph
        
        graph: CompiledStateGraph = (
            StateGraph(State)
            .add_sequence([retrieve, generate])
            .add_edge(START, "retrieve")
            .compile()
        )                                         

        # 10. 运行查询
        resp: Dict[str, str] = graph.invoke({"question": question})
        return str(resp["answer"])


