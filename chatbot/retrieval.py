from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from .vector_store import create_vector_store

async def get_product_info(query: str):
    """Recherche un produit en base et génère une réponse."""
    vector_store = await create_vector_store()
    
    if not vector_store:
        return " Aucun produit en base."

    llm = ChatOpenAI(model="gpt-4")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

    return qa.invoke(query)

