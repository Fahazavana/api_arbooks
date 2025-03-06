from api.bd_scraping_arbook.database import DatabaseManager
import logging
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from .chromadb import ChromaManager
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)


class Chatbot:
    _inited = False

    def __init__(self):
        self.collection = None
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.chroma_db = None

    async def initialize(self):
        if not self._inited:
            await self.init_db()
            await self.init_chroma()
            self._inited = True
        else:
            logging.info("Chatbot déjà initialisé")

    @classmethod
    async def create(cls):
        instance = cls()
        await instance.initialize()
        return instance

    async def init_db(self):
        db_manager = DatabaseManager()
        await db_manager.initialize()
        client = db_manager.get_client()
        self.collection = client["scraping_arbook"]["Product_scraping"]

    async def init_chroma(self):
        self.chroma_db = ChromaManager(self.collection)
        await self.chroma_db.initialize()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    async def retrieve(self, query):
        await self.chroma_db.update_if_needed()
        vector_store = self.chroma_db.get_vector_store()
        if not vector_store:
            return "La base de données vectorielle n'est pas initialisée."

        template = """
        Tu es un assistant de vente en ligne expert, conçu pour aider les clients à trouver les produits qu'ils recherchent.

        Un client te pose une question ou cherche un produit spécifique. Voici les informations sur les produits disponibles :

        {context}

        Question du client : {question}

        Important :
            Pour chaque produit pertinent, formate les informations suivantes comme indiqué :
            <produit>
                <id>{{_id}}</id>
                <source>{{source}}</source>
                <product_id>{{product_id}}</product_id>
            </produit>

        Réponds à la question du client en utilisant les informations fournies, en te concentrant sur le nom et la description des produits. Si plusieurs produits similaires sont disponibles, liste-les de manière claire et concise, en mettant en évidence leurs principales caractéristiques et différences.
        Si la question du client ne concerne pas directement les produits disponibles, réponds de manière informative et professionnelle, en précisant que tu ne peux pas fournir d'informations supplémentaires en dehors des produits présents dans ta base de données.

        Remarques Importantes :
            - Ne donne pas de lien dans ta reponse.
            - Considère un produit comme étant de "seconde main" si sa source n'est pas une source de produits neufs.
            - Limite ta réponse à 3 produits." \
        """

        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = self.llm
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        try:
            return rag_chain.invoke(query)
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du contexte: {e}")
            return f"Erreur lors de la récupération du contexte: {e}"

    async def handle_query(self, query):
        try:
            result = await self.collection.find_one(
                {"name": {"$regex": query, "$options": "i"}}
            )

            if result:
                return f"{result['name']} - {result['description']} \nPrice : {result.get('price', 'Non précisé')}€\n Stock : {result.get('stock', 'Non précisé')}"

            alternative = await self.retrieve(query)
            return alternative if alternative else "Produit non disponible."

        except Exception as e:
            logging.error(f"Erreur lors du traitement de votre demande: {e}")
            return f"Erreur lors du traitement de votre demande: {e}"
