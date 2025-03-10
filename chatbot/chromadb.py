from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import logging
import json

# Configuration des logs
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


class ChromaManager:
    """Gère l'initialisation et la mise à jour de la base de données vectorielle ChromaDB."""

    _instance = None
    _initialized = False
    _vector_store = None
    persist_directory = "./chroma_db"

    def __new__(cls, collection, *args, **kwargs):
        """Crée une instance unique de ChromaManager et stocke la collection MongoDB."""
        if cls._instance is None:
            cls._instance = super(ChromaManager, cls).__new__(cls)
            cls._instance.collection = collection
        return cls._instance

    async def __format_texts(self, products):
        """Formate les données des produits en texte pour l'intégration vectorielle."""

        def format_value(value):
            if isinstance(value, list):
                return ", ".join(map(str, value))
            elif isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)
            elif value is None:
                return "Non disponible."
            else:
                return str(value)

        return [
            f"""
            # Nom du produit/description: {format_value(p.get('name'))}
            * _id: {format_value(p.get('_id'))}
            * source: {format_value(p.get('source'))}
            * product_id: {format_value(p.get('product_id'))}
            * Prix: {format_value(p.get('price'))}
            * Description: {format_value(p.get('description'))}
            * Catégories: {format_value(p.get('categories'))}
            * Condition: {format_value(p.get('condition'))}
            * Tailles: {format_value(p.get('sizes'))}
            * En stock: {format_value(p.get('stock'))}
            * Marque: {format_value(p.get('brand'))}
            * Couleurs: {format_value(p.get('colors'))}
            * Caractéristiques/Description 1: {format_value(p.get('feature_table'))}
            * Caractéristiques/Description 2: {format_value(p.get('feature_bullet'))}
            """
            for p in products
        ]

    async def __add_new_documents(self):
        """Ajoute les nouveaux documents de MongoDB à ChromaDB."""
        try:
            chroma_ids = set()
            if self._vector_store:
                chroma_ids = set(self._vector_store._collection.get()["ids"])
                
            mongo_docs = await self.collection.find().to_list(length=None)
            new_docs = [doc for doc in mongo_docs if str(doc["_id"]) not in chroma_ids]

            if new_docs:
                new_texts = await self.__format_texts(new_docs)
                if not self._vector_store:
                    self._vector_store = Chroma.from_texts(
                        new_texts,
                        embedding=OpenAIEmbeddings(),
                        ids=[str(doc["_id"]) for doc in new_docs],
                        persist_directory=self.persist_directory,
                    )
                else:
                    self._vector_store.add_texts(
                        new_texts,
                        ids=[str(doc["_id"]) for doc in new_docs],
                    )
                logging.info(
                    f"Ajout de {len(new_texts)} nouveaux documents à ChromaDB."
                )
            else:
                logging.info("Aucun nouveau document trouvé dans MongoDB.")

        except Exception as e:
            logging.error(
                f"Erreur lors de l'ajout de nouveaux documents à ChromaDB: {e}"
            )

    async def initialize(self):
        """Initialise la base de données vectorielle ChromaDB."""
        if ChromaManager._initialized:
            logging.info("ChromaDB déjà initialisé.")
            return

        try:
            if os.path.exists(self.persist_directory) and any(
                os.scandir(self.persist_directory)
            ):
                self._vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=OpenAIEmbeddings(),
                )
                logging.info("ChromaDB chargé depuis le disque.")
                await self.__add_new_documents()
            else:
                await self.__add_new_documents()
            ChromaManager._initialized = True

        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation de ChromaDB: {e}")

    def get_vector_store(self):
        """Retourne la base de données vectorielle ChromaDB initialisée."""
        if not ChromaManager._initialized:
            logging.warning("ChromaDB n'est pas encore initialisé.")
            return None
        return self._vector_store

    async def update_if_needed(self):
        """Met à jour ChromaDB avec les nouveaux documents de MongoDB."""
        if not self._initialized:
            if os.path.exists(self.persist_directory) and any(
                os.scandir(self.persist_directory)
            ):
                logging.info("ChromaDB non initialisé, rechargement depuis le disque.")
                self._vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=OpenAIEmbeddings(),
                )
                await self.__add_new_documents()
                ChromaManager._initialized = True
                return
            else:
                logging.info("ChromaDB non initialisé, initialisation depuis MongoDB.")
                await self.initialize()
                return

        mongo_count = await self.collection.count_documents({})
        chroma_count = self._vector_store._collection.count()
        if mongo_count != chroma_count:
            await self.__add_new_documents()
        else:
            logging.info("ChromaDB est à jour.")

    @classmethod
    async def create(cls):
        """Create a new ChromaManager instance and initialize it."""
        instance = cls(None) 
        await instance.initialize()
        return instance

    def get_collection(self):
        """Return the chromadb collection."""
        return self._vector_store._collection if self._vector_store else None

    def get_similar(self, query_id, embedding, n_results=5):
        """Get similar embeddings from ChromaDB."""
        if not self._vector_store:
            logging.warning("ChromaDB n'est pas encore initialisé.")
            return None
        results = self._vector_store.similarity_search_by_vector(
            embedding=embedding, k=n_results
        )
        return [doc.id for doc in results if doc.id != query_id]
