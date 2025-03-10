from chatbot.chromadb import ChromaManager
from api.bd_scraping_arbook.database import DatabaseManager
import logging
from bson.objectid import ObjectId

class RecSys:
    def __init__(self):
        self.chroma_db = None
        self.collection = None
        self._inited = False

    async def initialize(self):
        if not self._inited:
            await self.init_db()
            await self.init_chroma()
            self._inited = True
        else:
            logging.info("Le système de recommandation a déjà été initialisé.")

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
        self.chroma_db = await ChromaManager.create()
        self.chroma_db.collection = self.collection
        await self.chroma_db.update_if_needed()

    async def get_similar(self, query_id, n_results=5):
        try:
            product = await self.collection.find_one({"_id": ObjectId(query_id)})
            if product is None:
                logging.warning(f"Produit avec l'ID {query_id} non trouvé.")
                return []
            await self.chroma_db.update_if_needed()
            chroma_results = self.chroma_db.get_collection().get(ids=[query_id], include=["embeddings"])
            if not chroma_results:
                logging.warning(f"Embedding non trouvé dans ChromaDB pour le produit avec l'ID {query_id}.")
                return []

            embedding = chroma_results["embeddings"][0]
            results = self.chroma_db.get_similar(query_id, embedding, n_results)
            
            if results:
                similar_products = []
                for prod_id in results:
                    try:
                        similar_product = await self.collection.find_one({"_id": ObjectId(prod_id)})
                        if similar_product:
                            similar_product['_id'] = str(similar_product['_id'])
                            similar_products.append(similar_product)
                    except Exception as e:
                        logging.error(f"Erreur lors de la récupération du produit similaire avec l'ID {query_id}: {e}")
                return similar_products
            else:
                logging.warning(f"Aucun produit similaire trouvé pour le produit avec l'ID {query_id}.")
                return []

        except Exception as e:
            logging.error(f"Erreur lors de la récupération des produits similaires: {e}")
            return []