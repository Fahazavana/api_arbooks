from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from app.bd_scraping_arbook.database import DatabaseManager
import logging

async def create_vector_store():
    """Charge les produits depuis MongoDB et crée un index vectoriel."""
    db_manager = DatabaseManager()
    await db_manager.initialize()

    client = db_manager.get_client()
    collection = client["scraping_arbook"]["Product_scraping"]

    products = await collection.find({}, {"_id": 0, "name": 1, "description": 1, "price": 1, "url": 1}).to_list(length=None)

    if not products:
        logging.warning(" Aucun produit trouvé en base MongoDB.")
        return None

    # Créer des descriptions si elles sont nulles
    texts = [
        f"{p['name']} - {p['description'] if p['description'] else 'Prix: ' + p['price']} - Voir ici: {p['url']}"
        for p in products
    ]

    vector_store = Chroma.from_texts(texts, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db")

    logging.info(f" {len(products)} produits indexés dans ChromaDB.")
    return vector_store
