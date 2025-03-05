from app.bd_scraping_arbook.database import DatabaseManager
from .retrieval import get_product_info

async def search_product(product_name):
    """Vérifie en base et lance la recherche RAG si nécessaire."""
    db_manager = DatabaseManager()
    await db_manager.initialize()

    collection = db_manager.get_client()["scraping_arbook"]["products"]
    result = await collection.find_one({"name": {"$regex": product_name, "$options": "i"}})

    if result:
        return f" {result['name']} - {result['description']} \n Prix : {result.get('price', 'Non précisé')}€\n📦 Stock : {result.get('stock', 'Non précisé')}"

    # 🔍 Recherche sémantique avec RAG si le produit n'est pas en base
    alternative = await get_product_info(product_name)
    return alternative if alternative else " Produit non disponible."
