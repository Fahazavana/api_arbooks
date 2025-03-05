import asyncio
from .query_handler import search_product

async def chatbot():
    while True:
        user_query = input(" Quel produit recherchez-vous ? ")
        response = await search_product(user_query)
        print(response)

if __name__ == "__main__":
    asyncio.run(chatbot())





# Modèle pour la requête utilisateur
class SearchRequest(BaseModel):
    query: str

@app.post("/search/")
async def search(request: SearchRequest):
    """🔍 Recherche un produit dans MongoDB et via RAG."""
    response = await search_product(request.query)
    return {"query": request.query, "response": response}
