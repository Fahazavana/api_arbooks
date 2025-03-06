from fuzzywuzzy import fuzz
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import Document
from typing import List, Optional
import logging
import re
from pymongo import ASCENDING, DESCENDING


class Query:
    """Classe pour effectuer des recherches"""

    def __init__(self, db_manager):
        """Initialise la classe Query avec un db_manager."""
        self.db_manager = db_manager
        self.collection = None

    async def __check_db(self):
        """Vérifie et initialise la base de données et la collection."""
        if not self.db_manager.is_initialized():
            success = await self.db_manager.initialize()
            if not success:
                logging.error("Échec de l'initialisation de la base de données.")
                return False
        if self.collection is None:
            client: AsyncIOMotorClient = self.db_manager.get_client()
            if not client:
                logging.error("MongoDB client is not initialized.")
                return False
            db = client[self.db_manager.database_name]
            self.collection = db["Product_scraping"]
            logging.info("Collection Product_scraping initialisée.")
        return True

    async def search_categories(self, query, similarity_threshold=80) -> List[Document]:
        """Effectue une recherche floue sur les catégories de produits."""
        if not await self.__check_db():
            return []
        try:
            results = await self.collection.find().to_list(length=None)
            filtered_results = self._filter_results(
                results, query, similarity_threshold
            )
            return filtered_results
        except Exception as e:
            logging.error(f"Erreur lors de la recherche dans la base de données: {e}")
            return []

    def _filter_results(self, results, query, similarity_threshold):
        """Filtre les résultats en utilisant fuzzywuzzy."""
        filtered_results = []
        for result in results:
            categories = result.get("categories", [])
            if categories is None:
                continue
            for category in categories:
                if category is None:
                    continue
                if (
                    fuzz.partial_ratio(query.lower(), category.lower())
                    > similarity_threshold
                ):
                    result["_id"] = str(result["_id"])
                    filtered_results.append(result)
        return filtered_results

    def __format_results(self, results):
        """Formate les résultats en convertissant _id en chaîne."""
        formatted_results = []
        for result in results:
            try:
                result["_id"] = str(result["_id"])
                formatted_results.append(result)
            except Exception as e:
                logging.error(f"Erreur lors du formatage des résultats: {e}")
        return formatted_results

    async def get_all_product(self, source: Optional[str] = None) -> List[Document]:
        """Récupère tous les produits, éventuellement filtrés par source."""
        if not await self.__check_db():
            return []
        try:
            query = {"source": source} if source else {}
            results = await self.collection.find(query).to_list()
            if results:
                return self.__format_results(results)
            return []
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des produits: {e}")
            return []

    async def get_products_by_category(self, category: str) -> List[Document]:
        """Récupère tous les produits appartenant à une catégorie spécifique."""
        if not await self.__check_db():
            return []
        try:
            results = await self.collection.find({"categories": category}).to_list()
            if results:
                return self.__format_results(results)
            return []
        except Exception as e:
            logging.error(
                f"Erreur lors de la récupération des produits par catégorie: {e}"
            )
            return []

    async def search_products_by_name(self, name: str) -> List[Document]:
        """Recherche les produits par nom, en utilisant une recherche floue."""
        if not await self.__check_db():
            return []
        try:
            regex_pattern = re.compile(re.escape(name), re.IGNORECASE)
            results = await self.collection.find(
                {"name": {"$regex": regex_pattern}}
            ).to_list()
            if results:
                return self.__format_results(results)
            return []
        except Exception as e:
            logging.error(f"Erreur lors de la recherche des produits par nom: {e}")
            return []

    async def search_products_by_price_range(
        self, min_price: float, max_price: float
    ) -> List[Document]:
        """Recherche les produits dans une plage de prix donnée."""
        if not await self.__check_db():
            return []
        try:
            results = await self.collection.find(
                {"price": {"$gte": min_price, "$lte": max_price}}
            ).to_list()
            if results:
                return self.__format_results(results)
            return []
        except Exception as e:
            logging.error(
                f"Erreur lors de la recherche des produits par plage de prix: {e}"
            )
            return []

    async def search_products_by_brand(self, brand: str) -> List[Document]:
        """Recherche les produits par marque."""
        if not await self.__check_db():
            return []
        try:
            results = await self.collection.find({"brand": brand}).to_list()
            if results:
                return self.__format_results(results)
            return []
        except Exception as e:
            logging.error(f"Erreur lors de la recherche des produits par marque: {e}")
            return []

    async def search_products_by_condition(self, condition: str) -> List[Document]:
        """Recherche les produits par état (condition)."""
        if not await self.__check_db():
            return []
        try:
            results = await self.collection.find({"condition": condition}).to_list()
            if results:
                return self.__format_results(results)
            return []
        except Exception as e:
            logging.error(f"Erreur lors de la recherche des produits par état: {e}")
            return []

    async def search_products_by_description_keywords(
        self, keywords: str
    ) -> List[Document]:
        """Recherche les produits par mots-clés dans la description."""
        if not await self.__check_db():
            return []
        try:
            regex_pattern = re.compile(re.escape(keywords), re.IGNORECASE)
            results = await self.collection.find(
                {"description": {"$regex": regex_pattern}}
            ).to_list()
            if results:
                return self.__format_results(results)
            return []
        except Exception as e:
            logging.error(
                f"Erreur lors de la recherche des produits par mots-clés dans la description: {e}"
            )
            return []
        
    async def get_products_with_pagination(self, page: int = 1, page_size: int = 10) -> List[Document]:
        """Récupère les produits avec pagination."""
        if not await self.__check_db():
            return []
        try:

            skip = (page - 1) * page_size
            results = await self.collection.find().skip(skip).limit(page_size).to_list()
            if results:
                return self.__format_results(results)
            return []
        except Exception as e:
            print(f"Erreur lors de la récupération des produits avec pagination: {e}")
            return []