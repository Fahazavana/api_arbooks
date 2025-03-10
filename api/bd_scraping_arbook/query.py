from fuzzywuzzy import fuzz
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import Document
from pydantic import BaseModel
from typing import List, Optional, Union, Dict
import logging
import re
from pymongo import ASCENDING, DESCENDING
import os

logging.basicConfig(level=os.environ.get("LOGLEVEL"))

class ProductResponse(BaseModel):
    source:str
    id: Optional[str]
    product_id: Optional[str] = None
    name: Optional[str] = None
    price: Optional[str] = None
    url: Optional[str] = None
    main_photo: Optional[str] = None
    description: Optional[str] = None
    price_with_protection: Optional[str] = None
    categories: Optional[List[str]] = None
    detailed_photos: Optional[List[str]] = None
    condition: Optional[str] = None
    sizes: Optional[Union[str, List[str]]] = None
    delivery_price: Optional[str] = None
    stock: Optional[bool] = None
    is_exclusive: Optional[bool] = None
    rating: Optional[str] = None
    brand: Optional[str] = None
    colors: Optional[Union[str, List[str], Dict[str, str]]] = None
    views: Optional[int] = None
    interested: Optional[int] = None
    uploaded: Optional[str] = None
    payment_methods: Optional[str] = None
    owner_name: Optional[str] = None
    owner_profile_url: Optional[str] = None
    feature_table: Optional[Dict[str, str]] = None
    feature_bullet: Optional[List[str]] = None


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

    async def search_categories(self, query, similarity_threshold=80) -> List[ProductResponse]:
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

    @staticmethod
    def _filter_results(results, query, similarity_threshold):
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
                    result["id"] = str(result["_id"])
                    filtered_results.append(result)
        return filtered_results

    @staticmethod
    def __format_results(results):
        """Formate les résultats en convertissant _id en chaîne."""
        formatted_results = []
        for result in results:
            try:
                result["id"] = str(result["_id"])
                formatted_results.append(result)
            except Exception as e:
                logging.error(f"Erreur lors du formatage des résultats: {e}")
        return formatted_results

    async def get_all_product(self, source: Optional[str] = None) -> List[ProductResponse]:
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

    async def search_products_by_name(self, name: str) -> List[ProductResponse]:
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

    # async def search_products_by_price_range(
    #     self, min_price: float, max_price: float
    # ) -> List[ProductResponse]:
    #     """Recherche les produits dans une plage de prix donnée."""
    #     if not await self.__check_db():
    #         return []
    #     try:
    #         results = await self.collection.find(
    #             {"price": {"$gte": min_price, "$lte": max_price}}
    #         ).to_list()
    #         if results:
    #             return self.__format_results(results)
    #         return []
    #     except Exception as e:
    #         logging.error(
    #             f"Erreur lors de la recherche des produits par plage de prix: {e}"
    #         )
    #         return []

    async def search_products_by_brand(self, brand: str) -> List[ProductResponse]:
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

    async def search_products_by_condition(self, condition: str) -> List[ProductResponse]:
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
    ) -> List[ProductResponse]:
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

    async def get_products_with_pagination(
        self, page: int = 1, page_size: int = 10
    ) -> List[ProductResponse]:
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
            logging.error(f"Erreur lors de la récupération des produits avec pagination: {e}")
            return []
