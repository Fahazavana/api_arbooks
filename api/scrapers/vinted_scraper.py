import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from .BaseScraper import BaseScraper
from api.bd_scraping_arbook.models_scraping import Product_scraping
from .utils import Product


async def save_to_mongo(db_manager, products: list[dict]):
    """Insère les produits scrappés dans MongoDB en évitant les doublons (mise à jour si déjà existant)."""

    # Initialiser la base de données si ce n'est pas déjà fait
    if not db_manager.is_initialized():
        success = await db_manager.initialize()
        if not success:
            logging.error(
                "Échec de l'initialisation de la base de données. Annulation de l'insertion."
            )
            return

    if not products:
        logging.info("Aucun produit à enregistrer dans MongoDB.")
        return

    for item in products:
        try:
            # Utilisation de Beanie pour insérer ou mettre à jour le produit
            existing_product = await Product_scraping.find_one(
                {"product_id": item["product_id"]}
            )
            if existing_product:
                await existing_product.set(item)  # Mise à jour
                logging.info(
                    f"Produit {item['name']} ({item['product_id']}) mis à jour avec succès !"
                )
            else:
                new_product = Product_scraping(**item)
                await new_product.insert()  # Insertion
                logging.info(
                    f"Produit {item['name']} ({item['product_id']}) inséré avec succès !"
                )

        except Exception as e:
            logging.error(
                f"Erreur lors de l'insertion du produit {item.get('product_id', 'inconnu')}: {e}"
            )


class VintedScraper(BaseScraper):

    BASE_URL = "https://www.vinted.fr"
    SEARCH_URL = "https://www.vinted.fr/catalog"

    def __init__(self, db_manager):
        self.db_manager = db_manager
        """Initialisation du navigateur avec les options."""
        options = Options()
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        )
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--headless")  # Set to False to see the browser
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-gpu")
        options.add_argument("--start-maximized")
        options.add_argument("--log-level=3")  # Reduce unnecessary logs

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *excinfo):
        self.driver.quit()

    async def get_page_content(self, url: str, wait_for: str) -> str:
        """Récupérer le contenu de la page avec Selenium et gestion des erreurs."""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
            )
            return self.driver.page_source
        except Exception as e:
            logging.error(
                f"Erreur lors du chargement de la page {url} : {str(e)}"
            )  # More specific error message
            return ""

    async def search(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        produits = []
        params = {
            "search_text": query,
            "order": "newest_first",
        }
        url = f"{self.SEARCH_URL}?{self._encode_params(params)}"  # build the url here
        try:
            logging.info(f" Recherche de '{query}' sur Vinted...")
            content = await self.get_page_content(url, "div.feed-grid__item-content")
            if not content:
                return []

            soup = BeautifulSoup(content, "lxml")
            items = soup.find_all("div", class_="feed-grid__item-content")
            logging.info(f"Trouvé {len(items)} articles sur Vinted pour: {query}")

            for item in items[:limit]:
                produit = self.parse_item(item)
                if produit:
                    produits.append(produit)

            # Sauvegarde dans MongoDB après extraction
            if produits:
                logging.info(" Enregistrement des produits dans MongoDB...")
                await save_to_mongo(self.db_manager, produits)

        except Exception as e:
            logging.error(f"Erreur lors du scraping de Vinted: {str(e)}")
            raise

        return produits

    def parse_item(self, item) -> Optional[Dict[str, Any]]:
        product = Product(source="vinted")
        try:
            # ID du produit (data-testid)
            product_id_element = item.select_one(
                '[data-testid*="product-item-id"]'
            )  # Select by partial attribute
            product.product_id = (
                product_id_element.get("data-testid").split("--")[0].split("-")[-1]
                if product_id_element
                else None
            )

            # Titre (plusieurs options car la structure peut varier)
            title_element = item.select_one(
                ".new-item-box__description p.web_ui__Text__text"
            )
            product.name = title_element.text.strip() if title_element else None

            # Prix
            price_element = item.select_one('p[data-testid*="--price-text"]')
            product.price = price_element.text.strip() if price_element else None

            # URL du produit
            link_element = item.select_one("a.new-item-box__overlay")
            product.url = (
                link_element.get("href")
                if link_element and link_element.get("href")
                else None
            )

            # Image URL
            image_element = item.select_one("img.web_ui__Image__content")
            product.main_photo = image_element.get("src") if image_element else None

            #  Description
            description = item.select_one('p[data-testid*="description-subtitle"]')
            product.description = description.text.strip() if description else None

            # prix avec protection
            price_element_pro = item.select_one(
                'button[aria-label*="Protection"]>span>span'
            )
            product.price_with_protection = (
                price_element_pro.text.strip() if price_element else None
            )

            if product.is_valid():
                logging.info(product)
                return product.to_dict()
            return None

        except Exception as e:
            logging.error(f"Erreur extraction article Vinted: {str(e)}")
            return None

    def parse_detail(self, soup) -> List[Dict[str, Any]]:
        product = Product(source="vinted")
        try:
            # Extract the canonical link for the product URL
            canonical_link = soup.select_one('link[rel="canonical"]')
            if canonical_link:
                canonical_url = canonical_link["href"]
                product.url = canonical_url
                product.product_id = canonical_url.split("/")[-1]

            # Extract product name
            name = soup.select_one(
                'aside div[data-testid="item-page-summary-plugin"] div span'
            )
            product.name = name.text.strip() if name else None

            # Extract detail container for further information
            detail_container = soup.select_one(
                "aside div.details-list.details-list--details"
            )
            if not detail_container:
                return []

            # Determine stock status based on the presence of status element
            status = detail_container.select_one(
                'div[data-testid="item-status--content"]'
            )
            product.stock = False if status else True

            # Extract price information
            price = soup.select_one('aside div[data-testid="item-price"]')
            product.price = price.text.strip().replace("\xa0", " ") if price else None

            # Extract price with protection information
            price_protection = soup.select_one(
                'aside button[aria-label*="Protection"] > div'
            )
            product.price_with_protection = (
                price_protection.text.strip().replace("\xa0", " ")
                if price_protection
                else None
            )

            # Extract categories from breadcrumbs
            categories = soup.select("ul.breadcrumbs.breadcrumbs--truncated a")
            if categories:
                product.categories = [
                    categorie.text.strip() for categorie in categories
                ][1:]

            # Extract detailed photos
            photos = soup.select_one("section.item-photos__container")
            if photos:
                imgs = photos.find_all("img")
                product.detailed_photos = [img.get("src") for img in imgs]

            # Extract additional product details
            brand_element = detail_container.select_one('span[itemprop="name"]')
            product.brand = brand_element.text.strip() if brand_element else None

            # Extract sizes
            size_element = detail_container.select_one('div[itemprop="size"]')
            product.sizes = size_element.text.strip() if size_element else None

            # Extract product condition
            condition_element = detail_container.select_one('div[itemprop="status"]')
            product.condition = (
                condition_element.text.strip() if condition_element else None
            )

            # Extract product color
            color_element = detail_container.select_one('div[itemprop="color"]')
            product.colors = color_element.text.strip() if color_element else None

            # Extract view count
            views_element = detail_container.select_one('div[itemprop="view_count"]')
            product.views = int(views_element.text.strip()) if views_element else None

            # Extract interested count
            interested_element = detail_container.select_one(
                'div[itemprop="interested"]'
            )
            product.interested = (
                int(interested_element.text.strip().split(" ")[0])
                if interested_element
                else None
            )

            # Extract payment methods
            payment_element = detail_container.select_one(
                'div[itemprop="payment_methods"] span'
            )
            product.payment_methods = (
                payment_element.text.strip() if payment_element else None
            )

            # Extract uploaded date
            uploaded_element = detail_container.select_one(
                'div[data-testid="item-attributes-upload_date"] [itemprop="upload_date"]'
            )
            product.uploaded = (
                {
                    "scraped": datetime.now().strftime("%Y-%m-%d"),
                    "time": uploaded_element.text.strip(),
                }
                if uploaded_element
                else None
            )

            # Extract delivery price
            delivery_element = soup.select_one(
                'aside [data-testid="item-shipping-banner-price"]'
            )
            product.delivery_price = (
                delivery_element.text.strip().replace("\xa0", " ")
                if delivery_element
                else None
            )

            # Extract product description
            description_element = soup.select_one('aside div[itemprop="description"]')
            product.description = (
                description_element.text.strip().replace("\n", "")
                if description_element
                else None
            )

            # Extract owner information
            owner_name = soup.select_one('aside [data-testid="profile-username"]')
            product.owner_name = owner_name.text.strip() if owner_name else ""

            owner_link_element = soup.select_one(
                "aside a.web_ui__Cell__cell.web_ui__Cell__default.web_ui__Cell__navigating.web_ui__Cell__with-chevron.web_ui__Cell__link"
            )
            owner_url = owner_link_element.get("href") if owner_link_element else None
            if owner_url:
                owner_url = self.BASE_URL + owner_url
            product.owner_profile_url = owner_url

            # Validate the product before returning
            if product.is_valid():
                return [product.to_dict()]
            else:
                return []
        except Exception as e:
            logging.error(f"Erreur extraction detail article Vinted: {str(e)}")
            return []

    # async def get_detail(self, product_url: str) -> List[Dict[str, Any]]:
    #     logging.info("Obtenir la page contenant les detail du produits")
    #     content = await self.get_page_content(product_url, "aside")

    #     if not content:
    #         return []
    #     soup = BeautifulSoup(content, "lxml")

    #     logging.info("Extraction des informations importantes")
    #     return self.parse_detail(soup)

    async def get_detail(self, product_url: str) -> List[Dict[str, Any]]:
        """Récupère et met à jour les détails d'un produit sur Vinted."""

        logging.info(f" Récupération du contenu de la page : {product_url}")
        content = await self.get_page_content(product_url, "aside")

        if not content:
            logging.warning(f" Aucun contenu récupéré pour {product_url}.")
            return []

        soup = BeautifulSoup(content, "lxml")

        logging.info(" Extraction des informations importantes...")
        details = self.parse_detail(soup)

        if not details:
            logging.warning(f" Impossible d'extraire les détails pour {product_url}.")
            return []

        logging.info(f" Détails extraits : {details}")

        #  Correction : Passer le dictionnaire extrait à update_product_details()
        updated_details = await self.update_product_details(details[0])

        return [updated_details]  # Toujours retourner une liste

    async def update_product_details(
        self, detailed_product: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Met à jour MongoDB avec les nouvelles données du produit."""
        try:
            if not isinstance(detailed_product, dict):
                logging.error(
                    f" Erreur: `detailed_product` n'est pas un dictionnaire ! Type reçu : {type(detailed_product)}"
                )
                return {}

            product_id = detailed_product.get("product_id")
            if not product_id:
                logging.warning(" Impossible de récupérer l'ID du produit.")
                return {}

            #  Vérifier que MongoDB est bien initialisé
            if not self.db_manager.is_initialized():
                logging.info("🛠️ Initialisation de MongoDB avec Beanie...")
                await self.db_manager.initialize()

            #  Correction : Convertir `uploaded` en `str` si c'est un `dict`
            if "uploaded" in detailed_product and isinstance(
                detailed_product["uploaded"], dict
            ):
                uploaded_data = detailed_product["uploaded"]
                detailed_product["uploaded"] = (
                    f"{uploaded_data.get('scraped', 'N/A')} - {uploaded_data.get('time', 'N/A')}"
                )

            #  Rechercher si le produit existe déjà en base
            existing_product = await Product_scraping.find_one(
                {"product_id": product_id}
            )

            if existing_product:
                updated_fields = {
                    k: v
                    for k, v in detailed_product.items()
                    if getattr(existing_product, k, None) != v and v is not None
                }

                if updated_fields:
                    try:
                        await existing_product.set(updated_fields)
                        logging.info(
                            f" Produit {product_id} mis à jour avec {len(updated_fields)} nouvelles valeurs."
                        )
                    except Exception as e:
                        logging.error(
                            f" Erreur lors de la mise à jour du produit `{product_id}` dans MongoDB : {e}"
                        )
                else:
                    logging.info(f"ℹ️ Aucun changement détecté pour {product_id}.")
            else:
                try:
                    new_product = Product_scraping(**detailed_product)
                    await new_product.insert()
                    logging.info(f" Produit {product_id} ajouté en base.")
                except Exception as e:
                    logging.error(
                        f" Erreur lors de l'insertion du produit `{product_id}` dans MongoDB : {e}"
                    )

            return detailed_product

        except Exception as e:
            logging.error(
                f" Erreur inattendue lors de la mise à jour du produit `{product_id}` : {e}"
            )
            return {}

    async def update_product_details(
        self, detailed_product: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Met à jour MongoDB avec les nouvelles données du produit Amazon."""
        try:
            if not isinstance(detailed_product, dict):
                logging.error(
                    f" Erreur: `detailed_product` n'est pas un dictionnaire ! Type reçu : {type(detailed_product)}"
                )
                return {}

            product_id = detailed_product.get("product_id")
            if not product_id:
                logging.warning(" Impossible de récupérer l'ID du produit.")
                return {}

            #  Vérifier que MongoDB est bien initialisé
            if not self.db_manager.is_initialized():
                logging.info("🛠️ Initialisation de MongoDB avec Beanie...")
                await self.db_manager.initialize()

            #  Correction : Convertir `uploaded` en `str` si c'est un `dict`
            if "uploaded" in detailed_product and isinstance(
                detailed_product["uploaded"], dict
            ):
                uploaded_data = detailed_product["uploaded"]
                detailed_product["uploaded"] = (
                    f"{uploaded_data.get('scraped', 'N/A')} - {uploaded_data.get('time', 'N/A')}"
                )

            #  Rechercher si le produit existe déjà en base
            existing_product = await Product_scraping.find_one(
                {"product_id": product_id}
            )

            if existing_product:
                updated_fields = {
                    k: v
                    for k, v in detailed_product.items()
                    if getattr(existing_product, k, None) != v and v is not None
                }

                if updated_fields:
                    try:
                        await existing_product.set(updated_fields)
                        logging.info(
                            f" Produit {product_id} mis à jour avec {len(updated_fields)} nouvelles valeurs."
                        )
                    except Exception as e:
                        logging.error(
                            f" Erreur lors de la mise à jour du produit `{product_id}` dans MongoDB : {e}"
                        )
                else:
                    logging.info(f"ℹ️ Aucun changement détecté pour {product_id}.")
            else:
                try:
                    new_product = Product_scraping(**detailed_product)
                    await new_product.insert()
                    logging.info(f" Produit {product_id} ajouté en base.")
                except Exception as e:
                    logging.error(
                        f" Erreur lors de l'insertion du produit `{product_id}` dans MongoDB : {e}"
                    )

            return detailed_product

        except Exception as e:
            logging.error(
                f" Erreur inattendue lors de la mise à jour du produit `{product_id}` : {e}"
            )
            return {}
