from fastapi import FastAPI, APIRouter, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from fastapi import Query as FastAPIQuery
import os
import shutil
import logging

# Import custom modules
from fonction import exif_tools, image_tools, model_tools, ollama_tools
from .bd_scraping_arbook.database import DatabaseManager
from .scrapers.vinted_scraper import VintedScraper
from .scrapers.amazon_scraper import AmazonScraper
from services_reconnaissance.face_recognition import capture_face, recognize_face
from database.db import get_db
from .bd_scraping_arbook.query import Query
from .scrapers.utils import Product


# Chatbot
from chatbot.chat import Chatbot

# Disable logging by setting the level to WARNING
logging.basicConfig(level=logging.WARNING)

# Initialize FastAPI application and router
app = FastAPI()
router = APIRouter()

# Directory for uploaded files
UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Endpoint for uploading and analyzing an image
@router.post("/upload/", tags=["Image"])
async def upload_image(file: UploadFile, user_description: str = Form(...)):
    try:
        # Save the uploaded image
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract EXIF metadata
        metadata = exif_tools.extract_metadata(file_path)
        if metadata:
            is_retouched, tool_used = exif_tools.is_ai_generated(metadata)
            if is_retouched:
                return JSONResponse(
                    content={
                        "status": "failure",
                        "prediction": "truquée",
                        "explanation": f"L'image semble avoir été modifiée avec un générateur IA, comme {tool_used}.",
                        "decision": "RMA refusé",
                    }
                )

        # Analyze the image
        image = image_tools.load_image(file_path)
        prediction = model_tools.predict_image(image)

        if prediction == "truquée":
            return JSONResponse(
                content={
                    "status": "failure",
                    "prediction": prediction,
                    "explanation": "L'image contient des modifications détectées.",
                    "decision": "RMA refusé",
                }
            )

        # Encode image and analyze with LLaVA
        image_base64 = image_tools.encode_image_to_base64(file_path)
        ollama_tools.start_ollama_server()

        custom_prompt = f"L'utilisateur a signalé : '{user_description}'. Analysez l'image pour confirmer cette déclaration pour autorisation de retour de marchandise."
        explanation = ollama_tools.analyze_image_with_llava(image_base64, custom_prompt)
        decision = "RMA accepté" if "accepté" in explanation.lower() else "RMA refusé"

        return JSONResponse(
            content={
                "status": "success",
                "prediction": prediction,
                "explanation": explanation,
                "decision": decision,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de l'analyse : {str(e)}"
        )
    finally:
        ollama_tools.stop_ollama_server()


# Face recognition endpoints
class CaptureRequest(BaseModel):
    name: str
    image: str


class RecognizeRequest(BaseModel):
    image: str


@router.post("/capture_face/", tags=["Image"])
async def capture_face_route(request: CaptureRequest, db: Session = Depends(get_db)):
    """Capture and save a face."""
    try:
        message = capture_face(request.name, request.image, db)
        return {"message": message}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/recognize_face/", tags=["Image"])
async def recognize_face_route(
    request: RecognizeRequest, db: Session = Depends(get_db)
):
    """Recognize a face."""
    try:
        match = recognize_face(request.image, db)
        return {"match": match}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Scraping endpoints
db_manager = DatabaseManager()
vinted_scraper = VintedScraper(db_manager)
amazon_scraper = AmazonScraper(db_manager)
query_instance = Query(db_manager)

PLATFORM_SCRAPERS = {
    "vinted": vinted_scraper,
    "amazon": amazon_scraper,
}


@router.get("/platforms", tags=["Scraper"])
async def get_platforms():
    """Get list of available platforms."""
    return {"platforms": list(PLATFORM_SCRAPERS.keys())}

@router.get("/search/{platform}/{query}", tags=["Scraper"])
async def search_products(platform: str, query: str, limit: int = 100):
    """Search for products across specified platform."""
    if platform == "all":
        return await search_all_platforms(query, limit)

    if platform not in PLATFORM_SCRAPERS:
        raise HTTPException(status_code=400, detail=f"Platform '{platform}' not supported.")

    try:
        scraper = PLATFORM_SCRAPERS[platform]
        results = await scraper.search(query, limit)
        return results
    except Exception as e:
        logging.error(f"Error while scraping {platform}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detail/{platform}/{product_url:path}", tags=["Scraper"])
async def get_product_detail(platform: str, product_url: str):
    """Get details for a specific product on a platform."""
    if platform not in PLATFORM_SCRAPERS:
        raise HTTPException(
            status_code=400, detail=f"Platform '{platform}' not supported."
        )

    try:
        scraper = PLATFORM_SCRAPERS[platform]
        results = await scraper.get_detail(product_url)
        return results
    except Exception as e:
        logging.error(f"Error searching on {platform}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def search_all_platforms(query: str, limit: int = 10):
    """Search for products on all available platforms."""
    results = {}
    errors = []

    for platform, scraper in PLATFORM_SCRAPERS.items():
        try:
            platform_results = await scraper.search(query, limit)
            results[platform] = platform_results
        except Exception as e:
            logging.error(f"Error while scraping {platform}: {str(e)}")
            errors.append({"platform": platform, "error": str(e)})
            results[platform] = []

    return [{"results": results, "errors": errors}] if errors else [results]


# Query
@router.get("/products", tags=["Query"], response_model=List[Product])
async def get_all_products_endpoint(source: Optional[str] = None):
    """Récupère tous les produits, éventuellement filtrés par source."""
    try:
        results = await query_instance.get_all_product(source)
        return results
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des produits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/products/categories/{query}", tags=["Query"], response_model=List[Product]
)
async def search_categories_endpoint(query: str, similarity_threshold: int = 80):
    """Recherche floue sur les catégories de produits."""
    try:
        results = await query_instance.search_categories(query, similarity_threshold)
        return results
    except Exception as e:
        logging.error(f"Erreur lors de la recherche des catégories: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@router.get("/products/name/{name}", tags=["Query"], response_model=List[Product])
async def search_products_by_name_endpoint(name: str):
    """Recherche les produits par nom, en utilisant une recherche floue."""
    try:
        results = await query_instance.search_products_by_name(name)
        return results
    except Exception as e:
        logging.error(f"Erreur lors de la recherche des produits par nom: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/brand/{brand}", tags=["Query"], response_model=List[Product])
async def search_products_by_brand_endpoint(brand: str):
    """Recherche les produits par marque."""
    try:
        results = await query_instance.search_products_by_brand(brand)
        return results
    except Exception as e:
        logging.error(f"Erreur lors de la recherche des produits par marque: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/products/condition/{condition}", tags=["Query"], response_model=List[Product]
)
async def search_products_by_condition_endpoint(condition: str):
    """Recherche les produits par état (condition)."""
    try:
        results = await query_instance.search_products_by_condition(condition)
        return results
    except Exception as e:
        logging.error(f"Erreur lors de la recherche des produits par état: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get(
    "/products/description/{keywords}", tags=["Query"], response_model=List[Product]
)
async def search_products_by_description_keywords_endpoint(keywords: str):
    """Recherche les produits par mots-clés dans la description."""
    try:
        results = await query_instance.search_products_by_description_keywords(keywords)
        return results
    except Exception as e:
        logging.error(
            f"Erreur lors de la recherche des produits par mots-clés dans la description: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/products/categories/multiple", tags=["Query"], response_model=List[Product]
)
async def search_products_by_multiple_categories_endpoint(
    categories: List[str] = FastAPIQuery(...),
):
    """Recherche les produits appartenant à plusieurs catégories."""
    try:
        results = await query_instance.search_products_by_multiple_categories(
            categories
        )
        return results
    except Exception as e:
        logging.error(f"Erreur lors de la recherche des produits par catégories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/page", tags=["Query"], response_model=List[Product])
async def get_products_with_pagination_endpoint(page: int = 1, page_size: int = 10):
    """Récupère les produits avec pagination."""
    try:
        results = await query_instance.get_products_with_pagination(page, page_size)
        return results
    except Exception as e:
        logging.error(
            f"Erreur lors de la récupération des produits avec pagination: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


# Include the router in the FastAPI application
app.include_router(router, prefix="/api/v2")


# Modèle pour la requête utilisateur
class SearchRequest(BaseModel):
    query: str


@router.post("/bot/", tags=["bot"])
async def search(request: SearchRequest):
    """🔍 Recherche un produit dans MongoDB et via RAG."""
    chat_instance = await Chatbot.create()
    response = await chat_instance.handle_query(request.query)
    return {"query": request.query, "response": response}


# Include the router in the FastAPI application
app.include_router(router)
