from beanie import Document
from typing import Optional, List, Dict, Union


class Product_scraping(Document):
    source: str
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
    uploaded: Optional[Union[str, Dict[str, str]]] = None
    payment_methods: Optional[str] = None
    owner_name: Optional[str] = None
    owner_profile_url: Optional[str] = None
    feature_table: Optional[Dict[str, str]] = None
    feature_bullet: Optional[List[str]] = None

    class Settings:
        collection = "products_scraping"
