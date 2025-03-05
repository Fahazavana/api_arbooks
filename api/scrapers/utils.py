from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Union


@dataclass
class Product:
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
    description: Optional[str] = None
    brand: Optional[str] = None
    colors: Optional[Union[str, List[str], Dict[str, str]]] = None
    views: Optional[int] = None
    interested: Optional[int] = None
    uploaded: Optional[str] = None
    payment_methods: Optional[str] = None
    interested: Optional[int] = None
    owner_name: Optional[str] = None
    owner_profile_url: Optional[str] = None
    feature_table: Optional[Dict[str, str]] = None
    feature_bullet: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def is_valid(self) -> bool:
        return all(
            [
                self.url is not None,
                self.product_id is not None,
                self.name is not None,
                self.price is not None,
            ]
        )

    def __repr__(self):
        return f"🛒 {self.source}-{self.product_id}-{self.price}-{self.name}"
