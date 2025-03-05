from typing import List, Dict, Any
from urllib.parse import urlencode


class BaseScraper:
    """Base class for scrapers."""

    def parse_item(self, item) -> Dict[str, Any]:
        """Base method to parse an item from a page."""
        raise NotImplementedError(
            "Each scraper must implement its own parse_item method"
        )

    def _encode_params(self, params: Dict[str, str]) -> str:
        """Helper method to encode URL parameters."""
        return urlencode(params)

    async def get_page_content(
        self, url: str, limit: int = 60000
    ) -> List[Dict[str, Any]]:
        """Base method to get the content of a page."""
        raise NotImplementedError(
            "Each scraper must implement its own get_page_content method"
        )

    async def search(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Base search method to be implemented by each scraper"""
        raise NotImplementedError("Each scraper must implement its own search method")

    async def get_detail(self, product_url: Dict) -> List[Dict[str, Any]]:
        """Base search method to be implemented by each scraper"""
        raise NotImplementedError("Each scraper must implement its own search method")
