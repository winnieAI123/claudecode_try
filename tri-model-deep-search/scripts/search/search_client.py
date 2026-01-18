"""
Search client for web searches.
Supports multiple search providers: Serper, SerpAPI, Bing, DuckDuckGo.
"""

import os
import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import httpx


class SearchResult:
    """Represents a single search result."""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: Optional[str] = None,
        date: Optional[str] = None
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.date = date

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "date": self.date
        }


class BaseSearchProvider(ABC):
    """Abstract base class for search providers."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Execute a search query."""
        pass


class SerperSearchProvider(BaseSearchProvider):
    """Serper.dev search provider."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._api_key = os.environ.get(config.get("api_key_env", "SEARCH_API_KEY"))
        self._base_url = "https://google.serper.dev"

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Execute a search using Serper API."""
        if not self._api_key:
            self.logger.error("Serper API key not found")
            return []

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self._base_url}/search",
                    headers={
                        "X-API-KEY": self._api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "q": query,
                        "num": num_results
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("organic", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source=item.get("source"),
                        date=item.get("date")
                    ))

                # Also include news if available
                for item in data.get("news", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source=item.get("source"),
                        date=item.get("date")
                    ))

                return results[:num_results]

            except Exception as e:
                self.logger.error(f"Serper search failed: {e}")
                return []


class SerpAPISearchProvider(BaseSearchProvider):
    """SerpAPI search provider."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._api_key = os.environ.get(config.get("api_key_env", "SEARCH_API_KEY"))
        self._base_url = "https://serpapi.com"

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Execute a search using SerpAPI."""
        if not self._api_key:
            self.logger.error("SerpAPI key not found")
            return []

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self._base_url}/search",
                    params={
                        "q": query,
                        "api_key": self._api_key,
                        "num": num_results
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("organic_results", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source=item.get("source"),
                        date=item.get("date")
                    ))

                return results[:num_results]

            except Exception as e:
                self.logger.error(f"SerpAPI search failed: {e}")
                return []


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """DuckDuckGo search provider (no API key required)."""

    def __init__(self, config: dict):
        super().__init__(config)

    async def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Execute a search using DuckDuckGo."""
        try:
            # Using duckduckgo-search library
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source=None,
                        date=None
                    ))
            return results

        except ImportError:
            self.logger.error("duckduckgo-search not installed")
            return []
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
            return []


class SearchClient:
    """
    Unified search client that manages search providers and caching.
    """

    def __init__(self, config: dict, cache_config: Optional[dict] = None):
        """
        Initialize the search client.

        Args:
            config: Search configuration from config.yaml
            cache_config: Optional cache configuration
        """
        self.config = config
        self.cache_config = cache_config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize the appropriate provider
        provider_name = config.get("provider", "serper").lower()
        if provider_name == "serper":
            self._provider = SerperSearchProvider(config)
        elif provider_name == "serpapi":
            self._provider = SerpAPISearchProvider(config)
        elif provider_name == "duckduckgo":
            self._provider = DuckDuckGoSearchProvider(config)
        else:
            self.logger.warning(f"Unknown provider {provider_name}, defaulting to DuckDuckGo")
            self._provider = DuckDuckGoSearchProvider(config)

        # Initialize cache
        self._cache_enabled = self.cache_config.get("enabled", False)
        self._cache_dir = Path(self.cache_config.get("cache_dir", "./cache"))
        self._cache_ttl = timedelta(hours=self.cache_config.get("ttl_hours", 24))

        if self._cache_enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _get_cached_result(self, query: str) -> Optional[list[dict]]:
        """Get cached search results if available and not expired."""
        if not self._cache_enabled:
            return None

        cache_key = self._get_cache_key(query)
        cache_file = self._cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - cached_time > self._cache_ttl:
                # Cache expired
                cache_file.unlink()
                return None

            return data["results"]

        except Exception as e:
            self.logger.warning(f"Failed to read cache: {e}")
            return None

    def _save_to_cache(self, query: str, results: list[dict]):
        """Save search results to cache."""
        if not self._cache_enabled:
            return

        cache_key = self._get_cache_key(query)
        cache_file = self._cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "results": results
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")

    async def search(self, query: str, num_results: Optional[int] = None) -> list[dict]:
        """
        Execute a search query.

        Args:
            query: The search query
            num_results: Number of results to return (default from config)

        Returns:
            List of search result dictionaries
        """
        if num_results is None:
            num_results = self.config.get("results_per_query", 10)

        # Check cache first
        cached = self._get_cached_result(query)
        if cached is not None:
            self.logger.debug(f"Cache hit for query: {query}")
            return cached[:num_results]

        # Execute search
        results = await self._provider.search(query, num_results)
        result_dicts = [r.to_dict() for r in results]

        # Save to cache
        self._save_to_cache(query, result_dicts)

        return result_dicts

    async def search_multiple(
        self,
        queries: list[str],
        num_results_per_query: Optional[int] = None,
        deduplicate: bool = True
    ) -> list[dict]:
        """
        Execute multiple search queries in parallel.

        Args:
            queries: List of search queries
            num_results_per_query: Results per query
            deduplicate: Whether to remove duplicate URLs

        Returns:
            Combined list of search result dictionaries
        """
        # Execute all queries in parallel
        tasks = [self.search(q, num_results_per_query) for q in queries]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_results = []
        seen_urls = set()

        for results in results_lists:
            if isinstance(results, Exception):
                self.logger.error(f"Search failed: {results}")
                continue

            for result in results:
                url = result.get("url", "")
                if deduplicate:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(result)
                else:
                    all_results.append(result)

        return all_results
