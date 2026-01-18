"""
Gemini provider implementation.
Handles Stage 2 of the tri-model deep search pipeline.
"""

import os
import asyncio
import logging
from typing import Optional

import httpx

from .base import BaseProvider, StageResult, ResearchPlan


class GeminiProvider(BaseProvider):
    """
    Google Gemini LLM provider for Stage 2 of the pipeline.
    Provides independent analysis with fresh search results.
    """

    def __init__(self, config: dict, search_client=None):
        """
        Initialize Gemini provider.

        Args:
            config: Configuration dictionary from config.yaml
            search_client: Search client for web searches
        """
        super().__init__(config, search_client)
        self._client: Optional[httpx.AsyncClient] = None
        self._model = config.get("model", "gemini-pro")

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "Gemini"

    def initialize(self) -> bool:
        """
        Initialize the Gemini provider.
        Load API key from environment and validate.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Get API key from environment
            api_key_env = self.config.get("api_key_env", "GEMINI_API_KEY")
            self._api_key = os.environ.get(api_key_env)

            if not self._api_key:
                self.logger.error(f"API key not found in environment variable: {api_key_env}")
                return False

            # Get base URL (Gemini uses a different URL structure)
            base_url_env = self.config.get("base_url_env", "GEMINI_BASE_URL")
            self._base_url = os.environ.get(base_url_env) or \
                "https://generativelanguage.googleapis.com/v1beta"

            # Initialize async HTTP client
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.config.get("timeout", 60.0)
            )

            self.logger.info("Gemini provider initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini provider: {e}")
            return False

    async def _call_api(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Make an API call to Gemini.

        Args:
            prompt: The prompt text
            temperature: Sampling temperature

        Returns:
            Response content as string
        """
        if not self._client:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        # Gemini API endpoint format
        endpoint = f"/models/{self._model}:generateContent?key={self._api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 4096,
                "topP": 0.95,
                "topK": 40
            }
        }

        response = await self._client.post(endpoint, json=payload)
        response.raise_for_status()

        data = response.json()

        # Extract text from Gemini response format
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                return "".join(part.get("text", "") for part in parts)

        raise ValueError("Invalid response format from Gemini API")

    async def expand_queries(self, topic: str, objective: str) -> list[str]:
        """
        Use Gemini to expand the initial query into related queries.

        Args:
            topic: The main research topic
            objective: What the user wants to know

        Returns:
            List of expanded query strings (3-8 queries)
        """
        prompt = f"""你是一位专业的研究分析师。请基于以下研究主题和目标，生成3-8个相关的搜索查询词，用于全面了解该主题。

研究主题: {topic}
研究目标: {objective}

请生成搜索查询词，包括:
- 同义词和相关术语
- 相关公司/产品/人物名称
- 具体的技术术语
- 行业相关关键词

请只输出查询词列表，每行一个查询词，不要有编号或其他格式。"""

        try:
            response = await self._call_api(prompt)
            queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
            return queries[:8] if len(queries) > 8 else queries
        except Exception as e:
            self.logger.error(f"Failed to expand queries: {e}")
            return []

    async def generate_counter_queries(self, topic: str, objective: str) -> list[str]:
        """
        Generate counter-queries to find opposing viewpoints.

        Args:
            topic: The main research topic
            objective: What the user wants to know

        Returns:
            List of counter-query strings (2-4 queries)
        """
        prompt = f"""你是一位专业的研究分析师。请基于以下研究主题，生成2-4个"反向"搜索查询词，用于找到反对意见、批评观点或潜在风险。

研究主题: {topic}
研究目标: {objective}

请生成能够找到以下内容的查询词:
- 批评和质疑声音
- 潜在风险和问题
- 竞争对手的观点
- 失败案例或负面新闻

请只输出查询词列表，每行一个查询词，不要有编号或其他格式。"""

        try:
            response = await self._call_api(prompt)
            queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
            return queries[:4] if len(queries) > 4 else queries
        except Exception as e:
            self.logger.error(f"Failed to generate counter queries: {e}")
            return []

    async def execute_search(
        self,
        research_plan: ResearchPlan,
        search_results: list[dict]
    ) -> StageResult:
        """
        Execute Stage 2: Gemini analysis of search results.

        Args:
            research_plan: The unified research plan
            search_results: Pre-fetched search results (fresh, not from Stage 1)

        Returns:
            StageResult containing findings, opinions, gaps, and sources
        """
        try:
            if not search_results:
                return StageResult(
                    provider_name=self.name,
                    success=False,
                    error_message="No search results provided"
                )

            # Build analysis prompt
            prompt = self._build_analysis_prompt(research_plan, search_results)

            # Call Gemini API
            response = await self._call_api(prompt, temperature=0.3)

            # Parse the response
            parsed = self._parse_analysis_response(response)

            return StageResult(
                provider_name=self.name,
                success=True,
                findings=parsed["findings"],
                opinions=parsed["opinions"],
                gaps=parsed["gaps"],
                sources=parsed["sources"],
                raw_response=response
            )

        except httpx.TimeoutException:
            return StageResult(
                provider_name=self.name,
                success=False,
                error_message="API call timed out"
            )
        except httpx.HTTPStatusError as e:
            return StageResult(
                provider_name=self.name,
                success=False,
                error_message=f"API error: {e.response.status_code}"
            )
        except Exception as e:
            self.logger.error(f"Gemini execution failed: {e}")
            return StageResult(
                provider_name=self.name,
                success=False,
                error_message=str(e)
            )

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
