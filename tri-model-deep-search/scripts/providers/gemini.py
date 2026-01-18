"""
Gemini provider implementation with native Google Search grounding.
Handles Stage 2 of the tri-model deep search pipeline.
"""

import os
import asyncio
import logging
import re
from typing import Optional

import httpx

from .base import BaseProvider, StageResult, ResearchPlan


class GeminiProvider(BaseProvider):
    """
    Google Gemini LLM provider for Stage 2 of the pipeline.
    Uses native Google Search grounding for real-time web search.
    """

    def __init__(self, config: dict, search_client=None):
        """
        Initialize Gemini provider.

        Args:
            config: Configuration dictionary from config.yaml
            search_client: Search client (not used - Gemini has native search)
        """
        super().__init__(config, search_client)
        self._client: Optional[httpx.AsyncClient] = None
        self._model = config.get("model", "gemini-2.0-flash")
        self._enable_grounding = config.get("enable_grounding", True)

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

            # Get base URL
            self._base_url = self.config.get(
                "default_base_url",
                "https://generativelanguage.googleapis.com/v1beta"
            )

            # Initialize async HTTP client
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.config.get("timeout", 90.0)  # Longer timeout for grounding
            )

            grounding_status = "enabled" if self._enable_grounding else "disabled"
            self.logger.info(f"Gemini provider initialized (model: {self._model}, grounding: {grounding_status})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini provider: {e}")
            return False

    async def _call_api_with_grounding(self, prompt: str, temperature: float = 0.7) -> tuple[str, list[dict]]:
        """
        Make an API call to Gemini with Google Search grounding.

        Args:
            prompt: The prompt text
            temperature: Sampling temperature

        Returns:
            Tuple of (response content, grounding sources)
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
                "maxOutputTokens": 8192,
                "topP": 0.95,
                "topK": 40
            }
        }

        # Add grounding tool if enabled
        if self._enable_grounding:
            # For Gemini 2.0 models
            if "2.0" in self._model or "gemini-2" in self._model:
                payload["tools"] = [{"google_search": {}}]
            else:
                # For Gemini 1.5 models
                payload["tools"] = [{
                    "google_search_retrieval": {
                        "dynamic_retrieval_config": {
                            "mode": "MODE_DYNAMIC",
                            "dynamic_threshold": 0.3
                        }
                    }
                }]

        response = await self._client.post(endpoint, json=payload)
        response.raise_for_status()

        data = response.json()
        grounding_sources = []

        # Extract text from Gemini response format
        response_text = ""
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                response_text = "".join(part.get("text", "") for part in parts)

            # Extract grounding metadata (sources)
            if "groundingMetadata" in candidate:
                grounding_meta = candidate["groundingMetadata"]

                # Extract from groundingChunks
                if "groundingChunks" in grounding_meta:
                    for chunk in grounding_meta["groundingChunks"]:
                        if "web" in chunk:
                            web = chunk["web"]
                            grounding_sources.append({
                                "url": web.get("uri", ""),
                                "title": web.get("title", "")
                            })

                # Extract from webSearchQueries (for reference)
                if "webSearchQueries" in grounding_meta:
                    self.logger.debug(f"Search queries used: {grounding_meta['webSearchQueries']}")

                # Extract from searchEntryPoint if available
                if "searchEntryPoint" in grounding_meta:
                    entry = grounding_meta["searchEntryPoint"]
                    if "renderedContent" in entry:
                        # This contains HTML with links - could parse if needed
                        pass

        if not response_text:
            raise ValueError("Invalid response format from Gemini API")

        return response_text, grounding_sources

    async def _call_api(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Make an API call to Gemini (without grounding, for simple queries).

        Args:
            prompt: The prompt text
            temperature: Sampling temperature

        Returns:
            Response content as string
        """
        if not self._client:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

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
        Execute Stage 2: Gemini analysis with native Google Search grounding.

        Note: Unlike other providers, Gemini uses its native search capability.
        The search_results parameter is ignored when grounding is enabled.

        Args:
            research_plan: The unified research plan
            search_results: Pre-fetched search results (ignored when grounding enabled)

        Returns:
            StageResult containing findings, opinions, gaps, and sources
        """
        try:
            # Build the research prompt
            prompt = self._build_grounded_research_prompt(research_plan)

            # Call Gemini API with grounding
            response_text, grounding_sources = await self._call_api_with_grounding(
                prompt,
                temperature=0.3
            )

            # Parse the response
            parsed = self._parse_analysis_response(response_text)

            # Add grounding sources to the parsed sources
            source_urls = parsed["sources"]
            for gs in grounding_sources:
                url = gs.get("url", "")
                if url and url not in source_urls:
                    source_urls.append(url)

            # Also add sources to findings if they don't have URLs
            for i, finding in enumerate(parsed["findings"]):
                if not finding.get("source_url") and i < len(grounding_sources):
                    finding["source_url"] = grounding_sources[i].get("url", "")

            return StageResult(
                provider_name=self.name,
                success=True,
                findings=parsed["findings"],
                opinions=parsed["opinions"],
                gaps=parsed["gaps"],
                sources=source_urls,
                raw_response=response_text
            )

        except httpx.TimeoutException:
            return StageResult(
                provider_name=self.name,
                success=False,
                error_message="API call timed out"
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", "")
            except:
                pass
            return StageResult(
                provider_name=self.name,
                success=False,
                error_message=f"API error: {e.response.status_code} - {error_detail}"
            )
        except Exception as e:
            self.logger.error(f"Gemini execution failed: {e}")
            return StageResult(
                provider_name=self.name,
                success=False,
                error_message=str(e)
            )

    def _build_grounded_research_prompt(self, research_plan: ResearchPlan) -> str:
        """
        Build a prompt optimized for grounded search.

        Args:
            research_plan: The research plan

        Returns:
            Formatted prompt string
        """
        prompt = f"""你是一位专业的研究分析师。请使用 Google 搜索来研究以下主题，并提供基于最新信息的分析报告。

研究主题: {research_plan.topic}
研究目标: {research_plan.objective}
时间窗口: 最近 {research_plan.time_window_days} 天的信息优先
地区: {research_plan.region}

请搜索并分析：
1. 主要查询: {research_plan.primary_query}
2. 相关查询: {', '.join(research_plan.expanded_queries[:5]) if research_plan.expanded_queries else '无'}
3. 反向查询（寻找反对意见）: {', '.join(research_plan.counter_queries[:3]) if research_plan.counter_queries else '无'}

请按照以下结构输出你的分析:

1. 关键事实 (每条必须包含来源URL):
   - 列出最重要的事实发现
   - 每条格式: 主体：事实陈述（来源URL）

2. 观点汇总:
   - 正方观点: 支持性观点及依据（URL）
   - 反方观点: 反对性观点及依据（URL）
   - 中性观点: 中立性观点及依据（URL）

3. 待验证/信息缺口:
   - 列出无法确认或需要进一步验证的信息
   - 每条格式: 主体：待验证点（原因/建议下一步）

请确保:
- 所有事实陈述都有URL支持
- 区分事实与观点
- 标注任何不确定的信息为"待验证"
- 使用简洁的中文，采用买方研究简报风格
"""
        return prompt

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
