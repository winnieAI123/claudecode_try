"""
Base provider class for all LLM providers.
Defines the common interface that all providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import logging


@dataclass
class StageResult:
    """Result from a single stage of the pipeline."""

    provider_name: str
    success: bool
    findings: list[dict] = field(default_factory=list)  # List of {fact, source_url}
    opinions: dict = field(default_factory=dict)  # {positive, negative, neutral}
    gaps: list[dict] = field(default_factory=list)  # List of {topic, reason, next_step}
    sources: list[str] = field(default_factory=list)  # List of URLs
    error_message: Optional[str] = None
    raw_response: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "provider_name": self.provider_name,
            "success": self.success,
            "findings": self.findings,
            "opinions": self.opinions,
            "gaps": self.gaps,
            "sources": self.sources,
            "error_message": self.error_message,
            "raw_response": self.raw_response
        }


@dataclass
class ResearchPlan:
    """Unified research plan for all stages."""

    topic: str
    objective: str
    time_window_days: int
    region: str
    language: str
    primary_query: str
    expanded_queries: list[str] = field(default_factory=list)
    counter_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "topic": self.topic,
            "objective": self.objective,
            "time_window_days": self.time_window_days,
            "region": self.region,
            "language": self.language,
            "primary_query": self.primary_query,
            "expanded_queries": self.expanded_queries,
            "counter_queries": self.counter_queries
        }


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    All providers (DeepSeek, Gemini, OpenAI) must implement this interface.
    """

    def __init__(self, config: dict, search_client=None):
        """
        Initialize the provider.

        Args:
            config: Configuration dictionary for this provider
            search_client: Optional search client for web searches
        """
        self.config = config
        self.search_client = search_client
        self.logger = logging.getLogger(self.__class__.__name__)
        self._api_key: Optional[str] = None
        self._base_url: Optional[str] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider (load API keys, validate connection).

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def execute_search(
        self,
        research_plan: ResearchPlan,
        search_results: list[dict]
    ) -> StageResult:
        """
        Execute the search and analysis stage.

        Args:
            research_plan: The unified research plan
            search_results: Pre-fetched search results

        Returns:
            StageResult containing findings, opinions, gaps, and sources
        """
        pass

    @abstractmethod
    async def expand_queries(self, topic: str, objective: str) -> list[str]:
        """
        Use the LLM to expand the initial query into related queries.

        Args:
            topic: The main research topic
            objective: What the user wants to know

        Returns:
            List of expanded query strings
        """
        pass

    @abstractmethod
    async def generate_counter_queries(self, topic: str, objective: str) -> list[str]:
        """
        Generate counter-queries to find opposing viewpoints.

        Args:
            topic: The main research topic
            objective: What the user wants to know

        Returns:
            List of counter-query strings
        """
        pass

    def _build_analysis_prompt(
        self,
        research_plan: ResearchPlan,
        search_results: list[dict]
    ) -> str:
        """
        Build the prompt for analyzing search results.

        Args:
            research_plan: The research plan
            search_results: Search results to analyze

        Returns:
            Formatted prompt string
        """
        search_content = "\n\n".join([
            f"来源 {i+1}:\nURL: {r.get('url', 'N/A')}\n标题: {r.get('title', 'N/A')}\n内容: {r.get('snippet', r.get('content', 'N/A'))}"
            for i, r in enumerate(search_results)
        ])

        prompt = f"""你是一位专业的研究分析师。请基于以下搜索结果，对研究主题进行深入分析。

研究主题: {research_plan.topic}
研究目标: {research_plan.objective}
时间窗口: 最近 {research_plan.time_window_days} 天
地区: {research_plan.region}

搜索结果:
{search_content}

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

    def _parse_analysis_response(self, response: str) -> dict:
        """
        Parse the LLM's analysis response into structured data.

        Args:
            response: Raw response from the LLM

        Returns:
            Dictionary with findings, opinions, and gaps
        """
        # Default structure
        result = {
            "findings": [],
            "opinions": {
                "positive": [],
                "negative": [],
                "neutral": []
            },
            "gaps": [],
            "sources": []
        }

        # Extract URLs from the response
        import re
        urls = re.findall(r'https?://[^\s\)]+', response)
        result["sources"] = list(set(urls))

        # Parse sections (basic parsing - can be enhanced)
        lines = response.split('\n')
        current_section = None
        current_subsection = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if '关键事实' in line or 'Key Facts' in line:
                current_section = 'findings'
                current_subsection = None
            elif '观点' in line or 'Opinions' in line:
                current_section = 'opinions'
            elif '待验证' in line or '缺口' in line or 'Gaps' in line:
                current_section = 'gaps'
                current_subsection = None
            elif current_section == 'opinions':
                if '正方' in line or 'positive' in line.lower():
                    current_subsection = 'positive'
                elif '反方' in line or 'negative' in line.lower():
                    current_subsection = 'negative'
                elif '中性' in line or 'neutral' in line.lower():
                    current_subsection = 'neutral'

            # Extract content (lines starting with - or ■)
            if line.startswith(('-', '■', '•', '*')):
                content = line.lstrip('-■•* ').strip()

                # Extract URL if present
                url_match = re.search(r'\((https?://[^\)]+)\)', content)
                url = url_match.group(1) if url_match else None
                fact = re.sub(r'\(https?://[^\)]+\)', '', content).strip()

                if current_section == 'findings':
                    result["findings"].append({
                        "fact": fact,
                        "source_url": url
                    })
                elif current_section == 'opinions' and current_subsection:
                    result["opinions"][current_subsection].append({
                        "opinion": fact,
                        "source_url": url
                    })
                elif current_section == 'gaps':
                    result["gaps"].append({
                        "topic": fact,
                        "reason": "",
                        "next_step": ""
                    })

        return result
