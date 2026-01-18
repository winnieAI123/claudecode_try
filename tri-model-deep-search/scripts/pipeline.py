"""
Main pipeline orchestration for tri-model deep search.
Coordinates the 3-stage search pipeline: DeepSeek → Gemini → ChatGPT.
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

from .providers.base import ResearchPlan, StageResult
from .providers.deepseek import DeepSeekProvider
from .providers.gemini import GeminiProvider
from .providers.openai_provider import OpenAIProvider
from .search.search_client import SearchClient
from .formatter.buyside_cn import BuySideCNFormatter, ConsolidatedResult


class TriModelSearchPipeline:
    """
    Main orchestration class for the tri-model deep search pipeline.

    Execution order (MUST be strict):
    1. Stage 1: DeepSeek + Search
    2. Stage 2: Gemini + Search
    3. Stage 3: ChatGPT + Search
    4. Cross-stage consolidation
    5. Formatted output
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to config.yaml file
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        self.config = self._load_config(config_path)

        # Initialize components
        self._init_logging()
        self._init_providers()
        self._init_search_client()
        self._init_formatter()

    def _load_config(self, config_path: str | Path) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _init_logging(self):
        """Initialize logging based on configuration."""
        log_config = self.config.get("logging", {})

        if not log_config.get("enabled", True):
            return

        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(level=log_level, format=log_format)

        # Create log directory if needed
        log_dir = Path(log_config.get("log_dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

    def _init_providers(self):
        """Initialize all LLM providers."""
        creds = self.config.get("credentials", {})

        self.providers = {
            "DeepSeek": DeepSeekProvider(creds.get("deepseek", {})),
            "Gemini": GeminiProvider(creds.get("gemini", {})),
            "ChatGPT": OpenAIProvider(creds.get("openai", {}))
        }

        # Initialize each provider
        for name, provider in self.providers.items():
            if not provider.initialize():
                self.logger.warning(f"Failed to initialize {name} provider")

    def _init_search_client(self):
        """Initialize the search client."""
        search_config = self.config.get("search", {})
        cache_config = self.config.get("cache", {})
        self.search_client = SearchClient(search_config, cache_config)

    def _init_formatter(self):
        """Initialize the output formatter."""
        tri_search_config = self.config.get("tri_search", {})
        self.formatter = BuySideCNFormatter(tri_search_config)

    def _detect_language(self, text: str) -> str:
        """Detect language of the input text."""
        # Simple heuristic: count Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return "zh"
        return "en"

    async def create_research_plan(
        self,
        topic: str,
        objective: Optional[str] = None
    ) -> ResearchPlan:
        """
        Create a research plan from user input.

        Args:
            topic: The research topic
            objective: Optional specific objective (defaults to general research)

        Returns:
            ResearchPlan object
        """
        tri_config = self.config.get("tri_search", {})

        # Detect language
        query_lang = tri_config.get("query_language", "auto")
        if query_lang == "auto":
            query_lang = self._detect_language(topic)

        # Use first available provider for query expansion
        expanded_queries = []
        counter_queries = []

        for provider in self.providers.values():
            try:
                expanded_queries = await provider.expand_queries(
                    topic,
                    objective or f"全面了解{topic}的最新情况"
                )
                counter_queries = await provider.generate_counter_queries(
                    topic,
                    objective or f"全面了解{topic}的最新情况"
                )
                if expanded_queries:
                    break
            except Exception as e:
                self.logger.warning(f"Query expansion failed: {e}")
                continue

        return ResearchPlan(
            topic=topic,
            objective=objective or f"全面了解{topic}的最新情况",
            time_window_days=tri_config.get("time_window_days", 30),
            region=tri_config.get("region", "US"),
            language=query_lang,
            primary_query=topic,
            expanded_queries=expanded_queries,
            counter_queries=counter_queries
        )

    async def _execute_stage(
        self,
        provider_name: str,
        research_plan: ResearchPlan
    ) -> StageResult:
        """
        Execute a single stage of the pipeline.

        Args:
            provider_name: Name of the provider (DeepSeek, Gemini, ChatGPT)
            research_plan: The research plan

        Returns:
            StageResult from the stage
        """
        provider = self.providers.get(provider_name)
        if not provider:
            return StageResult(
                provider_name=provider_name,
                success=False,
                error_message=f"Provider {provider_name} not found"
            )

        try:
            # Perform fresh search for this stage
            all_queries = (
                [research_plan.primary_query] +
                research_plan.expanded_queries +
                research_plan.counter_queries
            )

            tri_config = self.config.get("tri_search", {})
            max_sources = tri_config.get("max_sources_per_stage", 8)

            search_results = await self.search_client.search_multiple(
                all_queries,
                num_results_per_query=max_sources // len(all_queries) + 1,
                deduplicate=True
            )

            # Execute the provider's analysis
            result = await provider.execute_search(research_plan, search_results)

            # Save raw response if configured
            self._save_raw_response(provider_name, research_plan.topic, result)

            return result

        except Exception as e:
            self.logger.error(f"Stage {provider_name} failed: {e}")
            return StageResult(
                provider_name=provider_name,
                success=False,
                error_message=str(e)
            )

    def _save_raw_response(
        self,
        provider_name: str,
        topic: str,
        result: StageResult
    ):
        """Save raw response to logs if configured."""
        log_config = self.config.get("logging", {})
        if not log_config.get("save_raw_responses", False):
            return

        log_dir = Path(log_config.get("log_dir", "./logs"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]

        filename = f"{timestamp}_{provider_name}_{topic_hash}.json"
        filepath = log_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save raw response: {e}")

    def _consolidate_results(
        self,
        research_plan: ResearchPlan,
        stage_results: dict[str, StageResult]
    ) -> ConsolidatedResult:
        """
        Consolidate results from all stages.

        Args:
            research_plan: The research plan
            stage_results: Dict of provider name to StageResult

        Returns:
            ConsolidatedResult with consensus, divergences, etc.
        """
        # Collect all findings and opinions
        all_findings = {}
        all_opinions = {}

        for provider, result in stage_results.items():
            if result.success:
                all_findings[provider] = result.findings
                all_opinions[provider] = result.opinions

        # Find consensus (facts mentioned by multiple providers)
        consensus = []
        finding_counts = {}

        for provider, findings in all_findings.items():
            for finding in findings:
                fact = finding.get("fact", "").lower()
                if fact:
                    if fact not in finding_counts:
                        finding_counts[fact] = {
                            "count": 0,
                            "providers": [],
                            "url": finding.get("source_url")
                        }
                    finding_counts[fact]["count"] += 1
                    finding_counts[fact]["providers"].append(provider)

        for fact, data in finding_counts.items():
            if data["count"] >= 2:  # At least 2 providers agree
                consensus.append({
                    "topic": fact,
                    "evidence_url": data["url"]
                })

        # Find divergences (different conclusions on same topic)
        divergences = []
        # This is a simplified divergence detection
        # A more sophisticated version would use semantic similarity

        # Collect high and low confidence conclusions
        high_confidence = []
        low_confidence = []

        for item in consensus[:5]:  # Top consensus items are high confidence
            high_confidence.append({
                "conclusion": item["topic"],
                "url": item.get("evidence_url", "")
            })

        # Collect gaps as low confidence
        for provider, result in stage_results.items():
            if result.success:
                for gap in result.gaps[:3]:
                    low_confidence.append({
                        "conclusion": gap.get("topic", ""),
                        "reason": f"{provider}标记为待验证"
                    })

        # Generate next steps
        next_steps = []
        for provider, result in stage_results.items():
            if result.success:
                for gap in result.gaps[:2]:
                    next_step = gap.get("next_step") or gap.get("reason")
                    if next_step:
                        next_steps.append(next_step)

        if not next_steps:
            next_steps = ["持续关注该领域最新动态", "验证关键数据源"]

        return ConsolidatedResult(
            research_plan=research_plan,
            stage_results=stage_results,
            consensus=consensus,
            divergences=divergences,
            high_confidence=high_confidence,
            low_confidence=low_confidence,
            next_steps=list(set(next_steps))[:5]
        )

    async def execute(
        self,
        topic: str,
        objective: Optional[str] = None
    ) -> str:
        """
        Execute the full tri-model search pipeline.

        CRITICAL: All three stages MUST be executed in strict order.
        Even if a stage fails, remaining stages MUST still run.

        Args:
            topic: The research topic
            objective: Optional specific objective

        Returns:
            Formatted output string
        """
        self.logger.info(f"Starting tri-model search for: {topic}")

        try:
            # Create research plan
            research_plan = await self.create_research_plan(topic, objective)
            self.logger.info(f"Research plan created with {len(research_plan.expanded_queries)} expanded queries")

            # Execute all three stages in strict order
            stage_results = {}

            # Stage 1: DeepSeek
            self.logger.info("Executing Stage 1: DeepSeek")
            stage_results["DeepSeek"] = await self._execute_stage("DeepSeek", research_plan)

            # Stage 2: Gemini (MUST run regardless of Stage 1 result)
            self.logger.info("Executing Stage 2: Gemini")
            stage_results["Gemini"] = await self._execute_stage("Gemini", research_plan)

            # Stage 3: ChatGPT (MUST run regardless of previous results)
            self.logger.info("Executing Stage 3: ChatGPT")
            stage_results["ChatGPT"] = await self._execute_stage("ChatGPT", research_plan)

            # Log stage results
            for provider, result in stage_results.items():
                status = "SUCCESS" if result.success else "FAILED"
                self.logger.info(f"{provider}: {status}")

            # Consolidate results
            consolidated = self._consolidate_results(research_plan, stage_results)

            # Format output
            output = self.formatter.format(consolidated)

            self.logger.info("Pipeline completed successfully")
            return output

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return self.formatter.format_error(
                research_plan if 'research_plan' in locals() else ResearchPlan(
                    topic=topic,
                    objective=objective or "",
                    time_window_days=30,
                    region="US",
                    language="zh",
                    primary_query=topic
                ),
                str(e),
                stage_results if 'stage_results' in locals() else None
            )

    async def close(self):
        """Clean up resources."""
        for provider in self.providers.values():
            await provider.close()


async def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tri-Model Deep Search Pipeline"
    )
    parser.add_argument(
        "topic",
        help="Research topic to search"
    )
    parser.add_argument(
        "-o", "--objective",
        help="Specific research objective",
        default=None
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config.yaml",
        default=None
    )

    args = parser.parse_args()

    pipeline = TriModelSearchPipeline(args.config)
    try:
        result = await pipeline.execute(args.topic, args.objective)
        print(result)
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
