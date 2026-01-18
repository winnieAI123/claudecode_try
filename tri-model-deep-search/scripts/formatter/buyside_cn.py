"""
Buy-side brief formatter for Chinese output.
Formats the consolidated results according to the specified format rules.
"""

from dataclasses import dataclass
from typing import Optional
import logging

from ..providers.base import StageResult, ResearchPlan


@dataclass
class ConsolidatedResult:
    """Consolidated results from all three stages."""

    research_plan: ResearchPlan
    stage_results: dict[str, StageResult]  # {provider_name: StageResult}
    consensus: list[dict]  # [{topic, evidence_url}]
    divergences: list[dict]  # [{topic, positions: {provider: position}}]
    high_confidence: list[dict]  # [{conclusion, url}]
    low_confidence: list[dict]  # [{conclusion, reason}]
    next_steps: list[str]


class BuySideCNFormatter:
    """
    Formats research results in Chinese buy-side brief style.

    Output rules:
    1. Top-level sections: plain text + colon (e.g., 数据中心产业链信息：)
    2. Subcategories: use 【】 as semantic anchors
    3. Information lines: 主体 + 冒号 + 事实陈述
    4. Multiple items: use ■ as bullets
    5. No news headline style, no narrative, no journalist names
    6. No opinions except in 观点 section
    7. Output looks like a buy-side daily brief, not a blog post
    """

    def __init__(self, config: dict):
        """
        Initialize the formatter.

        Args:
            config: Formatter configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_bullets = config.get("max_bullets_per_section", 8)

    def format(self, result: ConsolidatedResult) -> str:
        """
        Format the consolidated result into a buy-side brief.

        Args:
            result: ConsolidatedResult containing all stage results

        Returns:
            Formatted string output
        """
        sections = []

        # 1. Task Overview
        sections.append(self._format_overview(result))

        # 2. DeepSeek Conclusions
        if "DeepSeek" in result.stage_results:
            sections.append(self._format_stage_result(
                "DeepSeek",
                result.stage_results["DeepSeek"]
            ))

        # 3. Gemini Conclusions
        if "Gemini" in result.stage_results:
            sections.append(self._format_stage_result(
                "Gemini",
                result.stage_results["Gemini"]
            ))

        # 4. ChatGPT Conclusions
        if "ChatGPT" in result.stage_results:
            sections.append(self._format_stage_result(
                "ChatGPT",
                result.stage_results["ChatGPT"]
            ))

        # 5. Cross-stage consolidation
        sections.append(self._format_consolidation(result))

        return "\n\n".join(sections)

    def _format_overview(self, result: ConsolidatedResult) -> str:
        """Format the search task overview section."""
        plan = result.research_plan
        stages = result.stage_results

        # Determine stage status
        stage_status = []
        for provider in ["DeepSeek", "Gemini", "ChatGPT"]:
            if provider in stages:
                status = "成功" if stages[provider].success else "失败"
            else:
                status = "未执行"
            stage_status.append(f"■ Stage{['DeepSeek', 'Gemini', 'ChatGPT'].index(provider) + 1}（{provider}）：{status}")

        lines = [
            "搜索任务概述：",
            f"【主题】：{plan.topic}",
            f"【目标】：{plan.objective}",
            f"【时间窗】：最近 {plan.time_window_days} 天",
            f"【地区/语言】：{plan.region} / {plan.language}",
            "【执行状态】：",
        ]
        lines.extend(stage_status)

        # Add error details if any stage failed
        for provider, stage_result in stages.items():
            if not stage_result.success and stage_result.error_message:
                lines.append(f"【失败原因】：{provider}：{stage_result.error_message}")
                lines.append("【补救建议】：建议重试或检查API配置")

        return "\n".join(lines)

    def _format_stage_result(self, provider_name: str, result: StageResult) -> str:
        """Format a single stage's results."""
        lines = [f"{provider_name} 结论："]

        if not result.success:
            lines.append(f"【执行状态】：失败")
            lines.append(f"【失败原因】：{result.error_message or '未知错误'}")
            return "\n".join(lines)

        # Key facts
        lines.append("【关键事实】：")
        if result.findings:
            for finding in result.findings[:self.max_bullets]:
                fact = finding.get("fact", "")
                url = finding.get("source_url", "")
                if url:
                    lines.append(f"■ {fact}（{url}）")
                else:
                    lines.append(f"■ {fact}")
        else:
            lines.append("■ 无关键事实发现")

        # Opinions
        lines.append("【观点】：")
        opinions = result.opinions

        # Positive opinions
        positive = opinions.get("positive", [])
        if positive:
            for op in positive[:self.max_bullets // 3]:
                opinion = op.get("opinion", "")
                url = op.get("source_url", "")
                if url:
                    lines.append(f"■ 正方：{opinion}（{url}）")
                else:
                    lines.append(f"■ 正方：{opinion}")
        else:
            lines.append("■ 正方：无明确正面观点")

        # Negative opinions
        negative = opinions.get("negative", [])
        if negative:
            for op in negative[:self.max_bullets // 3]:
                opinion = op.get("opinion", "")
                url = op.get("source_url", "")
                if url:
                    lines.append(f"■ 反方：{opinion}（{url}）")
                else:
                    lines.append(f"■ 反方：{opinion}")
        else:
            lines.append("■ 反方：无明确反面观点")

        # Neutral opinions
        neutral = opinions.get("neutral", [])
        if neutral:
            for op in neutral[:self.max_bullets // 3]:
                opinion = op.get("opinion", "")
                url = op.get("source_url", "")
                if url:
                    lines.append(f"■ 中性：{opinion}（{url}）")
                else:
                    lines.append(f"■ 中性：{opinion}")
        else:
            lines.append("■ 中性：无明确中性观点")

        # Gaps
        lines.append("【待验证/缺口】：")
        if result.gaps:
            for gap in result.gaps[:self.max_bullets]:
                topic = gap.get("topic", "")
                reason = gap.get("reason", "")
                next_step = gap.get("next_step", "")
                detail = f"（{reason}）" if reason else ""
                if next_step:
                    detail += f"；建议：{next_step}"
                lines.append(f"■ {topic}{detail}")
        else:
            lines.append("■ 无明显信息缺口")

        return "\n".join(lines)

    def _format_consolidation(self, result: ConsolidatedResult) -> str:
        """Format the cross-stage consolidation section."""
        lines = ["交叉对照与综合判断："]

        # Consensus
        lines.append("【共识】：")
        if result.consensus:
            for item in result.consensus[:self.max_bullets]:
                topic = item.get("topic", "")
                url = item.get("evidence_url", "")
                if url:
                    lines.append(f"■ {topic}（{url}）")
                else:
                    lines.append(f"■ {topic}")
        else:
            lines.append("■ 三方无明确共识")

        # Divergences
        lines.append("【分歧】：")
        if result.divergences:
            for div in result.divergences[:self.max_bullets]:
                topic = div.get("topic", "")
                positions = div.get("positions", {})
                pos_strs = []
                for provider, position in positions.items():
                    pos_strs.append(f"{provider}: {position}")
                lines.append(f"■ {topic}：{' vs '.join(pos_strs)}")
        else:
            lines.append("■ 三方无明显分歧")

        # High confidence conclusions
        lines.append("【高置信结论】：")
        if result.high_confidence:
            for item in result.high_confidence[:self.max_bullets]:
                conclusion = item.get("conclusion", "")
                url = item.get("url", "")
                if url:
                    lines.append(f"■ {conclusion}（{url}）")
                else:
                    lines.append(f"■ {conclusion}")
        else:
            lines.append("■ 无高置信结论")

        # Low confidence conclusions
        lines.append("【低置信结论】：")
        if result.low_confidence:
            for item in result.low_confidence[:self.max_bullets]:
                conclusion = item.get("conclusion", "")
                reason = item.get("reason", "待验证")
                lines.append(f"■ {conclusion}（{reason}）")
        else:
            lines.append("■ 无低置信结论")

        # Next steps
        lines.append("【下一步研究路径】：")
        if result.next_steps:
            for step in result.next_steps[:self.max_bullets]:
                lines.append(f"■ {step}")
        else:
            lines.append("■ 无进一步研究建议")

        return "\n".join(lines)

    def format_error(
        self,
        research_plan: ResearchPlan,
        error_message: str,
        stage_results: Optional[dict] = None
    ) -> str:
        """
        Format an error output when the pipeline fails.

        Args:
            research_plan: The research plan that was attempted
            error_message: The error message
            stage_results: Optional partial results

        Returns:
            Formatted error output
        """
        lines = [
            "搜索任务概述：",
            f"【主题】：{research_plan.topic}",
            f"【目标】：{research_plan.objective}",
            "【执行状态】：",
        ]

        if stage_results:
            for provider in ["DeepSeek", "Gemini", "ChatGPT"]:
                if provider in stage_results:
                    status = "成功" if stage_results[provider].success else "失败"
                else:
                    status = "未执行"
                lines.append(f"■ Stage{['DeepSeek', 'Gemini', 'ChatGPT'].index(provider) + 1}（{provider}）：{status}")
        else:
            lines.extend([
                "■ Stage1（DeepSeek）：未执行",
                "■ Stage2（Gemini）：未执行",
                "■ Stage3（ChatGPT）：未执行"
            ])

        lines.extend([
            "",
            f"【失败原因】：{error_message}",
            "【补救建议】：",
            "■ 检查API密钥配置是否正确",
            "■ 检查网络连接",
            "■ 尝试更换搜索词",
            "■ 稍后重试"
        ])

        return "\n".join(lines)
