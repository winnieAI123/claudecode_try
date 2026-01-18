# Tri-Model Deep Search Skill

name: tri-model-deep-search
description: Deep search skill that MUST execute a strict 3-stage search pipeline (DeepSeek+Search → Gemini+Search → ChatGPT+Search), summarize each stage with sources, then output in a buy-side brief format. Fully reusable via config.

---

## ⚡️ EXECUTION PROTOCOL (CRITICAL)

When the user has any search intent, or the task requires external research:

1. **MUST run Stage 1**: DeepSeek API + Search.
2. **MUST run Stage 2**: Gemini API + Search (same topic, same research goal).
3. **MUST run Stage 3**: ChatGPT API + Search (same topic, same research goal).
4. After each stage, **MUST**:
   * Extract key findings
   * Capture **source URLs**
   * Produce a concise stage summary
5. Finally, **MUST** produce a structured consolidated output (per formatting rules below).
6. **Never skip any stage**, even if earlier stages "already look sufficient".
7. If any stage fails, **MUST** still run the remaining stages, then report failure details for the failed stage(s).

---

## 🎯 USER INTENT DETECTION

Trigger this skill if:

* User explicitly asks to search / look up / verify / latest / sources / links / compare viewpoints
* User asks about anything time-sensitive, niche, factual verification, or "最新/近期/现在"
* The assistant lacks confidence and needs online confirmation

If uncertain, **default to triggering the pipeline**.

---

## 🔧 CONFIG (REUSABLE)

All behavior must be configurable via a single config file located at `config/config.yaml`.

See `config/config.yaml` for the full configuration options including:
- Query language and region settings
- Time window and source limits
- Output style configuration
- Caching settings
- Logging configuration
- API credentials (via environment variables)

---

## 🧠 PIPELINE SPEC

For each user request, produce a unified internal "Research Plan":

* **Topic**: user topic
* **Objective**: what the user wants to know
* **Constraints**: time window, region, format
* **Query set**:
  * Primary query (exact)
  * 3–8 expanded queries (synonyms, related entities, ticker/company/product names)
  * 2–4 "counter-queries" (to find refutations / opposing evidence)

### Stage 1 — DeepSeek + Search

* Use DeepSeek model for:
  * query expansion
  * research direction decomposition
  * search execution via a web search tool (or integrated search capability)
* Return:
  * Findings bullets
  * Sources list (URLs)
  * Gaps/unknowns list

### Stage 2 — Gemini + Search

* Repeat the same objective and query set
* MUST run search again (do not reuse Stage 1 results)
* Return the same structured artifacts

### Stage 3 — ChatGPT + Search

* Repeat the same objective and query set
* MUST run search again
* Return the same structured artifacts

### Cross-Stage Consolidation

* Deduplicate overlapping sources if configured
* Identify:
  * Consensus
  * Divergence (model disagreement)
  * Strongest evidence (highest-quality sources)
  * Remaining unknowns & how to resolve

---

## ✅ SOURCE QUALITY RULES

* Prefer primary sources: official docs, filings, vendor docs, academic papers, direct announcements
* Prefer reputable outlets for news and analysis
* Every factual claim must be supported by at least one URL when feasible
* If a claim is not well-supported, mark it explicitly as "不确定/待验证"

---

## 🧾 OUTPUT FORMAT (MUST FOLLOW)

**Language**: concise Chinese by default (unless user asks English).
**Tone**: buy-side daily brief. No narrative. No headlines. No journalist names.

### Rules:

1. All top-level sections must be plain text + colon (e.g. `数据中心产业链信息：`)
2. All subcategories must use `【】` as semantic anchors.
3. Every information line must be: `主体 + 冒号 + 事实陈述`.
4. Multiple items must use `■` as bullets.
5. No news headline style. No narrative. No journalist names.
6. No opinions except in 观点 section.
7. Output must look like a buy-side daily brief, not a blog post.

### Mandatory Sections Template

```
搜索任务概述：
【主题】：
【目标】：
【时间窗】：
【地区/语言】：
【执行状态】：
■ Stage1（DeepSeek）：成功/失败
■ Stage2（Gemini）：成功/失败
■ Stage3（ChatGPT）：成功/失败

DeepSeek 结论：
【关键事实】：
■ 主体：事实陈述（URL）
【观点】：
■ 正方：观点+依据（URL）
■ 反方：观点+依据（URL）
■ 中性：观点+依据（URL）
【待验证/缺口】：
■ 主体：待验证点（原因/下一步）

Gemini 结论：
（同上结构，必须独立输出，必须带 URL）

ChatGPT 结论：
（同上结构，必须独立输出，必须带 URL）

交叉对照与综合判断：
【共识】：
■ 主体：共识点（最强证据URL）
【分歧】：
■ 议题：DeepSeek vs Gemini vs ChatGPT 分歧描述（各自URL）
【高置信结论】：
■ 结论：一句话结论（URL）
【低置信结论】：
■ 结论：标注不确定（URL/缺口说明）
【下一步研究路径】：
■ 动作：要补的搜索/要读的原文/要核对的数据源
```

---

## 🧯 ERROR HANDLING (MUST)

* If a stage errors (timeout/API error), set that stage status=失败
* Still execute the remaining stages
* Final output must include:
  * `【执行状态】` with per-stage success/failure
  * `【失败原因】` with error summary (no sensitive keys)
  * `【补救建议】` (retry/backoff/change query/switch region)

---

## 📦 IMPLEMENTATION STRUCTURE

```
tri-model-deep-search/
├── skill.md                          # This specification
├── config/
│   └── config.yaml                   # All configuration
├── scripts/
│   ├── __init__.py
│   ├── pipeline.py                   # Main orchestration
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                   # Base provider class
│   │   ├── deepseek.py               # DeepSeek provider
│   │   ├── gemini.py                 # Gemini provider
│   │   └── openai_provider.py        # OpenAI/ChatGPT provider
│   ├── search/
│   │   ├── __init__.py
│   │   └── search_client.py          # Web search client
│   └── formatter/
│       ├── __init__.py
│       └── buyside_cn.py             # Chinese buy-side brief formatter
├── logs/                             # Log output directory
└── requirements.txt                  # Python dependencies
```

### Key Implementation Notes:

* Use environment variables for keys; never hardcode secrets
* Add caching keyed by `(topic + time_window + region + query_set_hash)` if enabled
* Save raw responses to logs if configured

---

## 🧪 ACCEPTANCE TESTS

1. **输入**：给我搜索"xxx 公司最近三个月 AI 业务进展"，必须输出三段独立结论（DeepSeek/Gemini/ChatGPT）且均带 URL。
2. **任意一个 provider 断网/报错**，仍必须完成另外两段搜索并在【执行状态】标失败与原因。
3. **输出必须满足格式规则**：顶层冒号、子类【】、行格式"主体：事实"、多条用■。
4. **明确出现【分歧】**当三方结论不一致。
