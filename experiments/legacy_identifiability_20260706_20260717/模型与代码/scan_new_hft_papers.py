from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(r"D:\高频变压器")
PAPER_DIR = ROOT / "新增高频变压器"
OUT = ROOT / "可辨识性分析" / "笔记" / "新增高频变压器文献初筛.md"

KEYWORDS = [
    "parameter extraction",
    "impedance matrix",
    "admittance matrix",
    "terminal capacitance",
    "parasitic capacitance",
    "leakage inductance",
    "frequency-dependent",
    "finite element",
    "experimental",
    "wideband",
    "resonance",
    "FRA",
    "equivalent circuit",
    "multiwinding",
    "identification",
    "litz",
]


def clean_text(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages(pdf_path: Path, max_pages: int = 4) -> str:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages[: min(max_pages, len(reader.pages))]:
        try:
            chunks.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            chunks.append(f"[extract error: {exc}]")
    return clean_text("\n".join(chunks))


def find_section(text: str, start_words: tuple[str, ...], stop_words: tuple[str, ...]) -> str:
    lower = text.lower()
    starts = [lower.find(word.lower()) for word in start_words]
    starts = [idx for idx in starts if idx >= 0]
    if not starts:
        return ""
    start = min(starts)
    ends = [lower.find(word.lower(), start + 20) for word in stop_words]
    ends = [idx for idx in ends if idx > start]
    end = min(ends) if ends else min(len(text), start + 1200)
    return text[start:end].strip()


def score(text: str) -> tuple[int, list[str]]:
    lower = text.lower()
    hits = [kw for kw in KEYWORDS if kw.lower() in lower]
    return len(hits), hits


def short_excerpt(text: str, n: int = 950) -> str:
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut + " ..."


def main() -> None:
    rows = []
    for pdf in sorted(PAPER_DIR.glob("*.pdf")):
        text = extract_pages(pdf, max_pages=5)
        abstract = find_section(
            text,
            ("Abstract", "ABSTRACT"),
            ("Index Terms", "Keywords", "I. Introduction", "1 Introduction", "Introduction"),
        )
        if not abstract:
            abstract = short_excerpt(text, 700)
        score_value, hits = score(text)
        rows.append((score_value, pdf.name, abstract, hits))

    rows.sort(reverse=True, key=lambda item: item[0])

    lines = [
        "# 新增高频变压器文献初筛",
        "",
        "说明：本文件由 `scan_new_hft_papers.py` 从 `新增高频变压器` 文件夹中的 PDF 前几页自动抽取摘要和关键词命中，后续人工判断以全文阅读为准。",
        "",
        "## 按相关性粗排",
        "",
        "| 优先级 | 文件 | 命中关键词 | 初步用途 |",
        "|---:|---|---|---|",
    ]

    for rank, (score_value, name, abstract, hits) in enumerate(rows, start=1):
        use = classify_use(name, hits)
        lines.append(
            f"| {rank} | `{name}` | {', '.join(hits) if hits else '-'} | {use} |"
        )

    lines.extend(["", "## 摘要摘录", ""])
    for score_value, name, abstract, hits in rows:
        lines.extend(
            [
                f"### {name}",
                "",
                f"- 关键词命中：{', '.join(hits) if hits else '-'}",
                f"- 初步用途：{classify_use(name, hits)}",
                "",
                short_excerpt(abstract, 1100),
                "",
            ]
        )

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(str(OUT))


def classify_use(name: str, hits: list[str]) -> str:
    lower = name.lower()
    if "full-order_impedance_matrix" in lower or "parameter_extracting" in lower:
        return "最相关：直接支撑宽频阻抗/导纳矩阵建模与参数提取"
    if "terminal_capacitance" in lower or "capacitive_effects" in lower or "capacitance" in lower:
        return "很相关：补足结构到寄生电容/端口电容的物理来源"
    if "leakage_inductance" in lower:
        return "很相关：补足漏感和 Rac(f) 的结构机理与频率相关性"
    if "fra" in lower or "resonance" in lower:
        return "相关：支撑宽频响应/谐振/诊断曲线和等效模型"
    if "artificial_intelligence" in lower:
        return "方法参考：可用于后续 AI/优化设计综述，不是当前主线"
    if "review" in lower:
        return "背景参考：中高频变压器综述、应用约束和典型参数"
    return "待全文确认"


if __name__ == "__main__":
    main()
