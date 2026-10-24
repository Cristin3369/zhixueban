# -*- coding: utf-8 -*-
"""考研教材条目：书名 + 版本，一条一条；英语/政治可只有书名。"""

from __future__ import annotations

import re
from typing import Any, Dict, List

VERSION_RE = re.compile(
    r"(?:同济(?:大学)?)?第?\s*[0-9一二三四五六七八九十]+\s*版"
)


def normalize_kind(raw: str) -> str:
    s = (raw or "").strip()
    if re.search(r"英语\s*[（(]?\s*[2二]|英二", s):
        return "英语二"
    if re.search(r"英语|英一", s):
        return "英语一"
    if "政治" in s or "肖" in s:
        return "政治"
    if "专业" in s:
        return "专业课"
    if re.search(r"数学|高数|线代|概率", s):
        return "数学"
    return s or "其他"


def infer_kind(title: str) -> str:
    t = title or ""
    if re.search(r"英语\s*[（(]?\s*[2二]|英二", t):
        return "英语二"
    if re.search(r"英语\s*[（(]?\s*[1一]|英一|^英语$", t):
        return "英语一"
    if re.search(r"政治|肖秀荣|肖爷爷|1000题", t):
        return "政治"
    if re.search(r"数学分析|解析几何|高等代数|抽象代数|实变|复变|点集", t):
        return "专业课"
    if re.search(r"数学|高数|线代|概率|张宇|李永乐|汤家凤", t):
        return "数学"
    return "其他"


def special_titles(title: str, edition: str, kind: str) -> tuple[str, str]:
    """英语只用真题：书名记英一/英二。政治默认肖秀荣1000题。"""
    if kind in ("英语一", "英语二"):
        return kind, ""
    if kind == "政治":
        if (not title) or title in ("政治", "考研政治") or "肖" in title or "1000" in title:
            return "肖秀荣1000题", edition
    return title.strip(), (edition or "").strip()


def split_title_edition(text: str) -> tuple[str, str]:
    text = (text or "").strip(" ，,、;；")
    parts = re.split(r"[，,]", text, maxsplit=1)
    if len(parts) == 2 and VERSION_RE.search(parts[1]):
        return parts[0].strip(), parts[1].strip()
    m = VERSION_RE.search(text)
    if m and m.start() > 0:
        left = text[: m.start()].strip(" ，,、")
        right = (m.group(0) + text[m.end() :]).strip(" ，,、")
        return left or text, right
    return text, ""


def parse_one(chunk: str) -> Dict[str, str]:
    raw = (chunk or "").strip(" ，,、;；")
    if not raw:
        return {}
    kind_prefix = ""
    m = re.match(
        r"^(专业课|数学|英语[一二12]?|英语|政治)\s*[:：]?\s*(.*)$",
        raw,
    )
    rest = raw
    if m:
        kind_prefix = normalize_kind(m.group(1))
        rest = (m.group(2) or "").strip() or m.group(1)
    title, edition = split_title_edition(rest)
    kind = kind_prefix or infer_kind(title or rest)
    title, edition = special_titles(title or rest, edition, kind)
    if not title:
        return {}
    return {"title": title, "edition": edition, "kind": kind}


def parse_books_text(text: str) -> List[Dict[str, str]]:
    """
    识别粘贴：专业课：数学分析，同济第4版，英语1，政治，专业课：解析几何，同济第4版
    """
    if not text or not str(text).strip():
        return []
    blob = str(text).replace("；", "，").replace(";", "，").replace("\n", "，")
    parts = [p.strip() for p in re.split(r"[，,、]", blob) if p.strip()]
    merged: List[str] = []
    for p in parts:
        looks_ver = bool(VERSION_RE.search(p)) and not re.search(
            r"英语|政治|专业课|数学分析|解析几何", p
        )
        if looks_ver and merged:
            merged[-1] = merged[-1] + "，" + p
        else:
            merged.append(p)
    out = []
    for item in merged:
        parsed = parse_one(item)
        if parsed:
            out.append(parsed)
    return _dedupe(out)


def normalize_books(books: Any, note: str = "") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if isinstance(books, dict):
        for kind, names in books.items():
            for name in names or []:
                parsed = parse_one(f"{kind}：{name}")
                if parsed:
                    out.append(parsed)
    elif isinstance(books, list):
        for item in books:
            if isinstance(item, dict) and (item.get("title") or item.get("name")):
                title = str(item.get("title") or item.get("name") or "").strip()
                edition = str(item.get("edition") or item.get("version") or "").strip()
                kind = str(item.get("kind") or infer_kind(title)).strip()
                title, edition = special_titles(title, edition, kind)
                out.append({"title": title, "edition": edition, "kind": kind})
            elif isinstance(item, str):
                out.extend(parse_books_text(item))
    elif isinstance(books, str):
        out.extend(parse_books_text(books))
    if note:
        extra = parse_books_text(note)
        have = {(b["title"], b["edition"], b["kind"]) for b in out}
        for b in extra:
            key = (b["title"], b["edition"], b["kind"])
            if key not in have:
                out.append(b)
                have.add(key)
    return _dedupe(out)


def format_book(book: Dict[str, str]) -> str:
    kind = (book.get("kind") or "").strip()
    title = (book.get("title") or "").strip()
    edition = (book.get("edition") or "").strip()
    if kind and kind != title:
        head = f"{kind}：{title}"
    else:
        head = title
    return f"{head} {edition}".strip()


def group_books(entries: List[Dict[str, str]]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for book in entries:
        key = book.get("kind") or book.get("title") or "其他"
        label = format_book(book)
        grouped.setdefault(key, [])
        if label not in grouped[key]:
            grouped[key].append(label)
    return grouped


def _dedupe(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for item in items:
        key = (item.get("title"), item.get("edition"), item.get("kind"))
        if key in seen or not item.get("title"):
            continue
        seen.add(key)
        out.append(item)
    return out
