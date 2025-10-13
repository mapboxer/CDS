# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re
import json
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

STATUS_STANDARD = "СТАНДАРТНЫЙ"
STATUS_NONSTANDARD = "НЕ СТАНДАРТНЫЙ"
STATUS_UNKNOWN = "НЕ МОГУ ОПРЕДЕЛИТЬ"

DEFAULT_CONFIG_PATH = os.environ.get(
    "CDS_AUDIT_RULES", "configs/audit_rules.yaml")

# Optional: spaCy NER (ru_core_news_sm / md)
_SPACY_NLP = None
try:
    import spacy  # type: ignore
    _SPACY_MODEL = os.environ.get("CDS_SPACY_MODEL", "ru_core_news_md")
    try:
        _SPACY_NLP = spacy.load(_SPACY_MODEL)
    except Exception:
        _SPACY_NLP = None
except Exception:
    _SPACY_NLP = None

# Optional: CrossEncoder reranker
_RERANKER = None
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _RERANKER_PATH = os.environ.get("CDS_RERANKER_PATH", "models/reranker_ru")
    if os.path.isdir(_RERANKER_PATH):
        _RERANKER = CrossEncoder(_RERANKER_PATH)
except Exception:
    _RERANKER = None

_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "мая": 5, "май": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}


def _to_date(s: str) -> Optional[dt.date]:
    s = s.strip().lower().replace("г.", "").replace("года", "").strip()
    m = re.match(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        if y < 100:
            y += 2000
        try:
            return dt.date(y, mo, d)
        except ValueError:
            return None
    m = re.match(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", s, re.I)
    if m:
        d = int(m.group(1))
        mon = m.group(2).lower()
        for key, val in _MONTHS.items():
            if mon.startswith(key):
                try:
                    return dt.date(int(m.group(3)), val, d)
                except ValueError:
                    return None
    return None


def _days_between(a: Optional[dt.date], b: Optional[dt.date]) -> Optional[int]:
    if a and b:
        return (b - a).days
    return None


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n|(?:\r?\n){1,}", text.strip())
    merged: List[str] = []
    buf = ""
    for p in parts:
        st = p.strip()
        if not st:
            continue
        if len(st) < 80 and buf:
            buf += " " + st
        else:
            if buf:
                merged.append(buf)
            buf = st
    if buf:
        merged.append(buf)
    return merged if merged else ([text.strip()] if text.strip() else [])


def _iter_matches(pattern: str, text: str) -> List[Tuple[int, int]]:
    spans = []
    for m in re.finditer(pattern, text, flags=re.I | re.M):
        spans.append((m.start(), m.end()))
    return spans


def _highlight(text: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return text
    spans = sorted(spans)
    merged = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    out = []
    last = 0
    for s, e in merged:
        out.append(text[last:s])
        out.append("<mark>")
        out.append(text[s:e])
        out.append("</mark>")
        last = e
    out.append(text[last:])
    return "".join(out)


# YAML loader
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            return json.loads(raw)
        except Exception as e:
            raise RuntimeError("Установите PyYAML: pip install pyyaml") from e
    else:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


@dataclass
class Rule:
    rule_id: str
    name: str
    description: str
    applies_to_templates: List[str]
    severity: str
    mode: str
    params: Dict[str, Any] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True


@dataclass
class Finding:
    rule_id: str
    rule_name: str
    severity: str
    paragraph_index: int
    paragraph_text: str
    paragraph_html: str
    highlights: List[Tuple[int, int]]
    details: Dict[str, Any]
    score: Optional[float] = None  # reranker score if available


class AuditEngine:
    def __init__(self, rules_path: str = DEFAULT_CONFIG_PATH):
        cfg = _load_yaml(rules_path)
        raw_rules = cfg.get("rules", [])
        self.rules: List[Rule] = [Rule(**r)
                                  for r in raw_rules if r.get("enabled", True)]

    def audit(self, text: str, template_hint: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        if not text or len(text.strip()) < 50:
            return STATUS_UNKNOWN, []

        paragraphs = _split_paragraphs(text)
        # optional reranking: pre-score paragraphs per rule to suppress noise
        pre_scores: Dict[Tuple[int, str], float] = {}
        if _RERANKER is not None:
            pairs = []
            keys = []
            for idx, para in enumerate(paragraphs):
                for rule in self.rules:
                    # quick trigger first; if no triggers defined, consider anyway
                    if rule.triggers and not self._any_trigger(rule, para):
                        continue
                    keys.append((idx, rule.rule_id))
                    pairs.append([para[:512], rule.name])
            if pairs:
                scores = _RERANKER.predict(pairs)
                for k, s in zip(keys, scores):
                    pre_scores[k] = float(s)

        findings: List[Finding] = []

        for idx, para in enumerate(paragraphs):
            para_l = para.lower()
            for rule in self.rules:
                if not self._rule_applies(rule, template_hint):
                    continue
                if rule.triggers and not self._any_trigger(rule, para):
                    continue
                # if reranker available: skip low-score pairs
                if pre_scores:
                    s = pre_scores.get((idx, rule.rule_id), 0.0)
                    if s < float(os.environ.get("CDS_RERANKER_THRESHOLD", "0.2")):
                        continue
                violated, details, spans = self._evaluate(rule, para, para_l)
                if violated:
                    score = pre_scores.get(
                        (idx, rule.rule_id)) if pre_scores else None
                    findings.append(Finding(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        paragraph_index=idx,
                        paragraph_text=para,
                        paragraph_html=_highlight(para, spans),
                        highlights=spans,
                        details=details,
                        score=score
                    ))

        if not findings:
            return STATUS_STANDARD, []

        severity_rank = {"critical": 3, "major": 2, "minor": 1}
        top = max((severity_rank.get(f.severity, 1)
                  for f in findings), default=1)
        status = STATUS_NONSTANDARD if top >= 2 else STATUS_STANDARD
        return status, [f.__dict__ for f in findings]

    def _rule_applies(self, rule: Rule, template_hint: Optional[str]) -> bool:
        if not rule.applies_to_templates:
            return True
        if "Все" in rule.applies_to_templates:
            return True
        if not template_hint:
            return True
        for t in rule.applies_to_templates:
            if t.startswith("r:"):
                if re.search(t[2:], template_hint, re.I):
                    return True
            else:
                if t.lower() in template_hint.lower():
                    return True
        return False

    def _any_trigger(self, rule: Rule, para: str) -> bool:
        for trg in rule.triggers:
            if trg.get("type") == "regex_any":
                for pat in trg.get("patterns", []):
                    if re.search(pat, para, flags=re.I | re.M):
                        return True
        return False

    def _evaluate(self, rule: Rule, para: str, para_l: str) -> Tuple[bool, Dict[str, Any], List[Tuple[int, int]]]:
        mode = rule.mode
        if mode == "deny_on_match":
            banned = rule.params.get("banned_regex", [])
            spans = []
            hit = False
            for pat in banned:
                s = _iter_matches(pat, para)
                if s:
                    spans.extend(s)
                    hit = True
            return (hit, {"matched": True} if hit else {}, spans)

        if mode == "custom":
            handler = getattr(self, f"_rule_{rule.rule_id.lower()}", None)
            if callable(handler):
                return handler(rule, para, para_l)
            banned = rule.params.get("banned_regex", [])
            spans = []
            hit = False
            for pat in banned:
                s = _iter_matches(pat, para)
                if s:
                    spans.extend(s)
                    hit = True
            return (hit, {"matched": True} if hit else {}, spans)

        return (False, {}, [])

    # helpers
    def _extract_days_number(self, para_l: str, regexes: List[str]) -> Optional[Tuple[int, str, Tuple[int, int]]]:
        # NER-powered normalization (optional)
        for r in regexes:
            m = re.search(r, para_l, flags=re.I | re.M)
            if m:
                try:
                    days = int(m.group(1))
                    kind = m.group(
                        2) if m.lastindex and m.lastindex >= 2 else ""
                    return days, kind, (m.start(1), m.end(1))
                except Exception:
                    continue
        # spaCy: parse "в двух неделях" etc. (простая эвристика)
        if _SPACY_NLP is not None:
            doc = _SPACY_NLP(para_l)
            for ent in doc.ents:
                if ent.label_.lower() in ("date", "time"):
                    if "дн" in ent.text or "недел" in ent.text:
                        # грубая конверсия
                        m = re.search(r"(\d{1,3})", ent.text)
                        if m:
                            val = int(m.group(1))
                            if "недел" in ent.text:
                                val *= 7
                            return val, "календарные", (ent.start_char, ent.end_char)
        return None

    # R2
    def _rule_r2_term_duration(self, rule: Rule, para: str, para_l: str):
        spans = []
        for pat in rule.params.get("banned_regex", []):
            s = _iter_matches(pat, para)
            spans.extend(s)
        date_pairs = []
        for pat in rule.params.get("date_regex", []):
            for m in re.finditer(pat, para, flags=re.I | re.M):
                if m.lastindex and m.lastindex >= 2:
                    d1 = _to_date(m.group(1))
                    d2 = _to_date(m.group(2))
                    if d1 and d2:
                        date_pairs.append(
                            (d1, d2, (m.start(1), m.end(1)), (m.start(2), m.end(2))))
                elif m.lastindex and m.lastindex >= 1:
                    d2 = _to_date(m.group(1))
                    if d2:
                        date_pairs.append(
                            (None, d2, None, (m.start(1), m.end(1))))
        details = {}
        violated = False
        max_years = int(rule.params.get("max_years", 3))
        for (d1, d2, s1, s2) in date_pairs:
            if s1:
                spans.append(s1)
            if s2:
                spans.append(s2)
            if d1 and d2:
                days = _days_between(d1, d2)
                details["duration_days"] = days
                if days is not None and days > max_years * 365:
                    violated = True
        if spans and any(re.search(p, para, flags=re.I | re.M) for p in rule.params.get("banned_regex", [])):
            violated = True
        return violated, details, spans

    # R3
    def _rule_r3_payment_term(self, rule: Rule, para: str, para_l: str):
        spans = []
        ex = any(re.search(p, para, flags=re.I | re.M)
                 for p in rule.params.get("exception_regex", []))
        res = self._extract_days_number(
            para_l, rule.params.get("days_regex", []))
        details = {}
        violated = False
        if res:
            days, kind, pos = res
            spans.append(pos)
            details["payment_term_days"] = days
            details["kind"] = kind
            min_days = int(rule.params.get("min_days", 60))
            if days < min_days and not ex:
                violated = True
        return violated, details, spans

    # R4
    def _rule_r4_prepayment(self, rule: Rule, para: str, para_l: str):
        spans = []
        prepay = any(re.search(p, para, flags=re.I | re.M)
                     for p in rule.params.get("prepay_regex", []))
        if not prepay:
            return False, {}, []
        cond = any(re.search(p, para, flags=re.I | re.M)
                   for p in rule.params.get("condition_regex", []))
        for p in rule.params.get("prepay_regex", []):
            spans.extend(_iter_matches(p, para))
        violated = not cond
        return violated, {"prepayment_detected": True, "conditions_present": cond}, spans

    # R5
    def _rule_r5_acceptance_term(self, rule: Rule, para: str, para_l: str):
        spans = []
        if not any(re.search(p, para, flags=re.I | re.M) for p in rule.params.get("subject_markers", [])):
            return False, {}, []
        res = self._extract_days_number(
            para_l, rule.params.get("days_regex", []))
        violated = False
        details = {}
        if res:
            days, kind, pos = res
            spans.append(pos)
            details["acceptance_days"] = days
            if days < int(rule.params.get("min_workdays", 5)):
                violated = True
        else:
            violated = True
            details["missing_term"] = True
        return violated, details, spans

    # R6
    def _rule_r6_payment_day(self, rule: Rule, para: str, para_l: str):
        spans = []

        # Проверяем исключающий контекст (санкции/ответственность)
        exclude = rule.params.get("exclude_context", [])
        if exclude:
            for exc_pattern in exclude:
                if re.search(exc_pattern, para, flags=re.I | re.M):
                    # Найден исключающий контекст - не срабатываем
                    return False, {}, []

        # Проверяем наличие якорей платежей (опционально)
        anchors = rule.params.get("anchor_any", [])
        if anchors:
            has_anchor = any(re.search(p, para, flags=re.I | re.M)
                             for p in anchors)
            if not has_anchor:
                # Нет якорей платежа - не срабатываем
                return False, {}, []

        weekly = any(re.search(p, para, flags=re.I | re.M)
                     for p in rule.params.get("weekly_regex", []))
        non_week = any(re.search(p, para, flags=re.I | re.M)
                       for p in rule.params.get("non_weekly_ban_regex", []))
        if non_week and not weekly:
            for p in rule.params.get("non_weekly_ban_regex", []):
                spans.extend(_iter_matches(p, para))
            return True, {"non_weekly_detected": True}, spans
        return False, {}, []

    # R7
    def _rule_r7_allowed_changes_in_contract(self, rule: Rule, para: str, para_l: str):
        spans = []
        violated = False
        for p in rule.params.get("banned_keywords", []):
            s = _iter_matches(p, para)
            if s:
                spans.extend(s)
                violated = True
        return violated, {}, spans

    # R8
    def _rule_r8_allowed_changes_in_addenda(self, rule: Rule, para: str, para_l: str):
        spans = []
        is_addendum = re.search(
            r"доп(?:олнительн\w*)?\s*соглашени", para, flags=re.I | re.M) is not None
        if not is_addendum:
            return False, {}, []
        for p in rule.params.get("ban_if_contains", []):
            s = _iter_matches(p, para)
            if s:
                spans.extend(s)
                return True, {"ban_phrase": True}, spans
        allowed = any(re.search(p, para, flags=re.I | re.M)
                      for p in rule.params.get("allowed_keywords", []))
        if not allowed:
            return True, {"allowed_matched": False}, []
        return False, {}, []

    # R9
    def _rule_r9_spec_app_tz(self, rule: Rule, para: str, para_l: str):
        spans = []
        is_spec = re.search(
            r"спецификац|приложени|технич[её]ск[оеая]\s+задани", para, flags=re.I | re.M) is not None
        if not is_spec:
            return False, {}, []
        must_any = rule.params.get("must_have_any", [])
        has_any = any(re.search(p, para, flags=re.I | re.M) for p in must_any)
        if not has_any:
            return True, {"missing_core_fields": True}, []
        for p in rule.params.get("ban_new_terms_regex", []):
            s = _iter_matches(p, para)
            if s:
                spans.extend(s)
                return True, {"new_terms_detected": True}, spans
        return False, {}, []

    # R10
    def _rule_r10_frame_contract(self, rule: Rule, para: str, para_l: str):
        spans = []
        is_frame = any(re.search(p, para, flags=re.I | re.M)
                       for p in rule.params.get("frame_markers", []))
        if not is_frame:
            return False, {}, []
        ok = any(re.search(p, para, flags=re.I | re.M)
                 for p in rule.params.get("must_have_any", []))
        if not ok:
            return True, {"missing_limits": True}, []
        return False, {}, []


_engine_singleton: Optional["AuditEngine"] = None


def _get_engine() -> "AuditEngine":
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = AuditEngine()
    return _engine_singleton


def check_contract_standardness(contract_text: str, template_hint: Optional[str] = None, config_path: Optional[str] = None):
    """
    Возвращает (status, findings). findings содержит поля:
      rule_id, rule_name, severity, paragraph_index, paragraph_text, paragraph_html, highlights, details, score
    """
    if config_path and config_path != DEFAULT_CONFIG_PATH:
        engine = AuditEngine(config_path)
        return engine.audit(contract_text, template_hint)
    return _get_engine().audit(contract_text, template_hint)
