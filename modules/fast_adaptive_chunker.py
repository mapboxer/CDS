import os
import re
import math
import time
import zipfile
from typing import List, Dict, Any, Tuple

# Константы
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TARGET_CHARS = 1200  # Целевой размер чанка в символах
MAX_CHARS = 2000     # Максимальный размер чанка
OVERLAP_CHARS = 200  # Размер перекрытия между чанками
WORDS_PER_PAGE = 300  # Примерное количество слов на страницу


def try_import_docx():
    try:
        import docx  # python-docx
        return docx
    except Exception:
        return None


DOCX_LIB = try_import_docx()


def _xml_text(elem):
    # Aggregate all text from w:t under an element
    texts = []
    for t in elem.iter(f"{W_NS}t"):
        texts.append(t.text or "")
    return "".join(texts)


def parse_docx_fallback(path: str) -> List[Dict[str, Any]]:
    """Fallback DOCX parser using zipfile + XML. Returns list of blocks with types and texts."""
    blocks = []
    with zipfile.ZipFile(path) as z:
        with z.open("word/document.xml") as f:
            data = f.read()
    import xml.etree.ElementTree as ET
    root = ET.fromstring(data)

    for child in root.iter():
        tag = child.tag
        if tag == f"{W_NS}p":
            # detect style
            style = None
            pPr = child.find(f"{W_NS}pPr")
            if pPr is not None:
                pStyle = pPr.find(f"{W_NS}pStyle")
                if pStyle is not None and "val" in pStyle.attrib:
                    style = pStyle.attrib[f"{W_NS}val"] if f"{W_NS}val" in pStyle.attrib else pStyle.attrib.get(
                        "val")
            text = _xml_text(child).strip()
            if not text:
                continue
            level = None
            if style:
                # normalize style id like Heading1, Heading2, or localized "Заголовок1"
                m = re.search(r"Heading([1-6])", style, re.IGNORECASE) or re.search(
                    r"Заголовок\s*([1-6])", style, re.IGNORECASE)
                if m:
                    level = int(m.group(1))
            if level:
                blocks.append(
                    {"type": "heading", "level": level, "text": text})
            else:
                blocks.append({"type": "paragraph", "text": text})
        elif tag == f"{W_NS}tbl":
            # gather table text rows
            rows_text = []
            for row in child.findall(f"{W_NS}tr"):
                cells = []
                for tc in row.findall(f"{W_NS}tc"):
                    cells.append(_xml_text(tc).strip())
                if any(cells):
                    rows_text.append(" | ".join(cells))
            if rows_text:
                blocks.append({"type": "table", "text": "\n".join(rows_text)})
    return blocks


def parse_docx_docxlib(path: str) -> List[Dict[str, Any]]:
    """Parser using python-docx when available for better heading detection and table text."""
    import docx
    doc = docx.Document(path)
    blocks = []

    # Iterate paragraphs and tables in document order by scanning the XML body elements
    body = doc._element.body
    for child in body.iterchildren():
        if child.tag.endswith('}p'):
            p = docx.text.paragraph.Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            # Determine heading level
            level = None
            style_name = getattr(p.style, "name", "") if p.style else ""
            m = re.search(r"Heading\s*([1-6])", style_name, re.IGNORECASE) or re.search(
                r"Заголовок\s*([1-6])", style_name, re.IGNORECASE)
            if m:
                level = int(m.group(1))
            if level:
                blocks.append(
                    {"type": "heading", "level": level, "text": text})
            else:
                blocks.append({"type": "paragraph", "text": text})
        elif child.tag.endswith('}tbl'):
            tbl = docx.table.Table(child, doc)
            rows_text = []
            for row in tbl.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows_text.append(" | ".join(cells))
            if rows_text:
                blocks.append({"type": "table", "text": "\n".join(rows_text)})
    return blocks


def parse_docx(path: str) -> List[Dict[str, Any]]:
    if DOCX_LIB is not None:
        try:
            return parse_docx_docxlib(path)
        except Exception:
            # fall through to fallback if python-docx fails
            pass
    return parse_docx_fallback(path)


def clean_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Normalize whitespace and remove ultra-short noise lines
    cleaned = []
    for b in blocks:
        t = re.sub(r"[ \t]+", " ", b.get("text", "")).strip()
        t = re.sub(r"\n{3,}", "\n\n", t)
        if not t:
            continue
        # Remove single-character paragraphs (often artifacts), but keep if it's a heading
        if b["type"] != "heading" and len(t) <= 2 and not re.search(r"\w", t):
            continue
        b2 = dict(b)
        b2["text"] = t
        cleaned.append(b2)

    # Boilerplate filter: drop lines repeated many times (likely headers/footers)
    all_lines = []
    for b in cleaned:
        for line in b["text"].splitlines():
            s = line.strip()
            if s:
                all_lines.append(s)
    freq = {}
    for line in all_lines:
        freq[line] = freq.get(line, 0) + 1

    def is_boiler(line: str) -> bool:
        return (freq.get(line, 0) >= 6) and (len(line) <= 80)
    # reassemble blocks without boilerplate-only content
    result = []
    for b in cleaned:
        lines = [ln for ln in b["text"].splitlines() if not is_boiler(ln)]
        if not lines:
            continue
        b2 = dict(b)
        b2["text"] = "\n".join(lines).strip()
        if b2["text"]:
            result.append(b2)
    return result


def estimate_pages(blocks: List[Dict[str, Any]]) -> int:
    words = 0
    for b in blocks:
        words += len(re.findall(r"\w+", b["text"], flags=re.UNICODE))
    return max(1, math.ceil(words / WORDS_PER_PAGE))


def chunk_blocks(blocks: List[Dict[str, Any]], file_id: str) -> List[Dict[str, Any]]:
    pages = estimate_pages(blocks)
    full_text = "\n\n".join([b["text"] for b in blocks])
    if pages < 4:
        return [{
            "file_id": file_id,
            "chunk_id": f"{file_id}::0",
            "section_path": "",
            "text": full_text,
            "meta": {"estimated_pages": pages, "no_chunking": True}
        }]

    # Structural chunking
    chunks = []
    section_path = []
    buff = ""
    start_idx = 0

    def flush(force=False):
        nonlocal buff, start_idx, chunks
        if not buff:
            return
        if (len(buff) >= TARGET_CHARS) or force:
            # build chunk
            path_str = " > ".join(section_path)
            text = buff.strip()
            chunk_id = f"{file_id}::{len(chunks)}"
            chunks.append({
                "file_id": file_id,
                "chunk_id": chunk_id,
                "section_path": path_str,
                "text": text,
                "meta": {"estimated_pages": pages, "no_chunking": False}
            })
            # create small overlap by keeping tail
            tail = text[-OVERLAP_CHARS:]
            buff = tail + "\n"
            start_idx += len(text)

    for b in blocks:
        if b["type"] == "heading":
            # starting a new logical section
            # flush current buffer before updating path if buffer is substantial
            if len(buff) >= TARGET_CHARS // 2:
                flush(force=True)
            # update section path by level
            level = b.get("level", 1)
            # ensure path of proper depth
            if level <= len(section_path):
                section_path = section_path[:level-1]
            # add this heading
            section_path = section_path + [b["text"]]
            # also include heading line to buffer to anchor semantics
            new_piece = f"\n# {b['text']}\n"
        else:
            new_piece = b["text"] + "\n"

        # if adding piece exceeds MAX_CHARS, flush first
        if len(buff) + len(new_piece) > MAX_CHARS:
            flush(force=True)
        buff += new_piece
        if len(buff) >= TARGET_CHARS:
            flush()

    # flush remainder
    flush(force=True)
    return chunks


def process_file(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    t0 = time.time()
    blocks = parse_docx(path)
    blocks = clean_blocks(blocks)
    pages = estimate_pages(blocks)
    chunks = chunk_blocks(blocks, os.path.basename(path))
    dt = (time.time() - t0) * 1000.0
    stats = {
        "file": os.path.basename(path),
        "estimated_pages": pages,
        "chunks": len(chunks),
        "avg_chunk_chars": int(sum(len(c["text"]) for c in chunks) / max(1, len(chunks))),
        "time_ms": round(dt, 2),
    }
    return chunks, stats

# Совместимость с существующей системой


class ChunkingStats:
    def __init__(self, total_chunks: int, avg_chunk_size: float, processing_time: float):
        self.total_chunks = total_chunks
        self.avg_chunk_size = avg_chunk_size
        self.semantic_coherence_score = 0.85  # Фиксированное значение для совместимости
        self.processing_time = processing_time


def chunk_elements(elements, cfg=None, model=None, tokenizer=None):
    """
    Адаптер для совместимости с существующей системой чанкинга.
    Принимает элементы документа и возвращает чанки в ожидаемом формате.
    """
    t0 = time.time()

    # Конвертируем elements в blocks для fast_adaptive_chunker
    blocks = []
    for elem in elements:
        if hasattr(elem, 'text') and elem.text:
            # Определяем тип элемента
            elem_type = "paragraph"
            level = None

            # Пытаемся определить заголовок
            if hasattr(elem, 'metadata') and elem.metadata:
                if 'heading' in str(elem.metadata).lower() or 'title' in str(elem.metadata).lower():
                    elem_type = "heading"
                    # Пытаемся извлечь уровень заголовка
                    level_match = re.search(r'(\d+)', str(elem.metadata))
                    if level_match:
                        level = int(level_match.group(1))
                        level = min(level, 6)  # Ограничиваем уровень 6
                    else:
                        level = 1

            # Также проверяем сам текст на признаки заголовка
            text = elem.text.strip()
            if (len(text) < 100 and
                (text.isupper() or
                 re.match(r'^\d+\.?\s+[А-ЯA-Z]', text) or
                 any(word in text.upper() for word in ['ГЛАВА', 'РАЗДЕЛ', 'СТАТЬЯ', 'ПУНКТ']))):
                elem_type = "heading"
                if level is None:
                    level = 1

            block = {"type": elem_type, "text": text}
            if level:
                block["level"] = level

            blocks.append(block)

    if not blocks:
        # Возвращаем пустой результат
        stats = ChunkingStats(0, 0, time.time() - t0)
        return [], stats

    # Очищаем блоки
    blocks = clean_blocks(blocks)

    # Создаем чанки
    file_id = "doc"  # Временный ID
    chunks = chunk_blocks(blocks, file_id)

    # Конвертируем в ожидаемый формат
    result_chunks = []
    for i, chunk in enumerate(chunks):
        # Создаем структуру совместимую с существующим кодом
        result_chunk = {
            "text": chunk["text"],
            "chunk_index": i,
            "heading_path": chunk["section_path"].split(" > ") if chunk["section_path"] else [],
            # Примерное количество токенов
            "token_len": len(chunk["text"].split()),
        }
        result_chunks.append(result_chunk)

    # Создаем статистику
    avg_chars = sum(len(c["text"])
                    for c in result_chunks) / max(1, len(result_chunks))
    stats = ChunkingStats(
        total_chunks=len(result_chunks),
        avg_chunk_size=avg_chars / 4,  # Примерное соотношение символов к токенам
        processing_time=time.time() - t0
    )

    return result_chunks, stats
