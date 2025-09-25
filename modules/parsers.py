# parsers.py
"""
Объединенный модуль для парсинга документов различных форматов.
Поддерживает: .txt, .pdf, .docx
"""

from typing import List, Optional
from pathlib import Path
from .models_data import DocElement
from .docx_parser import parse_docx
from .pdf_parser import parse_pdf


def parse_txt(path: str, category: Optional[str] = None) -> List[DocElement]:
    """Парсинг текстовых файлов"""
    p = Path(path)
    stem = p.stem
    doc_id = stem

    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Попробуем другую кодировку
        with open(path, 'r', encoding='cp1251') as f:
            text = f.read()

    # Разбиваем текст на абзацы
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    out: List[DocElement] = []
    for i, para in enumerate(paragraphs):
        if para:
            out.append(DocElement(
                element_type="paragraph",
                category=category,
                doc_path=path,
                doc_id=doc_id,
                source_type="txt",
                text=para,
                order=i
            ))

    return out


def parse_file_to_elements(
    path: str,
    category: Optional[str] = None,
    **kwargs
) -> List[DocElement]:
    """
    Универсальная функция парсинга файлов.
    Автоматически определяет формат по расширению.

    Args:
        path: путь к файлу
        category: категория документа
        **kwargs: дополнительные параметры для специфичных парсеров

    Returns:
        List[DocElement]: список элементов документа
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    suffix = p.suffix.lower()

    if suffix == '.txt':
        return parse_txt(path, category)
    elif suffix == '.docx':
        return parse_docx(path, category)
    elif suffix == '.pdf':
        # PDF парсер поддерживает дополнительные параметры
        keep_images = kwargs.get('keep_images', True)
        table_extraction = kwargs.get('table_extraction', True)
        ocr_fallback = kwargs.get('ocr_fallback', True)
        return parse_pdf(
            path,
            category,
            keep_images=keep_images,
            table_extraction=table_extraction,
            ocr_fallback=ocr_fallback
        )
    else:
        # Для неподдерживаемых форматов пробуем как текст
        try:
            return parse_txt(path, category)
        except Exception as e:
            raise ValueError(
                f"Неподдерживаемый формат файла: {suffix}. Ошибка: {e}")
