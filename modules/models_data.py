# models.py
"""
Модели данных для обработки документов.
Используем обычные Python классы вместо pydantic для совместимости.
"""
from typing import Optional, List, Dict, Literal
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path


ElementType = Literal["paragraph", "title", "list",
                      "table", "formula", "image", "metadata", "slide", "row"]


@dataclass
class DocElement:
    """Элемент документа"""
    element_type: str  # ElementType
    id: str = field(default_factory=lambda: str(uuid4()))
    category: Optional[str] = None
    doc_path: Optional[str] = None
    doc_id: Optional[str] = None
    source_type: Optional[str] = None  # pdf/docx/pptx/xlsx/csv/txt/pg
    page: Optional[int] = None
    order: Optional[int] = None
    bbox: Optional[List[float]] = None   # [x0,y0,x1,y1] для PDF
    text: Optional[str] = None           # основной текст
    html: Optional[str] = None           # табличный/формульный HTML
    # сохранённое изображение/формула (png)
    media_path: Optional[str] = None
    headings_path: Optional[List[str]] = None  # ["Гл.1","1.1",...]
    parents: Optional[List[str]] = None        # id узлов-родителей в графе
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Чанк документа"""
    text: str
    token_len: int
    char_len: int
    chunk_index: int
    from_order: int
    to_order: int
    id: str = field(default_factory=lambda: str(uuid4()))
    category: Optional[str] = None
    doc_id: Optional[str] = None
    doc_path: Optional[str] = None
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    heading_path: Optional[List[str]] = None
    element_types: Optional[List[str]] = None
    media_refs: Optional[List[str]] = None
