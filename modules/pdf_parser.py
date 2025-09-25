# parsers/pdf_parser.py
from typing import List, Dict, Iterable, Optional
from .models_data import DocElement
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io
import os
import numpy as np
try:
    import camelot  # optional
    HAS_CAMELOT = True
except Exception:
    HAS_CAMELOT = False


def _save_image(pix, stem: str, idx: int) -> str:
    img_path = Path(f"outputs/media/{stem}_img_{idx}.png")
    pix.save(img_path.as_posix())
    return img_path.as_posix()


def _two_means_threshold(xs: List[float]) -> Optional[float]:
    """Простейший k=2 clustering по x-центрам для разделения колонок (без sklearn)."""
    if len(xs) < 8:  # мало блоков — вероятно, одна колонка
        return None
    x = np.array(xs, dtype=float).reshape(-1, 1)
    c1, c2 = np.min(x), np.max(x)
    for _ in range(12):
        d1 = np.abs(x - c1)
        d2 = np.abs(x - c2)
        g1 = x[d1 <= d2]
        g2 = x[d2 < d1]
        nc1 = np.mean(g1) if len(g1) else c1
        nc2 = np.mean(g2) if len(g2) else c2
        if abs(nc1 - c1) < 1e-2 and abs(nc2 - c2) < 1e-2:
            break
        c1, c2 = nc1, nc2
    thr = float((c1 + c2) / 2.0)
    # убедимся, что разделение осмысленно
    left = (x < thr).sum()
    right = (x >= thr).sum()
    if left >= 3 and right >= 3:
        return thr
    return None


def _extract_text_blocks(page: fitz.Page):
    # blocks: (x0,y0,x1,y1, text, block_no, block_type, ...)
    blocks = page.get_text("blocks")
    txt_blocks = []
    for b in blocks:
        if len(b) >= 5 and isinstance(b[4], str) and b[4].strip():
            x0, y0, x1, y1, text = b[:5]
            txt_blocks.append((x0, y0, x1, y1, text))
    return txt_blocks


def parse_pdf(
    path: str,
    category: Optional[str] = None,
    keep_images: bool = True,
    table_extraction: bool = True,
    ocr_fallback: bool = True,
) -> List[DocElement]:
    doc = fitz.open(path)
    stem = Path(path).stem
    out: List[DocElement] = []
    doc_id = stem
    order = 0

    for pno in range(len(doc)):
        page = doc[pno]
        width, height = page.rect.width, page.rect.height
        blocks = _extract_text_blocks(page)
        # Попытка детекта 2-х колонок
        centers = [(b[0]+b[2])/2 for b in blocks]
        thr = _two_means_threshold(centers) if centers else None

        def emit_par(text, bbox):
            nonlocal order
            out.append(DocElement(
                category=category, doc_path=path, doc_id=doc_id, source_type="pdf",
                page=pno+1, order=order, bbox=list(bbox),
                element_type="paragraph", text=text
            ))
            order += 1

        # Текст: либо по колонкам, либо по естественному порядку
        if thr:
            left = [b for b in blocks if (b[0]+b[2])/2 < thr]
            right = [b for b in blocks if (b[0]+b[2])/2 >= thr]
            for col in (left, right):  # порядок чтения: слева направо
                col.sort(key=lambda b: (b[1], b[0]))
                for (x0, y0, x1, y1, text) in col:
                    emit_par(text.strip(), (x0, y0, x1, y1))
        else:
            # одна колонка/нехватает блоков — сортируем сверху вниз
            blocks.sort(key=lambda b: (b[1], b[0]))
            if blocks:
                for (x0, y0, x1, y1, text) in blocks:
                    emit_par(text.strip(), (x0, y0, x1, y1))
            else:
                # Возможно скан: OCR (fallback)
                if ocr_fallback:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    try:
                        import pytesseract
                        txt = pytesseract.image_to_string(img, lang="rus+eng")
                        if txt.strip():
                            emit_par(txt.strip(), (0, 0, width, height))
                    except Exception:
                        pass

        # Изображения
        if keep_images:
            imgs = page.get_images(full=True)
            for i, (xref, *_rest) in enumerate(imgs):
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    mpath = _save_image(pix, stem=f"{stem}_p{pno+1}", idx=i)
                    out.append(DocElement(
                        category=category, doc_path=path, doc_id=doc_id, source_type="pdf",
                        page=pno+1, element_type="image", media_path=mpath, order=order
                    ))
                    order += 1
                except Exception:
                    continue

        # Таблицы из PDF (best-effort)
        if table_extraction:
            try:
                with pdfplumber.open(path) as pl:
                    pg = pl.pages[pno]
                    tables = pg.extract_tables()  # stream-алгоритм
                    for ti, table in enumerate(tables):
                        if not table:
                            continue
                        rows = ["\t".join([c or "" for c in row])
                                for row in table]
                        text = "\n".join(rows)
                        out.append(DocElement(
                            category=category, doc_path=path, doc_id=doc_id, source_type="pdf",
                            page=pno+1, element_type="table", text=text, order=order
                        ))
                        order += 1
            except Exception:
                pass

            if HAS_CAMELOT:
                try:
                    # Camelot сам определит страницы с сеткой
                    import camelot
                    tables = camelot.read_pdf(
                        path, pages=str(pno+1), flavor="lattice")
                    for i, t in enumerate(tables):
                        csv_text = t.df.to_csv(index=False)
                        out.append(DocElement(
                            category=category, doc_path=path, doc_id=doc_id, source_type="pdf",
                            page=pno+1, element_type="table", text=csv_text, order=order,
                            metadata={"extractor": "camelot",
                                      "shape": list(t.df.shape)}
                        ))
                        order += 1
                except Exception:
                    pass

    return out
