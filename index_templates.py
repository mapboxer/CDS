
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Система индексации документов с продвинутым чанкингом и эмбеддингами.
Поддерживает форматы: .txt, .pdf, .docx
Использует локальную модель SBERT для создания эмбеддингов размерности 1024.
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Импорт модулей проекта
from modules.parsers import parse_file_to_elements
from modules.embeddings import EmbeddingBackend, EmbeddingsStore, EmbeddingConfig
from modules.fast_adaptive_chunker import chunk_elements, ChunkingStats
from modules.db import DB
from modules.extractor_names import extract_title_universal

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Главная функция для индексации документов.
    Выполняет:
    1. Парсинг документов из указанной директории
    2. Продвинутый семантический чанкинг с учетом структуры
    3. Создание эмбеддингов фиксированной размерности 1024
    4. Сохранение в базу данных и/или файловый индекс
    """
    parser = argparse.ArgumentParser(
        description="Индексация документов с продвинутым чанкингом и эмбеддингами"
    )
    parser.add_argument(
        "--templates-dir",
        default=str(Path(__file__).parent / "data" / "templates"),
        help="Директория с документами для индексации"
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(Path(__file__).parent / "outputs"),
        help="Директория для сохранения индекса"
    )
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Сохранять результаты в PostgreSQL базу данных"
    )
    parser.add_argument(
        "--sbert-path",
        default=str(Path(__file__).parent / "models" / "sbert_large_nlu_ru"),
        help="Путь к локальной модели SBERT"
    )
    parser.add_argument(
        "--file-formats",
        nargs="+",
        default=[".txt", ".pdf", ".docx"],
        help="Поддерживаемые форматы файлов"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1024,
        help="Целевая размерность эмбеддингов"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=350,
        help="Целевой размер чанка в токенах"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер батча для создания эмбеддингов"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Устройство для вычислений"
    )

    args = parser.parse_args()

    logger.info(f"Запуск индексации документов")
    logger.info(f"Директория документов: {args.templates_dir}")
    logger.info(f"Директория вывода: {args.outputs_dir}")
    logger.info(f"Использование БД: {args.use_db}")
    logger.info(f"Модель SBERT: {args.sbert_path}")
    logger.info(f"Размерность эмбеддингов: {args.embedding_dim}")

    # Проверка и создание директорий
    templates_dir = Path(args.templates_dir)
    outputs_dir = Path(args.outputs_dir)

    if not templates_dir.exists():
        logger.error(f"Директория с документами не найдена: {templates_dir}")
        sys.exit(1)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "media").mkdir(parents=True,
                                  exist_ok=True)  # для изображений из PDF

    # Конфигурация эмбеддингов и чанкинга
    cfg = EmbeddingConfig(
        # Параметры модели
        local_sbert_path=args.sbert_path,
        device=args.device,
        batch_size=args.batch_size,
        normalize=True,
        local_files_only=True,
        target_dimension=args.embedding_dim,  # Целевая размерность 1024

        # Параметры чанкинга
        heading_aware=True,  # Учитывать заголовки
        cohesion_aware=True,  # Включаем обратно!
        cohesion_split=True,  # Включаем обратно!
        chunk_target_tokens=args.chunk_size,  # Целевой размер чанка
        chunk_max_tokens=512,  # Максимальный размер
        chunk_min_tokens=64,   # Минимальный размер
        min_chunk_tokens=64,   # Алиас для совместимости
        overlap_sentences=1,   # Перекрытие предложений
        sentence_overlap=1,    # Алиас для совместимости
        table_as_is=True      # Таблицы как отдельные чанки
    )

    # Инициализация бэкенда эмбеддингов
    logger.info("Инициализация модели эмбеддингов...")
    emb = EmbeddingBackend(cfg)

    if emb.model is None:
        logger.warning(
            "Модель SBERT не загружена, используется TF-IDF fallback")
    else:
        logger.info(f"Модель загружена, размерность: {emb.dimension}")

    # Сбор файлов для обработки
    logger.info(f"Поиск файлов в {templates_dir}...")
    files = []
    for fmt in args.file_formats:
        pattern = f"**/*{fmt}"
        found = list(templates_dir.glob(pattern))
        files.extend(found)
        logger.info(f"  Найдено {len(found)} файлов формата {fmt}")

    files = sorted(set(files))  # Убираем дубликаты и сортируем

    if not files:
        logger.warning(f"Не найдено файлов для обработки в: {templates_dir}")
        logger.warning(f"Поддерживаемые форматы: {args.file_formats}")
        return

    logger.info(f"Всего файлов для обработки: {len(files)}")

    # Инициализация структур данных
    templates_meta: List[Dict[str, Any]] = []
    chunks_meta: List[Dict[str, Any]] = []
    all_vecs = []
    total_chunks = 0
    total_stats = []  # Для сбора статистики

    # Инициализация базы данных если требуется
    if args.use_db:
        logger.info("Подключение к базе данных...")
        try:
            db = DB()
            db.ensure_schema()
            logger.info("База данных инициализирована")
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            logger.info("Продолжаем без сохранения в БД")
            args.use_db = False

    # Обработка каждого файла
    for t_idx, path in enumerate(files, 1):
        try:
            logger.info(f"[{t_idx}/{len(files)}] Обработка: {path.name}")

            # 1. Парсинг документа
            logger.debug(f"  Парсинг документа...")
            elements = parse_file_to_elements(
                str(path),
                category=str(path.parent.name)
            )

            if not elements:
                logger.warning(
                    f"  Документ пустой или не удалось распарсить: {path.name}")
                continue

            logger.debug(f"  Найдено элементов: {len(elements)}")

            # 2. Продвинутый семантический чанкинг
            logger.debug(f"  Выполнение чанкинга...")
            chunks, stats = chunk_elements(
                elements,
                cfg=cfg,
                model=emb.model,
                tokenizer=None
            )

            if not chunks:
                logger.warning(f"  Не удалось создать чанки для: {path.name}")
                continue

            logger.info(f"  Создано чанков: {stats.total_chunks}, " +
                        f"ср. размер: {stats.avg_chunk_size:.0f} токенов, " +
                        f"качество: {stats.semantic_coherence_score:.2f}")

            total_stats.append(stats)

            # 3. Создание эмбеддингов для чанков
            logger.debug(f"  Создание эмбеддингов для чанков...")
            texts = [c["text"] for c in chunks]

            if texts:
                vecs = emb.encode(texts)
            else:
                # Пустой документ - создаем нулевой вектор
                vecs = np.zeros((1, args.embedding_dim))

            logger.debug(f"  Размерность эмбеддингов чанков: {vecs.shape}")

            # 3.1. Извлечение названия документа
            logger.debug(f"  Извлечение названия документа...")
            extracted_title = extract_title_universal(str(path))
            if extracted_title:
                logger.info(f"  Извлечено название: '{extracted_title}'")
            else:
                logger.debug("  Название не извлечено")

            # 3.2. Создание эмбеддинга всего документа
            logger.debug(f"  Создание эмбеддинга всего документа...")
            # Объединяем весь текст документа
            full_document_text = "\n".join(
                [el.text or "" for el in elements if el.text])

            if full_document_text.strip():
                document_embedding = emb.encode([full_document_text])[
                    0]  # Берем первый (единственный) вектор
                logger.debug(
                    f"  Размерность эмбеддинга документа: {document_embedding.shape}")
            else:
                # Если документ пустой
                document_embedding = np.zeros(args.embedding_dim)
                logger.debug(
                    "  Создан нулевой эмбеддинг для пустого документа")

            # 3.3. Создание эмбеддинга названия документа
            title_embedding = None
            if extracted_title and extracted_title.strip():
                logger.debug(f"  Создание эмбеддинга названия...")
                # Берем первый (единственный) вектор
                title_embedding = emb.encode([extracted_title])[0]
                logger.debug(
                    f"  Размерность эмбеддинга названия: {title_embedding.shape}")
            else:
                logger.debug(
                    "  Эмбеддинг названия не создан (название отсутствует)")

            # 4. Сохранение метаданных
            base_name = path.name
            template_rec = {
                "id": t_idx,
                "name": base_name,
                "version": "",
                "file_path": str(path),
                "chunks_count": len(chunks),
                "avg_chunk_size": stats.avg_chunk_size
            }
            templates_meta.append(template_rec)

            # Сохранение векторов и метаданных чанков
            offs = len(all_vecs)
            all_vecs.extend(list(vecs))

            for i, ch in enumerate(chunks):
                row = {
                    "template_id": t_idx,
                    "chunk_index": i,
                    "heading_path": ch.get("heading_path", []),
                    "text": ch["text"],
                    "vec_offset": offs + i,
                    "token_len": ch.get("token_len", 0),
                    "page_from": ch.get("page_from"),
                    "page_to": ch.get("page_to")
                }
                chunks_meta.append(row)

            total_chunks += len(chunks)

            # 5. Сохранение в базу данных
            if args.use_db:
                try:
                    # Сохраняем шаблон с эмбеддингом всего документа и названием
                    tid = db.insert_template(
                        base_name, "", str(path), document_embedding,
                        title=extracted_title, title_embedding=title_embedding)

                    # Сохраняем чанки с их эмбеддингами
                    db.insert_chunks(
                        tid,
                        [{
                            "text": c["text"],
                            "heading_path": c.get("heading_path", []),
                            "chunk_index": i
                        } for i, c in enumerate(chunks)],
                        vecs
                    )
                    logger.debug(f"  Сохранено в БД: template_id={tid}")
                    logger.debug(
                        f"    - Эмбеддинг документа: {document_embedding.shape}")
                    if title_embedding is not None:
                        logger.debug(
                            f"    - Эмбеддинг названия: {title_embedding.shape}")
                    if extracted_title:
                        logger.debug(f"    - Название: '{extracted_title}'")
                    logger.debug(f"    - Эмбеддинги чанков: {len(chunks)} шт.")
                except Exception as e:
                    logger.error(f"  Ошибка сохранения в БД: {e}")

        except Exception as e:
            logger.error(f"Ошибка обработки файла {path.name}: {e}")
            continue

    # Сохранение файлового индекса
    logger.info("Сохранение индекса...")

    if all_vecs:
        all_vecs_arr = np.vstack(all_vecs)
    else:
        all_vecs_arr = np.zeros((0, args.embedding_dim))

    store = EmbeddingsStore(base_dir=str(outputs_dir))
    store.save_index(templates_meta, chunks_meta, all_vecs_arr)

    # Вывод итоговой статистики
    logger.info("="*60)
    logger.info("ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    logger.info("="*60)
    logger.info(f"Обработано документов: {len(templates_meta)}")
    logger.info(f"Создано чанков: {total_chunks}")
    logger.info(f"Размерность эмбеддингов: {args.embedding_dim}")

    if total_stats:
        avg_quality = np.mean(
            [s.semantic_coherence_score for s in total_stats])
        avg_chunk_size = np.mean([s.avg_chunk_size for s in total_stats])
        total_time = sum([s.processing_time for s in total_stats])

        logger.info(f"Средняя семантическая связность: {avg_quality:.3f}")
        logger.info(f"Средний размер чанка: {avg_chunk_size:.0f} токенов")
        logger.info(f"Общее время обработки: {total_time:.1f} сек")

    logger.info(f"Индекс сохранен в: {outputs_dir}")

    if args.use_db:
        logger.info("Данные также сохранены в PostgreSQL")

    return 0  # Успешное завершение


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nПрервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)
