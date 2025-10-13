#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Система классификации входящих документов.
Обрабатывает документы из указанной директории, создает эмбеддинги и 
находит наиболее похожие шаблоны из базы данных.
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Импорт модулей проекта
from modules.parsers import parse_file_to_elements
from modules.embeddings import EmbeddingBackend, EmbeddingConfig
from modules.fast_adaptive_chunker import chunk_elements, ChunkingStats
from modules.db import DB
from modules.extractor_names import extract_title_universal

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def classify_document(
    emb_backend: EmbeddingBackend,
    db: DB,
    document_text: str,
    title_text: str = None,
    document_weight: float = 0.70,
    title_weight: float = 0.30
) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, float]]]:
    """
    Улучшенная классификация документа по эмбеддингу документа и названия.
    Всегда возвращает наиболее похожий шаблон с реальным процентом схожести.

    Args:
        emb_backend: бэкенд для создания эмбеддингов
        db: подключение к базе данных
        document_text: полный текст документа
        title_text: название документа (опционально)
        document_weight: вес эмбеддинга документа (по умолчанию 0.70)
        title_weight: вес эмбеддинга названия (по умолчанию 0.30)

    Returns:
        Tuple[template_id, similarity_score, detailed_scores]: ID наиболее похожего шаблона, комбинированный скор и детальные скоры
    """
    # Создаем эмбеддинг документа
    document_embedding = emb_backend.encode([document_text])[0]

    # Создаем эмбеддинг названия, если есть
    title_embedding = None
    if title_text and title_text.strip():
        title_embedding = emb_backend.encode([title_text])[0]

    # Ищем наиболее похожий шаблон без порога отсечения
    similar_docs = db.find_similar_documents_enhanced(
        document_embedding,
        title_embedding=title_embedding,
        limit=1,
        threshold=0.0,  # Убираем порог отсечения
        document_weight=document_weight,
        title_weight=title_weight
    )

    if similar_docs:
        best_match = similar_docs[0]
        detailed_scores = {
            "doc_similarity": best_match["doc_similarity"],
            "title_similarity": best_match["title_similarity"],
            "combined_similarity": best_match["combined_similarity"]
        }
        return best_match["template_id"], best_match["combined_similarity"], detailed_scores

    return None, None, None


def main():
    """
    Главная функция классификации входящих документов.
    """
    parser = argparse.ArgumentParser(
        description="Классификация входящих документов по похожести с шаблонами"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Директория с входящими документами для классификации"
    )
    parser.add_argument(
        "--sbert-path",
        default=str(Path(__file__).parent / "models" / "sbert_large_nlu_ru"),
        help="Путь к локальной модели SBERT"
    )
    parser.add_argument(
        "--similar_id",
        type=float,
        default=0.84,
        help="Порог схожести для классификации (по умолчанию 84%)"
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
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Сохранять результаты в PostgreSQL базу данных"
    )
    parser.add_argument(
        "--file-formats",
        nargs="+",
        default=[".txt", ".pdf", ".docx"],
        help="Поддерживаемые форматы файлов"
    )
    parser.add_argument(
        "--session-id",
        default="default",
        help="Идентификатор сессии для изоляции данных пользователей"
    )
    parser.add_argument(
        "--document-weight",
        type=float,
        default=0.70,
        help="Вес эмбеддинга документа при комбинированной классификации (по умолчанию 0.70)"
    )
    parser.add_argument(
        "--title-weight",
        type=float,
        default=0.30,
        help="Вес эмбеддинга названия при комбинированной классификации (по умолчанию 0.30)"
    )

    args = parser.parse_args()

    logger.info("ЗАПУСК КЛАССИФИКАЦИИ ВХОДЯЩИХ ДОКУМЕНТОВ")
    logger.info("=" * 60)
    logger.info(f"Директория входящих документов: {args.input_dir}")
    logger.info(f"Модель SBERT: {args.sbert_path}")
    logger.info(f"Режим: поиск наиболее похожего шаблона (без порога отсечения)")
    logger.info(f"Размерность эмбеддингов: {args.embedding_dim}")
    logger.info(f"Использование БД: {args.use_db}")
    logger.info(f"ID сессии: {args.session_id}")
    logger.info(
        f"Веса классификации - документ: {args.document_weight:.2f}, название: {args.title_weight:.2f}")

    # Проверка директории
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Директория с документами не найдена: {input_dir}")
        sys.exit(1)

    # Конфигурация эмбеддингов и чанкинга
    cfg = EmbeddingConfig(
        # Параметры модели
        local_sbert_path=args.sbert_path,
        device=args.device,
        batch_size=args.batch_size,
        normalize=True,
        local_files_only=True,
        target_dimension=args.embedding_dim,

        # Параметры чанкинга
        heading_aware=True,
        cohesion_aware=True,
        cohesion_split=True,
        chunk_target_tokens=args.chunk_size,
        chunk_max_tokens=512,
        chunk_min_tokens=64,
        min_chunk_tokens=64,
        overlap_sentences=1,
        sentence_overlap=1,
        table_as_is=True
    )

    # Инициализация бэкенда эмбеддингов
    logger.info("Инициализация модели эмбеддингов...")
    emb = EmbeddingBackend(cfg)

    if emb.model is None:
        logger.error("Не удалось загрузить модель SBERT!")
        sys.exit(1)
    else:
        logger.info(f"Модель загружена, размерность: {emb.dimension}")

    # Инициализация базы данных
    if args.use_db:
        logger.info("Подключение к базе данных...")
        try:
            db = DB()
            db.ensure_schema()
            logger.info("База данных инициализирована")
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            sys.exit(1)
    else:
        logger.error(
            "Для классификации необходимо подключение к БД (--use-db)")
        sys.exit(1)

    # Сбор файлов для обработки
    logger.info(f"Поиск файлов в {input_dir}...")
    files = []
    for fmt in args.file_formats:
        pattern = f"**/*{fmt}"
        found = list(input_dir.glob(pattern))
        files.extend(found)
        logger.info(f"  Найдено {len(found)} файлов формата {fmt}")

    files = sorted(set(files))

    if not files:
        logger.warning(f"Не найдено файлов для обработки в: {input_dir}")
        logger.warning(f"Поддерживаемые форматы: {args.file_formats}")
        return 0

    logger.info(f"Всего файлов для обработки: {len(files)}")

    # Статистика обработки
    total_processed = 0
    total_classified = 0
    total_unclassified = 0
    classification_results = []

    # Обработка каждого файла
    for file_idx, file_path in enumerate(files, 1):
        try:
            logger.info(
                f"[{file_idx}/{len(files)}] Обработка: {file_path.name}")

            # 1. Парсинг документа
            logger.debug("  Парсинг документа...")
            elements = parse_file_to_elements(
                str(file_path),
                category=str(file_path.parent.name)
            )

            if not elements:
                logger.warning(
                    f"  Документ пустой или не удалось распарсить: {file_path.name}")
                continue

            logger.debug(f"  Найдено элементов: {len(elements)}")

            # 2. Извлечение названия документа
            logger.debug(f"  Извлечение названия документа...")
            extracted_title = extract_title_universal(str(file_path))
            if extracted_title:
                logger.info(f"  Извлечено название: '{extracted_title}'")
            else:
                logger.debug("  Название не извлечено")

            # 3. Создание эмбеддинга всего документа
            logger.debug("  Создание эмбеддинга документа...")
            full_document_text = "\n".join(
                [el.text or "" for el in elements if el.text])

            if not full_document_text.strip():
                logger.warning(
                    f"  Документ не содержит текста: {file_path.name}")
                continue

            document_embedding = emb.encode([full_document_text])[0]
            logger.debug(
                f"  Размерность эмбеддинга: {document_embedding.shape}")

            # 3.1. Создание эмбеддинга названия документа
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

            # 4. Классификация - поиск похожего шаблона
            logger.debug("  Поиск похожих шаблонов...")
            similar_template_id, similarity_score, detailed_scores = classify_document(
                emb, db, full_document_text,
                title_text=extracted_title,
                document_weight=args.document_weight,
                title_weight=args.title_weight
            )

            if similar_template_id:
                logger.info(
                    f"  🟢 НАЙДЕН НАИБОЛЕЕ ПОХОЖИЙ ШАБЛОН: комбинированная похожесть {similarity_score*100:.1f}% с шаблоном {similar_template_id}")
                if detailed_scores:
                    logger.info(
                        f"     - Похожесть по документу: {detailed_scores['doc_similarity']*100:.1f}%")
                    logger.info(
                        f"     - Похожесть по названию: {detailed_scores['title_similarity']*100:.1f}%")

                # Считаем как классифицированный, если похожесть >= 50%
                if similarity_score >= 0.5:
                    total_classified += 1
                else:
                    total_unclassified += 1
            else:
                logger.info(
                    f"  🔴 НЕ НАЙДЕН ПОДХОДЯЩИЙ ШАБЛОН")
                total_unclassified += 1

            # 4. Продвинутый чанкинг (как в index_templates.py)
            logger.debug("  Выполнение чанкинга...")
            chunks, stats = chunk_elements(
                elements,
                cfg=cfg,
                model=emb.model,
                tokenizer=None
            )

            if chunks:
                logger.debug(f"  Создано чанков: {stats.total_chunks}")

                # Создание эмбеддингов для чанков
                texts = [c["text"] for c in chunks]
                chunk_embeddings = emb.encode(texts)
                logger.debug(f"  Эмбеддинги чанков: {chunk_embeddings.shape}")
            else:
                logger.warning(
                    f"  Не удалось создать чанки для: {file_path.name}")
                chunks = []
                chunk_embeddings = np.zeros((0, args.embedding_dim))

            # 5. Сохранение в базу данных
            try:
                doc_id = db.insert_incoming_document(
                    name=file_path.name,
                    version="",
                    file_path=str(file_path),
                    document_embedding=document_embedding,
                    similar_template_id=similar_template_id,
                    similarity_score=similarity_score,
                    session_id=args.session_id,
                    title=extracted_title,
                    title_embedding=title_embedding
                )

                if chunks:
                    db.insert_incoming_chunks(
                        doc_id,
                        [{
                            "text": c["text"],
                            "heading_path": c.get("heading_path", []),
                            "chunk_index": i
                        } for i, c in enumerate(chunks)],
                        chunk_embeddings
                    )

                logger.debug(f"  Сохранено в БД: doc_id={doc_id}")
                if title_embedding is not None:
                    logger.debug(
                        f"    - Эмбеддинг названия: {title_embedding.shape}")
                if extracted_title:
                    logger.debug(f"    - Название: '{extracted_title}'")
                logger.debug(f"    - Чанков: {len(chunks)}")

                # Сохраняем результат для итогового отчета
                classification_results.append({
                    "file_name": file_path.name,
                    "doc_id": doc_id,
                    "similar_template_id": similar_template_id,
                    "similarity_score": similarity_score,
                    "chunks_count": len(chunks),
                    "extracted_title": extracted_title,
                    "detailed_scores": detailed_scores
                })

            except Exception as e:
                logger.error(f"  Ошибка сохранения в БД: {e}")
                # Не продолжаем, если не удалось сохранить в БД
                continue

            total_processed += 1

        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path.name}: {e}")
            continue

    # Итоговая статистика
    logger.info("=" * 60)
    logger.info("КЛАССИФИКАЦИЯ ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"Обработано документов: {total_processed}")
    logger.info(f"Найдено похожих шаблонов (>=50%): {total_classified}")
    logger.info(f"Низкая похожесть (<50%): {total_unclassified}")

    if total_processed > 0:
        success_rate = (total_classified / total_processed) * 100
        logger.info(
            f"Процент документов с высокой похожестью (>=50%): {success_rate:.1f}%")

    # Подробный отчет по классифицированным документам
    if classification_results:
        logger.info("\nДЕТАЛЬНЫЙ ОТЧЕТ:")
        logger.info("-" * 40)

        for result in classification_results:
            if result["similar_template_id"]:
                logger.info(f"📄 {result['file_name']}")
                if result.get("extracted_title"):
                    logger.info(f"   Название: '{result['extracted_title']}'")
                logger.info(
                    f"   Комбинированная похожесть: {result['similarity_score']*100:.1f}%")
                if result.get("detailed_scores"):
                    scores = result["detailed_scores"]
                    logger.info(
                        f"     • По документу: {scores['doc_similarity']*100:.1f}%")
                    logger.info(
                        f"     • По названию: {scores['title_similarity']*100:.1f}%")
                logger.info(f"   Шаблон: {result['similar_template_id']}")
                logger.info(f"   Чанков: {result['chunks_count']}")
            else:
                logger.info(f"📄 {result['file_name']} - ШАБЛОН НЕ НАЙДЕН")
                if result.get("extracted_title"):
                    logger.info(f"   Название: '{result['extracted_title']}'")
                    logger.info(f"   Чанков: {result['chunks_count']}")

    # Получение и вывод общих результатов из БД
    try:
        db_results = db.get_classification_results(limit=50)
        if db_results:
            logger.info(f"\nПоследние {len(db_results)} записей в БД:")
            logger.info("-" * 40)
            for result in db_results[:10]:  # Показываем только первые 10
                if result["similarity_score"]:
                    logger.info(
                        f"📄 {result['doc_name']} -> {result['template_name']} ({result['similarity_score']*100:.1f}%)")
                else:
                    logger.info(
                        f"📄 {result['doc_name']} -> НЕ КЛАССИФИЦИРОВАН")
    except Exception as e:
        logger.error(f"Ошибка получения результатов из БД: {e}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nПрервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)
