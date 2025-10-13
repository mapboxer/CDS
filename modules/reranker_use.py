#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Модуль для переранжирования документов с использованием reranker модели.
Используется только когда разница между топ-2 шаблонами < 3%.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# Используем только те библиотеки, которые уже есть в проекте
from sentence_transformers import CrossEncoder


class DocumentReranker:
    """
    Класс для переранжирования документов на основе чанков
    с использованием reranker модели.
    """

    def __init__(self, model_path: str = "./models/reranker_ru", device: str = "cpu"):
        """
        Инициализация reranker модели.

        Args:
            model_path: путь к модели reranker
            device: устройство для вычислений (cpu/cuda)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Загрузка reranker модели."""
        try:
            print(f"Загрузка reranker модели из {self.model_path}...")
            self.model = CrossEncoder(
                str(self.model_path),
                device=self.device,
                max_length=512
            )
            print(f"✅ Reranker модель загружена на {self.device}")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки reranker модели: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Проверка доступности модели."""
        return self.model is not None

    def rerank_candidates(
        self,
        doc_chunks: List[Dict[str, Any]],
        template_candidates: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Переранжирование кандидатов на основе чанков.

        Args:
            doc_chunks: список чанков документа
                [{"text": "...", "heading": "...", ...}, ...]
            template_candidates: список кандидатов-шаблонов
                [{"template_id": "...", "template_name": "...", 
                  "chunks": [...], "initial_score": 0.85}, ...]
            top_k: количество возвращаемых кандидатов (None = все)

        Returns:
            List[Dict]: переранжированные кандидаты с новыми скорами
        """
        if not self.is_available():
            print("⚠️ Reranker недоступен, возвращаем исходный порядок")
            return template_candidates

        if not doc_chunks or not template_candidates:
            return template_candidates

        print(f"\n🔄 Переранжирование {len(template_candidates)} кандидатов...")
        print(f"   Чанков документа: {len(doc_chunks)}")

        reranked_results = []

        for candidate in template_candidates:
            template_id = candidate["template_id"]
            template_name = candidate["template_name"]
            template_chunks = candidate.get("chunks", [])
            initial_score = candidate.get("initial_score", 0.0)

            if not template_chunks:
                # Если нет чанков, оставляем исходный скор
                reranked_results.append({
                    **candidate,
                    "reranker_score": initial_score,
                    "final_score": initial_score,
                    "chunk_scores": []
                })
                continue

            print(
                f"\n   Обработка: {template_name} (ID: {template_id[:8]}...)")
            print(f"   Чанков шаблона: {len(template_chunks)}")

            # Создаем пары (doc_chunk, template_chunk) для reranker
            chunk_pairs = []
            pair_info = []

            for doc_chunk in doc_chunks:
                doc_text = doc_chunk.get("text", "")
                if not doc_text.strip():
                    continue

                for template_chunk in template_chunks:
                    template_text = template_chunk.get("text", "")
                    if not template_text.strip():
                        continue

                    # Пара для reranker: [query, passage]
                    chunk_pairs.append([doc_text, template_text])
                    pair_info.append({
                        "doc_chunk_idx": doc_chunk.get("ord", 0),
                        "template_chunk_idx": template_chunk.get("ord", 0),
                        "doc_heading": doc_chunk.get("heading", ""),
                        "template_heading": template_chunk.get("heading", "")
                    })

            if not chunk_pairs:
                # Если не удалось создать пары, оставляем исходный скор
                reranked_results.append({
                    **candidate,
                    "reranker_score": initial_score,
                    "final_score": initial_score,
                    "chunk_scores": []
                })
                continue

            print(f"   Создано пар для сравнения: {len(chunk_pairs)}")

            # Получаем скоры от reranker для всех пар
            try:
                scores = self.model.predict(
                    chunk_pairs, batch_size=32, show_progress_bar=False)

                # Нормализуем скоры в диапазон [0, 1] через сигмоид
                normalized_scores = 1 / (1 + np.exp(-scores))

                # Вычисляем агрегированный скор
                # Используем максимальный скор среди всех пар (best chunk match)
                max_score = float(np.max(normalized_scores))
                # Используем средний скор топ-10% пар (robust aggregation)
                top_10_percent = max(1, len(normalized_scores) // 10)
                top_scores = np.partition(
                    normalized_scores, -top_10_percent)[-top_10_percent:]
                avg_top_score = float(np.mean(top_scores))

                # Комбинированный скор: 60% максимальный + 40% средний топ-10%
                reranker_score = 0.6 * max_score + 0.4 * avg_top_score

                # Финальный скор: комбинация initial_score и reranker_score
                # 50% исходный скор (embeddings) + 50% reranker скор
                final_score = 0.5 * initial_score + 0.5 * reranker_score

                print(f"   Reranker скор: {reranker_score:.4f}")
                print(
                    f"   Финальный скор: {final_score:.4f} (был {initial_score:.4f})")

                # Сохраняем результаты
                reranked_results.append({
                    **candidate,
                    "reranker_score": reranker_score,
                    "final_score": final_score,
                    "max_chunk_score": max_score,
                    "avg_top_chunk_score": avg_top_score,
                    "total_chunk_pairs": len(chunk_pairs)
                })

            except Exception as e:
                print(f"   ⚠️ Ошибка при обработке: {e}")
                # В случае ошибки оставляем исходный скор
                reranked_results.append({
                    **candidate,
                    "reranker_score": initial_score,
                    "final_score": initial_score,
                    "chunk_scores": []
                })

        # Сортируем по финальному скору
        reranked_results.sort(key=lambda x: x["final_score"], reverse=True)

        # Возвращаем top_k результатов
        if top_k:
            reranked_results = reranked_results[:top_k]

        print(f"\n✅ Переранжирование завершено")
        print("=" * 70)
        print("РЕЗУЛЬТАТЫ:")
        for i, result in enumerate(reranked_results[:5], 1):
            print(f"  {i}. {result['template_name'][:50]}")
            print(f"     Исходный: {result['initial_score']:.4f} → "
                  f"Reranker: {result['reranker_score']:.4f} → "
                  f"Финальный: {result['final_score']:.4f}")
        print("=" * 70)

        return reranked_results

    def should_use_reranker(
        self,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.03
    ) -> bool:
        """
        Определяет, нужно ли использовать reranker.
        Используется, если разница между топ-2 кандидатами < threshold.

        Args:
            candidates: список кандидатов с initial_score
            threshold: порог разницы (по умолчанию 3%)

        Returns:
            bool: True если нужно использовать reranker
        """
        if not self.is_available():
            return False

        if len(candidates) < 2:
            return False

        # Сортируем по initial_score
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("initial_score", 0),
            reverse=True
        )

        top1_score = sorted_candidates[0].get("initial_score", 0)
        top2_score = sorted_candidates[1].get("initial_score", 0)

        diff = abs(top1_score - top2_score)

        use_reranker = diff < threshold

        if use_reranker:
            print(f"\n⚠️ БЛИЗКИЕ РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ!")
            print(
                f"   Топ-1: {sorted_candidates[0].get('template_name', 'N/A')[:50]} - {top1_score:.4f}")
            print(
                f"   Топ-2: {sorted_candidates[1].get('template_name', 'N/A')[:50]} - {top2_score:.4f}")
            print(
                f"   Разница: {diff:.4f} ({diff*100:.2f}%) < {threshold*100:.1f}%")
            print(f"   🔄 Запускаем RERANKER для уточнения...")
        else:
            print(f"\n✅ Результаты классификации уверенные")
            print(
                f"   Разница: {diff:.4f} ({diff*100:.2f}%) >= {threshold*100:.1f}%")
            print(f"   ℹ️  Reranker не требуется")

        return use_reranker


def get_document_chunks_from_db(db, doc_id: str) -> List[Dict[str, Any]]:
    """
    Получение чанков документа из базы данных.

    Args:
        db: объект подключения к БД
        doc_id: ID документа

    Returns:
        List[Dict]: список чанков документа
    """
    try:
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, ord, heading, text
                    FROM docs_inserted_chunks
                    WHERE doc_id = %s
                    ORDER BY ord
                """, (doc_id,))

                chunks = []
                for row in cur.fetchall():
                    chunks.append({
                        "chunk_id": row[0],
                        "ord": row[1],
                        "heading": row[2] or "",
                        "text": row[3] or ""
                    })

                return chunks
    except Exception as e:
        print(f"Ошибка получения чанков документа: {e}")
        return []


def get_template_chunks_from_db(db, template_id: str) -> List[Dict[str, Any]]:
    """
    Получение чанков шаблона из базы данных.

    Args:
        db: объект подключения к БД
        template_id: ID шаблона

    Returns:
        List[Dict]: список чанков шаблона
    """
    try:
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, ord, heading, text
                    FROM template_chunks
                    WHERE template_id = %s
                    ORDER BY ord
                """, (template_id,))

                chunks = []
                for row in cur.fetchall():
                    chunks.append({
                        "chunk_id": row[0],
                        "ord": row[1],
                        "heading": row[2] or "",
                        "text": row[3] or ""
                    })

                return chunks
    except Exception as e:
        print(f"Ошибка получения чанков шаблона: {e}")
        return []
