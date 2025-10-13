#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit приложение для классификации документов с многопользовательским режимом.
Показывает результаты анализа похожести документов с шаблонами.
Поддерживает изоляцию данных между пользователями через сессии.
"""

import streamlit as st
import pandas as pd
import subprocess
import tempfile
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import numpy as np

# Импорт модулей проекта
from modules.db import DB
from modules.audit_standart import check_contract_standardness, STATUS_STANDARD, STATUS_NONSTANDARD, STATUS_UNKNOWN
from modules.reranker_use import DocumentReranker, get_document_chunks_from_db, get_template_chunks_from_db


def get_session_id() -> str:
    """
    Получает или создает уникальный идентификатор сессии для текущего пользователя.
    Использует встроенные механизмы сессий Streamlit.

    Returns:
        str: Уникальный идентификатор сессии
    """
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def classify_similarity(similarity_score: Optional[float]) -> tuple[str, str]:
    """
    Классификация документа по проценту схожести.

    Returns:
        tuple[status, color]: статус и цвет для отображения
    """
    if similarity_score is None or similarity_score == 0:
        return "🔴 Затрудняюсь ответить", "gray"
    elif 0.9 <= similarity_score <= 1.0:
        return "🟢 Соответствует найденному шаблону", "green"
    elif 0.8 <= similarity_score < 0.9:
        return "🟡 Скорее всего соответствует предложенным шаблонам", "orange"
    else:
        return "🔴 Скорее всего не соответствует ни одному шаблону", "red"


def get_classification_results(session_id: str) -> List[Dict[str, Any]]:
    """
    Получение результатов классификации из базы данных для конкретной сессии.

    Args:
        session_id: Идентификатор сессии пользователя

    Returns:
        List[Dict]: Список результатов классификации
    """
    try:
        db = DB()
        results = db.get_classification_results_by_session(
            session_id, limit=100)
        return results
    except Exception as e:
        st.error(f"Ошибка подключения к базе данных: {e}")
        return []


def get_all_templates() -> List[Dict[str, Any]]:
    """Получение всех шаблонов для выпадающего меню."""
    try:
        db = DB()
        templates = db.get_all_templates()
        return templates
    except Exception as e:
        st.error(f"Ошибка получения шаблонов: {e}")
        return []


def calculate_similarity(doc_embedding, template_embedding,
                         doc_title_emb=None, template_title_emb=None,
                         document_weight: float = 0.70, title_weight: float = 0.30) -> float:
    """
    Вычисление процента соответствия между документом и шаблоном.
    Использует ту же логику, что и в classification.py

    Args:
        doc_embedding: эмбеддинг документа
        template_embedding: эмбеддинг шаблона
        doc_title_emb: эмбеддинг названия документа (опционально)
        template_title_emb: эмбеддинг названия шаблона (опционально)
        document_weight: вес документа (по умолчанию 0.70)
        title_weight: вес названия (по умолчанию 0.30)

    Returns:
        float: процент соответствия от 0 до 1
    """
    import numpy as np

    # Вычисляем косинусное сходство для документов
    doc_similarity = np.dot(doc_embedding, template_embedding) / (
        np.linalg.norm(doc_embedding) * np.linalg.norm(template_embedding)
    )

    # Если есть эмбеддинги названий, вычисляем их сходство
    title_similarity = 0.0
    if doc_title_emb is not None and template_title_emb is not None:
        title_similarity = np.dot(doc_title_emb, template_title_emb) / (
            np.linalg.norm(doc_title_emb) * np.linalg.norm(template_title_emb)
        )

    # Комбинированное сходство
    combined_similarity = document_weight * \
        doc_similarity + title_weight * title_similarity

    return float(combined_similarity)


def recalculate_similarity_for_template(doc_id: str, template_id: str) -> Optional[float]:
    """
    Пересчет процента соответствия для выбранного пользователем шаблона.

    Args:
        doc_id: ID документа
        template_id: ID выбранного шаблона

    Returns:
        float: новый процент соответствия или None при ошибке
    """
    try:
        db = DB()

        # Получаем эмбеддинги документа
        doc_embeddings = db.get_document_embeddings(doc_id)
        if not doc_embeddings or doc_embeddings[0] is None:
            st.error("Не удалось получить эмбеддинг документа")
            return None

        doc_emb, doc_title_emb = doc_embeddings

        # Получаем эмбеддинги шаблона
        template_embeddings = db.get_template_embeddings(template_id)
        if not template_embeddings or template_embeddings[0] is None:
            st.error("Не удалось получить эмбеддинг шаблона")
            return None

        template_emb, template_title_emb = template_embeddings

        # Вычисляем новый процент соответствия
        new_similarity = calculate_similarity(
            doc_emb, template_emb,
            doc_title_emb, template_title_emb,
            document_weight=0.70,
            title_weight=0.30
        )

        return new_similarity

    except Exception as e:
        st.error(f"Ошибка пересчета соответствия: {e}")
        return None


def update_user_choice(doc_id: str, template_id: str, new_similarity_score: float = None) -> bool:
    """Обновление выбора пользователя для документа с пересчетом similarity_score."""
    try:
        db = DB()
        return db.update_user_choice(doc_id, template_id, new_similarity_score)
    except Exception as e:
        st.error(f"Ошибка обновления выбора: {e}")
        return False


def get_document_data(doc_id: str) -> Optional[Dict[str, Any]]:
    """Получение данных документа из базы данных."""
    try:
        db = DB()
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        di.id,
                        di.name,
                        di.version,
                        di.created_at,
                        di.similarity_score,
                        di.session_id,
                        di.user_choice_doc_id,
                        t.id as template_id,
                        t.name as template_name,
                        t.version as template_version
                    FROM docs_inserted di
                    LEFT JOIN templates t ON di.user_choice_doc_id = t.id
                    WHERE di.id = %s
                """, (doc_id,))

                row = cur.fetchone()
                if row:
                    return {
                        "doc_id": row[0],
                        "doc_name": row[1],
                        "doc_version": row[2],
                        "created_at": row[3],
                        "similarity_score": float(row[4]) if row[4] else None,
                        "session_id": row[5],
                        "user_choice_doc_id": row[6],
                        "template_id": row[7],
                        "template_name": row[8],
                        "template_version": row[9]
                    }
        return None
    except Exception as e:
        st.error(f"Ошибка получения данных документа: {e}")
        return None


def get_template_chunks(template_id: str) -> List[Dict[str, Any]]:
    """Получение чанков шаблона для редактирования."""
    try:
        db = DB()
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
                        "order": row[1],
                        "heading": row[2],
                        "text": row[3]
                    })
                return chunks
    except Exception as e:
        st.error(f"Ошибка получения чанков шаблона: {e}")
        return []


def save_edited_document(doc_id: str, edited_chunks: List[Dict[str, Any]]) -> bool:
    """Сохранение отредактированного документа."""
    try:
        db = DB()
        with db._conn() as conn:
            with conn.cursor() as cur:
                # Удаляем старые чанки документа
                cur.execute("""
                    DELETE FROM docs_inserted_chunks 
                    WHERE doc_id = %s
                """, (doc_id,))

                # Вставляем новые чанки
                for i, chunk in enumerate(edited_chunks):
                    chunk_id = f"{doc_id}:{i}"
                    cur.execute("""
                        INSERT INTO docs_inserted_chunks(chunk_id, doc_id, ord, heading, text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (chunk_id, doc_id, i, chunk.get("heading", ""), chunk["text"], "[]"))

        conn.commit()
        return True
    except Exception as e:
        st.error(f"Ошибка сохранения документа: {e}")
        return False


def run_classification(uploaded_files: List, temp_dir: str, session_id: str) -> bool:
    """
    Запуск скрипта классификации для загруженных файлов с привязкой к сессии.

    Args:
        uploaded_files: Список загруженных файлов
        temp_dir: Временная директория для файлов
        session_id: Идентификатор сессии пользователя

    Returns:
        bool: True если обработка прошла успешно
    """
    try:
        # Команда для запуска классификации с передачей session_id
        cmd = [
            "python", "classification.py",
            "--input-dir", temp_dir,
            "--sbert-path", "./models/sbert_large_nlu_ru",
            "--similar_id", "0.0",  # Обновленный порог по умолчанию
            "--embedding-dim", "1024",
            "--chunk-size", "350",
            "--batch-size", "16",  # Обновленный размер батча
            "--device", "cpu",
            "--use-db",
            "--session-id", session_id,  # Передаем session_id в скрипт
            "--document-weight", "0.70",
            "--title-weight", "0.30"
        ]

        # Запуск в conda окружении
        conda_cmd = ["conda", "run", "-n", "corp_pp_doc"] + cmd

        with st.spinner("Обработка документов... Это может занять несколько минут."):
            result = subprocess.run(
                conda_cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )

        if result.returncode == 0:
            st.success("🟢 Документы успешно обработаны!")
            return True
        else:
            st.error(f"🔴 Ошибка обработки: {result.stderr}")
            return False

    except Exception as e:
        st.error(f"🔴 Ошибка запуска классификации: {e}")
        return False


def clear_session_data(session_id: str) -> bool:
    """
    Очистка данных текущей сессии из базы данных.

    Args:
        session_id: Идентификатор сессии пользователя

    Returns:
        bool: True если очистка прошла успешно
    """
    try:
        db = DB()
        db.clear_session_data(session_id)
        st.success("🟢 Данные сессии очищены!")
        return True
    except Exception as e:
        st.error(f"🔴 Ошибка очистки данных: {e}")
        return False


def get_document_full_text(doc_id: str) -> Optional[str]:
    """
    Получение полного текста документа из чанков в базе данных.

    Args:
        doc_id: Идентификатор документа

    Returns:
        str: Полный текст документа или None при ошибке
    """
    try:
        db = DB()
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT text
                    FROM docs_inserted_chunks 
                    WHERE doc_id = %s
                    ORDER BY ord
                """, (doc_id,))

                chunks = cur.fetchall()
                if chunks:
                    return ' '.join([chunk[0] for chunk in chunks])
                return None
    except Exception as e:
        st.error(f"Ошибка получения текста документа: {e}")
        return None


def match_chunks_with_reranker(doc_chunks: List[Dict], template_chunks: List[Dict]) -> List[Dict]:
    """
    Сопоставление чанков документа с чанками шаблона через reranker.

    Args:
        doc_chunks: чанки документа
        template_chunks: чанки шаблона

    Returns:
        List[Dict]: список сопоставлений с ключами template_chunk, doc_chunk, score
    """
    if not doc_chunks or not template_chunks:
        return []

    try:
        reranker = DocumentReranker(device="cpu")
        if not reranker.is_available():
            # Fallback: простое сопоставление по порядку
            matches = []
            for i, tmpl_chunk in enumerate(template_chunks):
                doc_chunk = doc_chunks[i] if i < len(doc_chunks) else None
                if doc_chunk:
                    matches.append({
                        "template_chunk": tmpl_chunk,
                        "doc_chunk": doc_chunk,
                        "score": 0.5
                    })
            return matches

        # Используем reranker для сопоставления
        matches = []
        for tmpl_chunk in template_chunks:
            best_match = None
            best_score = 0.0

            # Ищем лучший док-чанк для каждого чанка шаблона
            for doc_chunk in doc_chunks:
                pairs = [[tmpl_chunk["text"], doc_chunk["text"]]]
                scores = reranker.model.predict(pairs, show_progress_bar=False)
                # sigmoid normalization
                score = float(1 / (1 + np.exp(-scores[0])))

                if score > best_score:
                    best_score = score
                    best_match = doc_chunk

            if best_match:
                matches.append({
                    "template_chunk": tmpl_chunk,
                    "doc_chunk": best_match,
                    "score": best_score
                })

        return matches
    except Exception as e:
        st.error(f"Ошибка сопоставления чанков: {e}")
        return []


def highlight_text_diff(text1: str, text2: str) -> str:
    """
    Пословное сравнение текстов с подсветкой различий желтым цветом.

    Args:
        text1: текст шаблона
        text2: текст документа

    Returns:
        str: HTML с подсветкой различий
    """
    import difflib

    words1 = text1.split()
    words2 = text2.split()

    diff = difflib.SequenceMatcher(None, words1, words2)
    result = []

    for tag, i1, i2, j1, j2 in diff.get_opcodes():
        if tag == 'equal':
            result.append(' '.join(words2[j1:j2]))
        elif tag == 'replace':
            result.append(
                f'<mark style="background-color: yellow;">{" ".join(words2[j1:j2])}</mark>')
        elif tag == 'insert':
            result.append(
                f'<mark style="background-color: yellow;">{" ".join(words2[j1:j2])}</mark>')
        elif tag == 'delete':
            # Удаленные слова не показываем в документе
            pass

    return ' '.join(result)


def perform_standard_audit(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Выполнение анализа стандартности документа.

    Args:
        doc_id: Идентификатор документа

    Returns:
        Dict: Результаты анализа с ключами 'status', 'findings', 'document_text'
    """
    try:
        # Получаем полный текст документа
        document_text = get_document_full_text(doc_id)
        if not document_text:
            return {
                'status': STATUS_UNKNOWN,
                'findings': [],
                'document_text': '',
                'error': 'Не удалось получить текст документа'
            }

        # Получаем информацию о документе для определения типа шаблона
        doc_data = get_document_data(doc_id)
        template_hint = None
        if doc_data and doc_data.get('template_name'):
            # Извлекаем тип шаблона из названия
            template_name = doc_data['template_name']
            if 'поставк' in template_name.lower():
                template_hint = 'Поставка'
            elif 'услуг' in template_name.lower():
                template_hint = 'Услуги'
            elif 'подряд' in template_name.lower():
                template_hint = 'Подряд'
            elif 'рамочн' in template_name.lower():
                template_hint = 'Рамочный'

        # Выполняем анализ стандартности с подсказкой типа шаблона
        status, findings = check_contract_standardness(
            document_text, template_hint=template_hint)

        return {
            'status': status,
            'findings': findings,
            'document_text': document_text,
            'error': None
        }
    except Exception as e:
        return {
            'status': STATUS_UNKNOWN,
            'findings': [],
            'document_text': '',
            'error': str(e)
        }


def show_audit_mode():
    """Отображение страницы анализа стандартности документа."""
    st.title("🔍 Анализ стандартности договора")
    st.markdown("---")

    # Кнопка "Назад"
    if st.button("⬅️ Назад к классификации", help="Вернуться к основному приложению"):
        st.session_state.audit_mode = False
        if 'audit_results' in st.session_state:
            del st.session_state.audit_results
        st.rerun()

    # Проверяем, есть ли результаты анализа
    if 'audit_results' not in st.session_state:
        st.error("🔴 Нет данных для анализа стандартности.")
        return

    doc_data = st.session_state.selected_doc
    audit_results = st.session_state.audit_results

    # Объединенный раздел информации
    st.subheader("📄 Информация")

    # Информация о документе
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Название:** {doc_data['doc_name']}")
        st.write(f"**ID документа:** `{doc_data['doc_id'][:8]}...`")

    with col2:
        st.write(f"**Шаблон:** {doc_data['template_name'] or 'Не выбран'}")
        similarity_display = f"{doc_data['similarity_score']*100:.1f}%" if doc_data['similarity_score'] else "0%"
        st.write(f"**Схожесть с шаблоном:** {similarity_display}")

    st.markdown("---")

    # Заключение о стандартности (перенесено в раздел "Информация")
    if audit_results.get('error'):
        st.error(f"🔴 Ошибка анализа: {audit_results['error']}")
        return

    status = audit_results['status']
    findings = audit_results.get('findings', [])

    # Отображение статуса с соответствующим цветом
    st.write("**🎯 Заключение о стандартности:**")
    if status == STATUS_STANDARD:
        st.success(f"🟢 **{status}**")
        st.info(
            "Документ соответствует стандартной форме договора. Нарушений стандартных условий не обнаружено.")
    elif status == STATUS_NONSTANDARD:
        st.error(f"🔴 **{status}**")
        st.warning(
            "Документ содержит отклонения от стандартной формы договора. Обнаружены нарушения стандартных условий.")
    else:  # STATUS_UNKNOWN
        st.info(f"🔴 **{status}**")
        st.warning(
            "Не удалось определить соответствие стандартной форме из-за недостатка информации или особенностей текста.")

    # Сводка (перенесена в раздел "Информация")
    if findings:
        st.write("**📊 Сводка:**")
        col_summary1, col_summary2, col_summary3 = st.columns(3)

        with col_summary1:
            st.metric("Общее количество нарушений", len(findings))

        # Категоризация нарушений по severity и rule_name
        categories = {}
        severity_counts = {'critical': 0, 'major': 0, 'minor': 0}

        for finding in findings:
            # Подсчет по severity
            severity = finding.get('severity', 'minor')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Подсчет по категориям
            rule_name = finding.get('rule_name', 'Прочие')
            categories[rule_name] = categories.get(rule_name, 0) + 1

        with col_summary2:
            st.write("**По критичности:**")
            if severity_counts['critical'] > 0:
                st.write(f"🔴 Критические: {severity_counts['critical']}")
            if severity_counts['major'] > 0:
                st.write(f"🟠 Важные: {severity_counts['major']}")
            if severity_counts['minor'] > 0:
                st.write(f"🟡 Незначительные: {severity_counts['minor']}")

        with col_summary3:
            if categories:
                st.write("**По категориям:**")
                # Показываем первые 3 категории
                for category, count in list(categories.items())[:3]:
                    st.write(f"• {category}: {count}")
    else:
        st.write("**📊 Сводка:**")
        st.success(
            "🟢 Нарушений не обнаружено - документ полностью соответствует стандартной форме!")

    st.markdown("---")

    # Критерии оценки - выпадающее меню
    with st.expander("📋 Критерии оценки", expanded=False):
        st.markdown("""
        ### 🔍 **Чек-лист проверки стандартности**
        
        #### **1. Форма и шаблон договора**
        - **Стандартно:** используется утверждённая форма без изменений, допускаются только оговорённые допустимые изменения (см. п. 2–4 ниже)
        - **Нестандартно:** протокол разногласий/внесение изменений не из перечня допустимых; договор по форме контрагента
        - **Сигналы:** упоминания «протокол разногласий», «по форме контрагента», «не по форме, утверждённой...»
        
        #### **2. Срок действия договора**
        - **Стандартно:** срок не более 3 лет, фиксирован конечной календарной датой; без бессрочности и автопролонгации
        - **Нестандартно:** срок > 3 лет, бессрочно, «до полного исполнения», автоматическая пролонгация
        - **Исключения:** продление после 3 лет допускается как отдельная пролонгация с согласованием/архивированием в EDM
        
        #### **3. Срок оплаты**
        - **Стандартно:** ≥ 60 календарных дней с момента поставки/работ/услуг (если иное не предписано законом РФ)
        - **Нестандартно:** < 60 календарных дней и нет ссылки на норму закона как исключение
        
        #### **4. Предоплата (аванс)**
        - **Стандартно в рамках исключения:** предоплата возможна только при соблюдении требований п. 6.6: одобрения по Положению о полномочиях, действующая аккредитация, банковская гарантия по порогам, вовлечение юрподдержки/комплаенса/казначейства и др.
        - **Нестандартно:** предоплата без выполнения условий 6.6 (нет БГ при пороге, нет одобрений/аккредитации и пр.)
        
        #### **5. Срок приёмки работ/услуг**
        - **Стандартно:** ≥ 5 рабочих дней; допускается фактическое принятие раньше. Применяется, когда предмет — работы/услуги
        - **Нестандартно:** < 5 рабочих дней или отсутствие условия при наличии работ/услуг
        
        #### **6. Платёжный день**
        - **Стандартно:** один платёжный день в неделю по внутренним процедурам Компании
        - **Нестандартно:** иная периодичность платежей или отсутствие условия, если оно требуемо корпоративным стандартом
        
        #### **7. Допустимые изменения в самом договоре (чтобы он оставался «стандартным»)**
        **Допустимы только:**
        - сокращение срока оплаты (или включение предоплаты) — при выполнении условий 6.6
        - сокращение сроков приёмки, но не менее 5 РД
        - введение ответственности Заказчика, симметричной ответственности Поставщика за сроки
        - изменение/уточнение гарантийного срока и порядка исчисления
        
        **Нестандартно:** любые иные изменения базовых условий
        
        #### **8. Допустимые изменения в допсоглашении (чтобы оно оставалось «стандартным»)**
        **Разрешены:** лимит (для рамочного договора), срок действия, сокращение срока оплаты/предоплата (с 6.6), увеличение срока приёмки, изменение сроков поставки/услуг/работ, реквизиты сторон и договора, добавление симметричной ответственности Заказчика
        
        **Нестандартно:** выход за указанный перечень
        
        #### **9. Стандартная спецификация/приложение/ТЗ**
        **Должны конкретизировать:** предмет, цену, сроки поставки/работ/услуг, порядок исполнения иных обязательств (если отнесено к спецификации договором), условия/адреса доставки/оказания услуг, допускаемые условия по оплате/предоплате (с 6.6), симметричную ответственность Заказчика за просрочку оплаты, гарантийный срок, а также переносить условия из договора без изменений
        
        **Нестандартно:** содержит условия, не предусмотренные договором, либо отличные от допустимых для стандартной спецификации
        
        #### **10. Рамочный договор**
        Признаётся рамочным при описанной в документе природе правоотношений; изменение лимитов и конкретизация условий допускаются через стандартные допы/спеки (см. п. 8–9)
        """)

    # Детали нарушений (если есть)
    if findings:
        st.markdown("---")
        st.subheader("🚨 Выявленные нарушения")

        # Группируем findings по severity для лучшей организации
        critical_findings = [
            f for f in findings if f.get('severity') == 'critical']
        major_findings = [f for f in findings if f.get('severity') == 'major']
        minor_findings = [f for f in findings if f.get('severity') == 'minor']

        # Сначала показываем критические
        if critical_findings:
            st.error("**🔴 Критические нарушения:**")
            for finding in critical_findings:
                with st.expander(f"{finding['rule_id']}: {finding['rule_name']}", expanded=True):
                    # Показываем абзац с HTML подсветкой
                    st.markdown(finding.get('paragraph_html', finding.get(
                        'paragraph_text', '')), unsafe_allow_html=True)

                    # Показываем дополнительные детали, если есть
                    if finding.get('details'):
                        st.json(finding['details'])

        # Затем важные
        if major_findings:
            st.warning("**🟠 Важные нарушения:**")
            for finding in major_findings:
                with st.expander(f"{finding['rule_id']}: {finding['rule_name']}", expanded=False):
                    # Показываем абзац с HTML подсветкой
                    st.markdown(finding.get('paragraph_html', finding.get(
                        'paragraph_text', '')), unsafe_allow_html=True)

                    # Показываем дополнительные детали, если есть
                    if finding.get('details'):
                        st.json(finding['details'])

        # И незначительные
        if minor_findings:
            st.info("**🟡 Незначительные нарушения:**")
            for finding in minor_findings:
                with st.expander(f"{finding['rule_id']}: {finding['rule_name']}", expanded=False):
                    # Показываем абзац с HTML подсветкой
                    st.markdown(finding.get('paragraph_html', finding.get(
                        'paragraph_text', '')), unsafe_allow_html=True)

                    # Показываем дополнительные детали, если есть
                    if finding.get('details'):
                        st.json(finding['details'])

    st.markdown("---")

    # Новый раздел: Детальное сравнение документа и шаблона
    with st.expander("📊 Детальное сравнение документа и шаблона", expanded=False):
        if not doc_data.get('template_id'):
            st.warning(
                "🟡 Для детального сравнения необходимо выбрать и подтвердить шаблон")
        else:
            # Таймер для оценки времени обработки
            start_time = time.time()

            with st.spinner("Загрузка и сопоставление чанков..."):
                # Получаем чанки
                db = DB()
                doc_chunks = get_document_chunks_from_db(
                    db, doc_data['doc_id'])
                template_chunks = get_template_chunks_from_db(
                    db, doc_data['template_id'])

                if not doc_chunks or not template_chunks:
                    st.error("🔴 Не удалось загрузить чанки для сравнения")
                else:
                    # Сопоставляем чанки через reranker
                    matches = match_chunks_with_reranker(
                        doc_chunks, template_chunks)

                    # Вычисляем время обработки
                    processing_time = time.time() - start_time

                    if not matches:
                        st.error("🔴 Не удалось сопоставить чанки")
                    else:
                        # Отображаем время обработки
                        st.success(
                            f"⏱️ Время обработки: {processing_time:.2f} секунд")

                        # Общая информация о сопоставлении
                        st.info(f"📊 **Статистика сопоставления:**")
                        col_info1, col_info2, col_info3 = st.columns(3)

                        with col_info1:
                            st.metric("Чанков документа", len(doc_chunks))
                        with col_info2:
                            st.metric("Чанков шаблона", len(template_chunks))
                        with col_info3:
                            st.metric("Сопоставлений", len(matches))
                        # Фильтр: показывать только чанки с расхождениями
                        show_only_diff = st.checkbox(
                            "Показать только чанки с расхождениями", value=True)

                        st.write(
                            f"**Всего пар для сравнения:** {len(matches)}")

                        # Хранилище для отредактированных чанков
                        if 'edited_chunks' not in st.session_state:
                            st.session_state.edited_chunks = {}

                        # Счетчик для отображаемых чанков
                        displayed_count = 0

                        # Проходим по всем сопоставлениям
                        for idx, match in enumerate(matches):
                            tmpl_chunk = match['template_chunk']
                            doc_chunk = match['doc_chunk']
                            score = match['score']

                            # Проверяем, есть ли расхождения
                            has_diff = tmpl_chunk['text'].strip(
                            ) != doc_chunk['text'].strip()

                            # Фильтрация
                            if show_only_diff and not has_diff and score > 0.95:
                                continue

                            # Проверяем, есть ли нарушения в этом чанке
                            violations_for_chunk = []
                            for finding in findings:
                                # Сопоставляем по тексту параграфа
                                para_text = finding.get('paragraph_text', '')
                                if para_text and para_text in doc_chunk['text']:
                                    violations_for_chunk.append(finding)

                            displayed_count += 1

                            # Заголовок чанка с дополнительной информацией
                            heading = tmpl_chunk.get(
                                'heading', f'Чанк {idx + 1}')
                            st.markdown(f"### 📋 {heading}")

                            # Информация о сопоставлении
                            col_score, col_diff = st.columns([1, 2])
                            with col_score:
                                st.caption(f"🎯 Соответствие: {score*100:.1f}%")
                            with col_diff:
                                if has_diff:
                                    st.caption("⚠️ Обнаружены различия")
                                else:
                                    st.caption("✅ Тексты идентичны")

                            # 3 колонки с полным отображением
                            col1, col2, col3 = st.columns([1, 1, 1])

                            # Колонка 1: Шаблон (полный текст)
                            with col1:
                                st.markdown("**🔹 Текст шаблона:**")
                                st.text_area(
                                    "Шаблон",
                                    value=tmpl_chunk['text'],
                                    height=300,
                                    key=f"template_{idx}",
                                    disabled=True,
                                    label_visibility="collapsed",
                                    help="Полный текст чанка из выбранного шаблона"
                                )

                                # Дополнительная информация о шаблоне
                                st.caption(
                                    f"📏 Длина: {len(tmpl_chunk['text'])} символов")
                                if tmpl_chunk.get('ord'):
                                    st.caption(
                                        f"🔢 Порядок: {tmpl_chunk['ord']}")

                            # Колонка 2: Документ (полный текст с подсветкой и редактированием)
                            with col2:
                                st.markdown("**📄 Текст документа:**")

                                # Подсветка различий (полный текст)
                                highlighted_text = highlight_text_diff(
                                    tmpl_chunk['text'], doc_chunk['text'])
                                st.markdown(highlighted_text,
                                            unsafe_allow_html=True)

                                st.markdown("**✏️ Редактирование:**")
                                # Получаем текущий текст (отредактированный или оригинальный)
                                chunk_key = doc_chunk['chunk_id']
                                current_text = st.session_state.edited_chunks.get(
                                    chunk_key,
                                    doc_chunk['text']
                                )

                                edited_text = st.text_area(
                                    "Редактировать текст документа",
                                    value=current_text,
                                    height=200,
                                    key=f"edit_{idx}",
                                    label_visibility="collapsed",
                                    help="Отредактируйте текст документа. Изменения будут сохранены при нажатии кнопки 'Сохранить все изменения'"
                                )

                                # Сохраняем изменения в session_state
                                if edited_text != doc_chunk['text']:
                                    st.session_state.edited_chunks[chunk_key] = edited_text

                                # Дополнительная информация о документе
                                st.caption(
                                    f"📏 Длина: {len(doc_chunk['text'])} символов")
                                if doc_chunk.get('ord'):
                                    st.caption(
                                        f"🔢 Порядок: {doc_chunk['ord']}")

                                # Индикатор изменений
                                if edited_text != doc_chunk['text']:
                                    st.success("📝 Текст изменен")

                            # Колонка 3: Нарушения стандартности (полная информация)
                            with col3:
                                st.markdown("**⚠️ Анализ стандартности:**")

                                if violations_for_chunk:
                                    st.error(
                                        f"🔴 Найдено нарушений: {len(violations_for_chunk)}")

                                    for i, violation in enumerate(violations_for_chunk):
                                        with st.expander(f"Нарушение {i+1}: {violation['rule_id']}", expanded=False):
                                            st.error(
                                                f"**Правило:** {violation['rule_name']}")
                                            st.write(
                                                f"**Описание:** {violation.get('description', 'Нет описания')}")

                                            # Показываем полный текст параграфа с нарушением
                                            if violation.get('paragraph_html'):
                                                st.markdown(
                                                    "**Текст с нарушением:**")
                                                st.markdown(
                                                    violation['paragraph_html'], unsafe_allow_html=True)
                                            elif violation.get('paragraph_text'):
                                                st.markdown(
                                                    "**Текст с нарушением:**")
                                                st.text_area(
                                                    "Параграф с нарушением",
                                                    value=violation['paragraph_text'],
                                                    height=150,
                                                    disabled=True,
                                                    key=f"violation_{idx}_{i}",
                                                    label_visibility="collapsed"
                                                )

                                            # Дополнительные детали
                                            if violation.get('details'):
                                                st.json(violation['details'])
                                else:
                                    st.success("✅ Нарушений не обнаружено")
                                    st.info(
                                        "Данный чанк соответствует стандартным требованиям")

                            st.markdown("---")

                        st.info(
                            f"Отображено чанков: {displayed_count} из {len(matches)}")

                        # Кнопка сохранения изменений
                        if st.session_state.edited_chunks:
                            st.markdown("### 💾 Сохранить изменения")
                            col_save1, col_save2 = st.columns([1, 3])

                            with col_save1:
                                if st.button("💾 Сохранить все изменения", type="primary"):
                                    # Подготавливаем данные для сохранения
                                    chunks_to_save = []
                                    for chunk in doc_chunks:
                                        chunk_id = chunk['chunk_id']
                                        edited_text = st.session_state.edited_chunks.get(
                                            chunk_id, chunk['text'])
                                        is_modified = edited_text != chunk['text']

                                        chunks_to_save.append({
                                            "chunk_id": chunk_id,
                                            "ord": chunk['ord'],
                                            "heading": chunk.get('heading', ''),
                                            "original_text": chunk['text'],
                                            "edited_text": edited_text,
                                            "is_modified": is_modified
                                        })

                                    # Сохраняем в БД
                                    if db.save_edited_chunks(doc_data['doc_id'], chunks_to_save):
                                        st.success(
                                            "✅ Изменения успешно сохранены!")
                                        # Очищаем session_state
                                        st.session_state.edited_chunks = {}
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(
                                            "🔴 Ошибка при сохранении изменений")

                            with col_save2:
                                st.info(
                                    f"📝 Изменено чанков: {len(st.session_state.edited_chunks)}")

    st.markdown("---")


# Сводка и заключение теперь находятся в разделе "Информация"

# Старый раздел критериев удален - теперь полный чек-лист показан в верхней части страницы


def show_edit_mode():
    """Отображение режима редактирования документа."""
    st.title("📝 Редактор документов")
    st.markdown("---")

    # Кнопка "Назад"
    if st.button("⬅️ Назад к классификации", help="Вернуться к основному приложению"):
        st.session_state.edit_mode = False
        st.rerun()

    doc_data = st.session_state.selected_doc
    doc_id = doc_data['doc_id']

    # Получаем полные данные документа из БД
    full_doc_data = get_document_data(doc_id)
    if not full_doc_data:
        st.error("🔴 Не удалось загрузить данные документа.")
        return

    # Отображаем информацию о документе
    st.subheader("📄 Информация о документе")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Название:** {full_doc_data['doc_name']}")
        st.write(f"**ID документа:** `{doc_id[:8]}...`")

    with col2:
        st.write(
            f"**Дата обработки:** {full_doc_data['created_at'].strftime('%d.%m.%Y %H:%M') if full_doc_data['created_at'] else 'Неизвестно'}")
        st.write(f"**ID сессии:** `{full_doc_data['session_id'][:8]}...`")

    with col3:
        similarity_display = f"{full_doc_data['similarity_score']*100:.1f}%" if full_doc_data['similarity_score'] else "0%"
        st.write(f"**Схожесть:** {similarity_display}")
        st.write(
            f"**Шаблон:** {full_doc_data['template_name'] or 'Не выбран'}")

    st.markdown("---")

    # Проверяем, есть ли выбранный шаблон
    if not full_doc_data['user_choice_doc_id']:
        st.warning(
            "🟡 Для редактирования необходимо выбрать шаблон в основном приложении.")
        st.info(
            "Вернитесь к основному приложению и подтвердите выбор шаблона кнопкой 'ОК'.")
        return

    # Получаем чанки шаблона
    template_chunks = get_template_chunks(full_doc_data['user_choice_doc_id'])
    if not template_chunks:
        st.warning("🟡 Шаблон не содержит чанков для редактирования.")
        return

    st.subheader("📝 Редактирование документа")
    st.write(
        f"**Шаблон:** {full_doc_data['template_name']} (v{full_doc_data['template_version'] or 'Не указана'})")
    st.write(f"**Количество чанков:** {len(template_chunks)}")

    # Создаем форму для редактирования
    edited_chunks = []

    for i, chunk in enumerate(template_chunks):
        st.markdown(f"### Чанк {i + 1}")

        # Заголовок чанка
        heading = st.text_input(
            f"Заголовок чанка {i + 1}:",
            value=chunk.get("heading", ""),
            key=f"edit_heading_{i}"
        )

        # Текст чанка
        text = st.text_area(
            f"Текст чанка {i + 1}:",
            value=chunk.get("text", ""),
            height=200,
            key=f"edit_text_{i}",
            help="Отредактируйте текст чанка по необходимости"
        )

        edited_chunks.append({
            "heading": heading,
            "text": text
        })

        st.markdown("---")

    # Кнопки управления
    col_save, col_preview, col_cancel = st.columns(3)

    with col_save:
        if st.button("💾 Сохранить изменения", type="primary"):
            if save_edited_document(doc_id, edited_chunks):
                st.success("🟢 Документ успешно сохранен!")
            else:
                st.error("🔴 Ошибка при сохранении документа")

    with col_preview:
        if st.button("👁️ Предварительный просмотр"):
            st.subheader("📋 Предварительный просмотр документа")
            for i, chunk in enumerate(edited_chunks):
                if chunk["heading"]:
                    st.markdown(f"### {chunk['heading']}")
                st.write(chunk["text"])
                if i < len(edited_chunks) - 1:
                    st.markdown("---")

    with col_cancel:
        if st.button("🔴 Отменить изменения"):
            st.rerun()

    # Дополнительная информация
    st.markdown("---")
    st.subheader("ℹ️ Информация")
    st.info("""
    **Инструкции по редактированию:**
    
    • **Заголовки чанков** - используйте для структурирования документа
    • **Текст чанков** - основное содержимое документа
    • **Сохранение** - изменения сохраняются в базе данных
    • **Предварительный просмотр** - позволяет увидеть итоговый документ
    """)

    # Статистика редактирования
    st.subheader("📊 Статистика")

    total_chars = sum(len(chunk["text"]) for chunk in edited_chunks)
    total_words = sum(len(chunk["text"].split()) for chunk in edited_chunks)

    col_stat1, col_stat2, col_stat3 = st.columns(3)

    with col_stat1:
        st.metric("Количество чанков", len(edited_chunks))

    with col_stat2:
        st.metric("Общее количество символов", total_chars)

    with col_stat3:
        st.metric("Общее количество слов", total_words)


def main():
    """Главная функция Streamlit приложения с многопользовательским режимом."""
    st.set_page_config(
        page_title="Сравнение договоров с применением ИИ",
        page_icon="📄",
        layout="wide"
    )

    # Получаем уникальный идентификатор сессии
    session_id = get_session_id()

    # Проверяем, находимся ли мы в режиме анализа стандартности
    if st.session_state.get('audit_mode', False) and 'selected_doc' in st.session_state:
        show_audit_mode()
        return

    # Проверяем, находимся ли мы в режиме редактирования
    if st.session_state.get('edit_mode', False) and 'selected_doc' in st.session_state:
        show_edit_mode()
        return

    # st.title("👥 Система классификации документов (Многопользовательский режим)")
    st.title("Классификатор документов")
    st.subheader("Система для определения стандартности договоров")
    st.markdown("---")

    # # Отображение информации о сессии
    # with st.expander("Информация о сессии", expanded=False):
    #     st.write(f"**ID сессии:** `{session_id}`")
    #     st.write(
    #         "**Описание:** Каждый пользователь имеет уникальную сессию с изолированными данными.")
    #     st.write(
    #         "**Безопасность:** Данные разных пользователей полностью разделены в базе данных.")

    # Боковая панель для загрузки файлов
    with st.sidebar:
        # Логотип компании (отцентрированный)
        st.markdown('<div style="text-align: center;">',
                    unsafe_allow_html=True)
        st.image("logo/logo_w.svg", width=300)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

        # st.header("🔄 Загрузка документов")
        st.info("**Загрузка документов**")

        uploaded_files = st.file_uploader(
            "Выберите документы для классификации",
            type=['docx', 'pdf', 'txt'],
            accept_multiple_files=True,
            help="Поддерживаемые форматы: .docx, .pdf, .txt"
        )

        if uploaded_files:
            st.write(f"📁 Загружено файлов: {len(uploaded_files)}")
            for file in uploaded_files:
                st.write(f"• {file.name}")
        # 🚀
        process_button = st.button(
            "**Обработать документы**",
            disabled=not uploaded_files,
            type="primary"
        )
        # Для управления данными разблокировать

        # Кнопка очистки данных сессии
        st.markdown("---")
        # st.subheader("🗑️ Управление данными")
        if st.button("**Очистить данные сессии**", help="Удалить все данные текущей сессии"):
            if clear_session_data(session_id):
                st.rerun()

    # Основная область
    col1, col2 = st.columns([2, 1])

    with col1:
        # st.header("📊 Результаты классификации")
        st.info("ℹ️ Результаты классификации")

        # Если нажата кнопка обработки
        if process_button and uploaded_files:
            # Создаем временную директорию
            with tempfile.TemporaryDirectory() as temp_dir:
                # Сохраняем загруженные файлы
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Запускаем классификацию с привязкой к сессии
                if run_classification(uploaded_files, temp_dir, session_id):
                    st.rerun()  # Перезагружаем страницу для отображения новых результатов

        # Получаем и отображаем результаты из БД для текущей сессии
        results = get_classification_results(session_id)

        if results:
            # Создаем DataFrame для отображения
            df_data = []
            for result in results:
                similarity_score = result.get('similarity_score')
                status, color = classify_similarity(similarity_score)

                df_data.append({
                    "📄 Документ": result['doc_name'],
                    "📅 Дата обработки": result['created_at'].strftime("%d.%m.%Y %H:%M") if result['created_at'] else "",
                    "📈 Схожесть": f"{similarity_score*100:.1f}%" if similarity_score else "0%",
                    "🎯 Статус": status,
                    "📋 Шаблон": result['template_name'] or "Не найден",
                    "🆔 ID документа": result['doc_id'][:8] + "..." if result['doc_id'] else ""
                })

            df = pd.DataFrame(df_data)

            # Отображаем таблицу
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "📈 Схожесть": st.column_config.TextColumn(
                        "📈 Схожесть",
                        help="Процент схожести с найденным шаблоном"
                    ),
                    "🎯 Статус": st.column_config.TextColumn(
                        "🎯 Статус",
                        help="Статус классификации документа"
                    )
                }
            )

            # Детальная информация о последних документах
            st.subheader("🔍 Детальная информация")

            # Получаем все шаблоны для выпадающего меню
            all_templates = get_all_templates()
            template_options = {"---": None}
            for template in all_templates:
                template_options[f"{template['name']} (v{template['version']})"] = template['id']

            # Показываем только первые 5
            for i, result in enumerate(results[:5]):
                similarity_score = result.get('similarity_score')
                status, color = classify_similarity(similarity_score)

                with st.expander(f"📄 {result['doc_name']}", expanded=(i == 0)):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.write("**Основная информация:**")
                        st.write(
                            f"• Дата обработки: {result['created_at'].strftime('%d.%m.%Y %H:%M') if result['created_at'] else 'Неизвестно'}")
                        st.write(f"• ID документа: `{result['doc_id']}`")
                        st.write(
                            f"• ID сессии: `{result.get('session_id', 'N/A')}`")

                        if similarity_score:
                            st.metric(
                                "Схожесть с шаблоном",
                                f"{similarity_score*100:.1f}%",
                                delta=None
                            )
                        else:
                            st.metric("Схожесть с шаблоном", "0%")

                    with col_b:
                        st.write("**Найденный шаблон:**")
                        if result['template_name']:
                            st.write(f"• Название: {result['template_name']}")
                            st.write(
                                f"• ID шаблона: `{result['template_id']}`")
                            # st.write(
                            #     f"• Версия: {result['template_version'] or 'Не указана'}")
                        else:
                            st.write("• Подходящий шаблон не найден")

                    # Статус с цветом
                    if color == "green":
                        st.success(status)
                    elif color == "orange":
                        st.warning(status)
                    elif color == "red":
                        st.error(status)
                    else:
                        st.info(status)

                    # Выпадающее меню выбора шаблона
                    st.markdown("---")
                    st.write("**🎯 Подтверждение шаблона:**")

                    # Определяем значение по умолчанию
                    default_key = "---"
                    if result['template_name'] and result['template_id']:
                        # Ищем соответствующий ключ в template_options
                        for key, value in template_options.items():
                            if value == result['template_id']:
                                default_key = key
                                break

                    selected_template_key = st.selectbox(
                        "Выберите шаблон для подтверждения:",
                        options=list(template_options.keys()),
                        index=list(template_options.keys()).index(default_key),
                        key=f"template_select_{result['doc_id']}",
                        help="Выберите шаблон, который соответствует данному документу"
                    )

                    selected_template_id = template_options[selected_template_key]

                    # Кнопки подтверждения и продолжения
                    col_btn1, col_btn2 = st.columns(2)

                    with col_btn1:
                        if st.button(
                            "🟢 ОК",
                            key=f"confirm_{result['doc_id']}",
                            help="Нажимая кнопку вы подтверждаете шаблон, дальнейшая работа будет основана с применением выбранного шаблона"
                        ):
                            if selected_template_id:
                                # Пересчитываем процент соответствия для выбранного шаблона
                                with st.spinner("Пересчет процента соответствия..."):
                                    new_similarity = recalculate_similarity_for_template(
                                        result['doc_id'], selected_template_id
                                    )

                                if new_similarity is not None:
                                    # Обновляем выбор пользователя и similarity_score
                                    if update_user_choice(result['doc_id'], selected_template_id, new_similarity):
                                        st.success(
                                            f"🟢 Шаблон подтвержден для документа {result['doc_name']}")
                                        st.info(
                                            f"Новый процент соответствия: {new_similarity*100:.1f}%")
                                        st.rerun()
                                    else:
                                        st.error(
                                            "🔴 Ошибка при сохранении выбора")
                                else:
                                    st.error(
                                        "🔴 Ошибка при пересчете процента соответствия")
                            else:
                                st.warning(
                                    "🟡 Выберите шаблон для подтверждения")

                    with col_btn2:
                        # Проверяем, есть ли подтвержденный шаблон для этого документа
                        doc_has_confirmed_template = result.get(
                            'user_choice_doc_id') is not None

                        if st.button(
                            "➡️ ПРОДОЛЖИТЬ",
                            key=f"continue_{result['doc_id']}",
                            disabled=not doc_has_confirmed_template,
                            help="Перейти к анализу стандартности договора" if doc_has_confirmed_template else "Сначала подтвердите выбор шаблона кнопкой 'ОК'"
                        ):
                            # Сохраняем данные документа в session_state
                            st.session_state.selected_doc = {
                                'doc_id': result['doc_id'],
                                'doc_name': result['doc_name'],
                                'template_id': result.get('user_choice_doc_id'),
                                'template_name': result.get('template_name'),
                                'similarity_score': similarity_score,
                                'session_id': result.get('session_id')
                            }

                            # Выполняем анализ стандартности
                            with st.spinner("Выполняется анализ стандартности договора..."):
                                audit_results = perform_standard_audit(
                                    result['doc_id'])
                                if audit_results:
                                    st.session_state.audit_results = audit_results
                                    # Переключаемся в режим анализа стандартности
                                    st.session_state.audit_mode = True
                                    st.rerun()
                                else:
                                    st.error(
                                        "🔴 Ошибка при выполнении анализа стандартности")

        else:
            st.info(
                "📭 Нет данных для отображения. Загрузите и обработайте документы.")

    with col2:
        st.info("**Статистика**")

        if results:
            # Общая статистика
            total_docs = len(results)
            classified_docs = len(
                [r for r in results if r.get('similarity_score')])

            # Объединенная статистика в одной плашке
            if classified_docs > 0:
                success_rate = (classified_docs / total_docs) * 100
                st.info(f"""
                **Всего документов:** {total_docs}
                
                **Классифицировано:** {classified_docs}
                
                **Процент классификации:** {success_rate:.1f}%
                """)
            else:
                st.info(f"""
                **Всего документов:** {total_docs}
                
                **Классифицировано:** {classified_docs}
                """)

                # Распределение по категориям
                categories = {
                    "Соответствует (90-100%)": 0,
                    "Вероятно соответствует (80-90%)": 0,
                    "Не соответствует (0-80%)": 0,
                    "Неопределенно": 0
                }

                for result in results:
                    score = result.get('similarity_score')
                    if score is None or score == 0:
                        categories["Неопределенно"] += 1
                    elif 0.9 <= score <= 1.0:
                        categories["Соответствует (90-100%)"] += 1
                    elif 0.8 <= score < 0.9:
                        categories["Вероятно соответствует (80-90%)"] += 1
                    else:
                        categories["Не соответствует (0-80%)"] += 1

                st.info("Распределение")
                for category, count in categories.items():
                    if count > 0:
                        percentage = (count / total_docs) * 100
                        st.write(
                            f"{category}: **{count}** ({percentage:.1f}%)")

        # Информация о системе
        st.info("**Информация**")
        st.info("""
        **Критерии классификации:**\n

        • **90-100%** - Документ соответствует найденному шаблону\n
        • **80-90%** - Документ скорее всего соответствует предложенным шаблонам\n  
        • **0-80%** - Документ скорее всего не соответствует ни одному шаблону\n
        • **0%** - Затрудняюсь ответить\n
        
        **Особенности:** Система всегда находит наиболее похожий шаблон с реальным процентом схожести.\n
        """)

        # st.info("Настройки")
        # st.write("**Модель:** SBERT Large NLU RU")
        # st.write("**Размерность эмбеддингов:** 1024")
        # st.write("**Режим:** Многопользовательский")

        # # Кнопка обновления данных
        # if st.button("Обновить данные", help="Загрузить последние результаты из базы данных"):
        #     st.rerun()

        # Информация о сессии
        st.info("**Информация о сессии**")
        st.info(f"**ID сессии:** `{session_id}`")
        # st.write(f"**Документов в сессии:** {len(results) if results else 0}")


if __name__ == "__main__":
    main()
