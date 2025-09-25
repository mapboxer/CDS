#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit приложение для редактирования документов.
Позволяет редактировать документы на основе выбранных шаблонов.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

# Импорт модулей проекта
from modules.db import DB


def get_session_id() -> str:
    """
    Получает или создает уникальный идентификатор сессии для текущего пользователя.
    """
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def get_document_data(doc_id: str) -> Optional[Dict[str, Any]]:
    """Получение данных документа из базы данных."""
    try:
        db = DB()
        # Получаем данные документа
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


def main():
    """Главная функция приложения редактирования документов."""
    st.set_page_config(
        page_title="Редактор документов",
        page_icon="📝",
        layout="wide"
    )

    # Получаем ID сессии
    session_id = get_session_id()

    st.title("📝 Редактор документов")
    st.markdown("---")

    # Кнопка "Назад"
    if st.button("⬅️ Назад к классификации", help="Вернуться к основному приложению"):
        st.rerun()

    # Проверяем, есть ли данные документа в session_state
    if 'selected_doc' not in st.session_state or not st.session_state.selected_doc:
        st.warning("⚠️ Нет выбранного документа для редактирования.")
        st.info(
            "Пожалуйста, выберите документ в основном приложении и нажмите 'ПРОДОЛЖИТЬ'.")
        return

    doc_data = st.session_state.selected_doc
    doc_id = doc_data['doc_id']

    # Получаем полные данные документа из БД
    full_doc_data = get_document_data(doc_id)
    if not full_doc_data:
        st.error("❌ Не удалось загрузить данные документа.")
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
            "⚠️ Для редактирования необходимо выбрать шаблон в основном приложении.")
        st.info(
            "Вернитесь к основному приложению и подтвердите выбор шаблона кнопкой 'ОК'.")
        return

    # Получаем чанки шаблона
    template_chunks = get_template_chunks(full_doc_data['user_choice_doc_id'])
    if not template_chunks:
        st.warning("⚠️ Шаблон не содержит чанков для редактирования.")
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
            key=f"heading_{i}"
        )

        # Текст чанка
        text = st.text_area(
            f"Текст чанка {i + 1}:",
            value=chunk.get("text", ""),
            height=200,
            key=f"text_{i}",
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
                st.success("✅ Документ успешно сохранен!")
            else:
                st.error("❌ Ошибка при сохранении документа")

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
        if st.button("❌ Отменить изменения"):
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


if __name__ == "__main__":
    main()
