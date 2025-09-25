#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для запуска приложений системы классификации документов.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_app(app_name: str, port: int = 8501):
    """Запуск Streamlit приложения."""
    app_path = Path(__file__).parent / app_name

    if not app_path.exists():
        print(f"❌ Файл {app_name} не найден!")
        return

    print(f"🚀 Запуск {app_name} на порту {port}...")
    print(f"📱 Откройте браузер: http://localhost:{port}")
    print("⏹️  Для остановки нажмите Ctrl+C")
    print("-" * 50)

    try:
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.port", str(port),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Приложение остановлено")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")


def main():
    """Главная функция."""
    print("📄 Система классификации документов")
    print("=" * 50)
    print("Выберите приложение для запуска:")
    print("1. app_nw.py - Многопользовательская классификация документов")
    print("2. doc_rebuild_app.py - Редактор документов")
    print("3. app.py - Оригинальное приложение (однопользовательское)")
    print("0. Выход")
    print("-" * 50)

    while True:
        try:
            choice = input("Введите номер (0-3): ").strip()

            if choice == "0":
                print("👋 До свидания!")
                break
            elif choice == "1":
                run_app("app_nw.py", 8501)
                break
            elif choice == "2":
                run_app("doc_rebuild_app.py", 8502)
                break
            elif choice == "3":
                run_app("app.py", 8503)
                break
            else:
                print("❌ Неверный выбор. Попробуйте снова.")

        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
