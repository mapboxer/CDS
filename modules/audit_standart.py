"""
Module: contract_standard_checker
Description: Provides functionality to determine if a given contract text follows the standard form.
The module analyzes the text of a contract and checks it against a set of predefined rules/conditions.
It returns a status (СТАНДАРТНЫЙ, НЕ СТАНДАРТНЫЙ, НЕ МОГУ ОПРЕДЕЛИТЬ) and a list of any violated conditions with their locations in the text.
"""
import re
import datetime

# Define possible output statuses
STATUS_STANDARD = "СТАНДАРТНЫЙ"
STATUS_NONSTANDARD = "НЕ СТАНДАРТНЫЙ"
STATUS_UNKNOWN = "НЕ МОГУ ОПРЕДЕЛИТЬ"

# Define month name mapping for parsing dates (genitive case in Russian to month number)
MONTH_MAP = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
}


def check_contract_standardness(contract_text: str):
    """
    Analyze the given contract text and determine if it conforms to the standard form.
    Returns a tuple: (status, violations), where status is one of:
        "СТАНДАРТНЫЙ" (standard),
        "НЕ СТАНДАРТНЫЙ" (non-standard),
        "НЕ МОГУ ОПРЕДЕЛИТЬ" (cannot determine).
    violations is a list of dictionaries, each containing:
        "description": description of the violated condition (in Russian),
        "found_text": excerpt of the contract where the violation was detected.
    If no violations are found, the violations list will be empty.
    """
    text = contract_text
    if text is None or text.strip() == "":
        # If there is no text to analyze, we cannot determine the status
        return STATUS_UNKNOWN, []
    if len(text) < 50:
        # Text is too short to be a full contract – likely insufficient information
        return STATUS_UNKNOWN, []

    text_lower = text.lower()
    violations = []  # List to collect any violation details

    # 1. Проверка формы и шаблона договора
    # Проверка на использование формы контрагента или упоминание протокола разногласий
    form_keywords = [
        "по форме контрагента", "форме поставщика", "форме подрядчика",
        "форме исполнителя", "не по форме, утверждённой"
    ]

    for keyword in form_keywords:
        if keyword in text_lower:
            idx = text_lower.find(keyword)
            sent_start = max(0, text.rfind('.', 0, idx) + 1)
            sent_end = text.find('.', idx)
            if sent_end == -1:
                sent_end = min(idx + 100, len(text))
            violations.append({
                "description": f"Договор составлен по форме контрагента, а не по утвержденной стандартной форме",
                "found_text": text[sent_start:sent_end].strip()
            })
            break

    # 2. Проверка срока действия договора (≤ 3 лет, без автопролонгации и бессрочности).
    term_clause = None
    term_idx = text_lower.find("срок действия")
    if term_idx != -1:
        # Extract the sentence or line containing "срок действия"
        end_idx = text.find('.', term_idx)
        nl_idx = text.find('\n', term_idx)
        if end_idx == -1 or (nl_idx != -1 and nl_idx < end_idx):
            end_idx = nl_idx
        if end_idx == -1:
            end_idx = len(text)
        term_clause = text[term_idx:end_idx].strip()
    else:
        # If "срок действия" not found, look for a phrase like "договор действует ..."
        act_idx = text_lower.find("договор действует")
        if act_idx != -1:
            end_idx = text.find('.', act_idx)
            nl_idx = text.find('\n', act_idx)
            if end_idx == -1 or (nl_idx != -1 and nl_idx < end_idx):
                end_idx = nl_idx
            if end_idx == -1:
                end_idx = len(text)
            term_clause = text[act_idx:end_idx].strip()
    # Analyze the identified term clause
    if term_clause:
        clause_lower = term_clause.lower()
        # Look for forbidden terms indicating indefinite duration or auto-extension
        if ("бессроч" in clause_lower) or ("автоматическ" in clause_lower) or ("пролонгац" in clause_lower) or ("полного выполн" in clause_lower):
            violations.append({
                "description": "Срок действия договора не ограничен 3 годами (бессрочный или с автоматической пролонгацией)",
                "found_text": term_clause
            })
        else:
            # Check for explicit duration in years or months (but not calendar years like "2027 года")
            # Ищем конструкции типа "5 лет", "36 месяцев", но не "2027 года"
            dur_match = re.search(
                r'(?:^|[^0-9])(\d{1,2})\s*(?:год[а-я]*|лет|месяц[а-я]*)', clause_lower)
            if dur_match:
                num = int(dur_match.group(1))
                unit = dur_match.group(0)  # e.g., "5 лет" or "48 месяцев"
                # Проверяем, что это именно срок, а не календарный год (например "2027 года")
                if num > 100:  # Если число больше 100, это скорее всего календарный год
                    pass  # Игнорируем календарные годы
                elif "месяц" in unit:
                    # Duration given in months
                    if num > 36:
                        violations.append({
                            "description": f"Срок действия договора превышает 3 года ({num} мес.)",
                            "found_text": term_clause
                        })
                else:
                    # Duration given in years
                    if num > 3:
                        violations.append({
                            "description": f"Срок действия договора превышает 3 года ({num} года)",
                            "found_text": term_clause
                        })
            else:
                # Check for an explicit end date in the clause (e.g., "до 31.12.2028")
                date_match = re.search(
                    r'(?:до|по)\s+(\d{1,2}\.\d{1,2}\.(20\d\d))', clause_lower)
                date_obj = None
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        date_obj = datetime.datetime.strptime(
                            date_str, "%d.%m.%Y")
                    except ValueError:
                        date_obj = None
                else:
                    # Try matching a date with month name (e.g., "до 31 декабря 2028")
                    date_match2 = re.search(
                        r'(?:до|по)\s+(\d{1,2})\s+([а-я]+)\s+(20\d\d)', clause_lower)
                    if date_match2:
                        day = int(date_match2.group(1))
                        month_name = date_match2.group(2)
                        year = int(date_match2.group(3))
                        month_num = MONTH_MAP.get(month_name)
                        if month_num:
                            try:
                                date_obj = datetime.datetime(
                                    year, month_num, day)
                            except ValueError:
                                date_obj = None
                if date_obj:
                    # Compare end date with today's date to estimate duration
                    today = datetime.datetime.now()
                    # Проверяем, что срок не превышает 3 года от текущей даты
                    max_allowed_date = today + datetime.timedelta(days=3*365)
                    if date_obj > max_allowed_date:
                        years_diff = (date_obj - today).days / 365
                        violations.append({
                            "description": f"Срок действия договора превышает 3 года ({years_diff:.1f} года до {date_obj.strftime('%d.%m.%Y')})",
                            "found_text": term_clause
                        })
                else:
                    # No explicit duration or date found (term possibly not specified -> considered indefinite)
                    violations.append({
                        "description": "Срок действия договора не указан явно (нет конкретной конечной даты)",
                        "found_text": term_clause
                    })
    else:
        # No contract term clause found at all
        violations.append({
            "description": "Срок действия договора не указан",
            "found_text": ""
        })

    # 3. Проверка срока оплаты (≥ 60 календарных дней, если иное не требует закон) и отсутствия предоплаты.
    payment_term_found = False
    payment_clause = None
    # Regex to find payment term (number of days) - more flexible pattern
    pay_pattern = re.compile(
        r'(?:течение|срок|в)\s+(\d+)\s+(календарн\w*|рабочих?)\s+дн\w*', flags=re.IGNORECASE)
    pay_match = pay_pattern.search(text)
    if pay_match:
        payment_term_found = True
        days = int(pay_match.group(1))
        match_text = pay_match.group(0)
        payment_clause = match_text
        # Determine if "рабочих дней" was specified
        is_working = "рабочих" in match_text.lower()
        if is_working:
            # Convert working days to approximate calendar days (5 work days ≈ 7 calendar days)
            if days < 43:  # 42 рабочие дни ≈ 60 календарных дней
                violations.append({
                    "description": f"Срок оплаты установлен менее 60 календарных дней ({days} рабочих дн.)",
                    "found_text": match_text
                })
        else:
            if days < 60:
                # Check context for references to law (to allow exception if law mandates shorter term)
                context_start = max(0, pay_match.start() - 50)
                context_end = min(len(text), pay_match.end() + 50)
                context = text_lower[context_start:context_end]
                if "закона" in context or " фз" in context or "федерального закона" in context:
                    # If the clause explicitly references a law (RF legislation) for this term, do not flag as violation
                    pass
                else:
                    violations.append({
                        "description": f"Срок оплаты установлен менее 60 календарных дней ({days} дн.)",
                        "found_text": match_text
                    })
    # If no numeric payment term found, search for spelled-out terms
    if not payment_term_found:
        for line in text.splitlines():
            line_low = line.lower()
            if ("оплат" in line_low or "платеж" in line_low or "расчет" in line_low) and ("дней" in line_low or "дня" in line_low):
                if re.search(r'\d', line_low):
                    # skip lines that already contain a digit (would have been caught above)
                    continue
                if "шестидесяти" in line_low:
                    # "шестидесяти дней" (60 days) spelled out
                    payment_term_found = True
                    payment_clause = line.strip()
                    # 60 days (or more) is compliant, so no violation added
                    continue
                # Check for spelled numbers indicating <60 days (e.g., "тридцати дней" = 30 days, "сорока пяти дней" = 45 days)
                if any(word in line_low for word in ["двадцати", "тридцати", "сорока", "пятидесяти"]):
                    payment_term_found = True
                    payment_clause = line.strip()
                    # Determine specific number for description if possible
                    desc = None
                    if "сорока пяти" in line_low:
                        desc = "45 дней"
                    elif "сорока" in line_low and "сорока пяти" not in line_low:
                        desc = "40 дней"
                    elif "тридцати пяти" in line_low:
                        desc = "35 дней"
                    elif "тридцати" in line_low and "тридцати пяти" not in line_low:
                        desc = "30 дней"
                    elif "пятидесяти" in line_low:
                        desc = "50 дней"
                    elif "двадцати" in line_low:
                        desc = "20 дней"
                    violations.append({
                        "description": f"Срок оплаты установлен менее 60 календарных дней ({desc if desc else 'менее 60 дней'})",
                        "found_text": line.strip()
                    })
    if not payment_term_found:
        # If we found no mention of payment timing at all, mark as missing info (non-standard)
        violations.append({
            "description": "Условие о сроке оплаты не найдено",
            "found_text": ""
        })

    # Check for prepayment (предоплата/аванс)
    # Согласно п.4 чек-листа, предоплата допустима только при соблюдении требований п. 6.6
    for m in re.finditer(r'предоплат|аванс', text_lower):
        idx = m.start()
        # Determine sentence or line boundaries around the found word
        sent_start = text.rfind('.', 0, idx)
        line_start = text.rfind('\n', 0, idx)
        start_idx = max(sent_start, line_start)
        if start_idx == -1:
            start_idx = 0
        else:
            start_idx += 1  # skip the delimiter
        sent_end = text.find('.', idx)
        line_end = text.find('\n', idx)
        end_idx = sent_end
        if end_idx == -1 or (line_end != -1 and line_end < end_idx):
            end_idx = line_end
        if end_idx == -1:
            end_idx = len(text)
        sentence = text_lower[start_idx:end_idx] if start_idx < end_idx else text_lower[start_idx:]

        # If the sentence contains negation (e.g., "не предусмотрена предоплата" or "без предоплаты"), skip violation.
        if sentence and (("без предоплат" in sentence) or ("без аванс" in sentence) or ("не " in sentence)):
            continue

        # Check if prepayment conditions p.6.6 are mentioned (банковская гарантия, аккредитация, одобрения)
        p66_keywords = ["банковская гарантия", "банковской гарантии",
                        "аккредитация", "одобрен", "полномочи"]
        has_p66_conditions = any(
            keyword in text_lower for keyword in p66_keywords)

        if has_p66_conditions:
            # Предоплата с упоминанием условий п.6.6 - требует дополнительной проверки
            violations.append({
                "description": "Предоплата требует проверки соблюдения всех условий п.6.6 (банковская гарантия, аккредитация, одобрения и др.)",
                "found_text": text[start_idx:end_idx].strip()
            })
        else:
            # Предоплата без упоминания условий п.6.6 - нарушение
            violations.append({
                "description": "Предусмотрена предоплата без соблюдения требований п.6.6 (отсутствуют банковская гарантия, аккредитация, одобрения)",
                "found_text": text[start_idx:end_idx].strip()
            })
        break  # one instance is enough to flag

    # 4. Проверка срока приемки работ/услуг (≥ 5 рабочих дней).
    # Only enforce if contract involves works/services (Компания как заказчик услуг/работ).
    has_services = ("услуг" in text_lower or "работ" in text_lower)
    if has_services:
        accept_pattern = re.compile(
            r'(?:прием|акт)[^.\n]*?(\d+)\s*рабочих?\s*дн', flags=re.IGNORECASE)
        acc_match = accept_pattern.search(text)
        if acc_match:
            days = int(acc_match.group(1))
            match_text = acc_match.group(0)
            if days < 5:
                violations.append({
                    "description": f"Срок приемки результатов работ/услуг менее 5 рабочих дней ({days} дн.)",
                    "found_text": match_text
                })
        else:
            violations.append({
                "description": "Срок приемки работ/услуг не указан (требуется ≥ 5 рабочих дней)",
                "found_text": ""
            })
    # If the contract is only for goods (no "услуг" or "работ"), the 5-day acceptance rule may not apply – we skip in that case.

    # 5. Проверка условия об одном платежном дне в неделю.
    pay_day_clause_found = False
    for line in text.splitlines():
        if "платеж" in line.lower() and "недел" in line.lower():
            pay_day_clause_found = True
            if ("1 раз в неделю" in line.lower()) or ("один раз в неделю" in line.lower()) or ("один платежный день" in line.lower()):
                # Clause correctly states one payment day per week (standard condition)
                pass  # no violation
            else:
                # Clause is present but altered (e.g., two payment days per week)
                violations.append({
                    "description": "Условие об одном платежном дне в неделю изменено",
                    "found_text": line.strip()
                })
            break
    if not pay_day_clause_found:
        violations.append({
            "description": "В тексте отсутствует условие об одном платежном дне в неделю",
            "found_text": ""
        })

    # 6. Проверка на наличие "протокола разногласий" (признак нестандартного договора).
    if "протокол разноглас" in text_lower:
        idx = text_lower.index("протокол разноглас")
        snippet_start = text.rfind('.', 0, idx)
        snippet_nl = text.rfind('\n', 0, idx)
        start_idx = max(snippet_start, snippet_nl)
        if start_idx == -1:
            start_idx = 0
        else:
            start_idx += 1
        snippet_end = text.find('.', idx)
        snippet_nl2 = text.find('\n', idx)
        end_idx = snippet_end
        if end_idx == -1 or (snippet_nl2 != -1 and snippet_nl2 < end_idx):
            end_idx = snippet_nl2
        if end_idx == -1:
            end_idx = len(text)
        violations.append({
            "description": "Обнаружено упоминание протокола разногласий (нестандартные изменения условий договора)",
            "found_text": text[start_idx:end_idx].strip()
        })

    # 7. Проверка допустимых изменений в самом договоре
    # Допустимы только: сокращение срока оплаты, сокращение сроков приёмки (но не менее 5 РД),
    # введение ответственности Заказчика, изменение гарантийного срока
    unauthorized_changes = []

    # Проверка на недопустимые изменения базовых условий
    change_indicators = [
        "изменение условий", "дополнительные условия", "особые условия",
        "отличающиеся от стандартных", "в отступление от"
    ]

    for indicator in change_indicators:
        if indicator in text_lower:
            idx = text_lower.find(indicator)
            sent_start = max(0, text.rfind('.', 0, idx) + 1)
            sent_end = text.find('.', idx)
            if sent_end == -1:
                sent_end = min(idx + 150, len(text))

            change_text = text_lower[sent_start:sent_end]

            # Проверяем, являются ли изменения допустимыми
            allowed_changes = [
                "срок оплат", "срок приемки", "ответственность заказчика",
                "гарантийн", "гарантий"
            ]

            is_allowed = any(
                allowed in change_text for allowed in allowed_changes)

            if not is_allowed:
                violations.append({
                    "description": "Обнаружены недопустимые изменения базовых условий договора",
                    "found_text": text[sent_start:sent_end].strip()
                })
                break

    # 8. Проверка типа документа (допсоглашение)
    is_additional_agreement = any(word in text_lower for word in [
        "дополнительное соглашение", "допсоглашение", "доп. соглашение"
    ])

    if is_additional_agreement:
        # Для допсоглашения разрешены: лимит, срок действия, сокращение срока оплаты/предоплата (с 6.6),
        # увеличение срока приёмки, изменение сроков поставки, реквизиты, симметричная ответственность
        allowed_additional_changes = [
            "лимит", "срок действия", "срок оплат", "предоплат", "срок приемки",
            "срок поставки", "реквизиты", "ответственность заказчика"
        ]
        # Эта проверка требует более детального анализа содержания допсоглашения
        # Пока добавляем информационное сообщение
        pass

    # 9. Проверка стандартной спецификации/приложения/ТЗ
    has_specification = any(word in text_lower for word in [
        "спецификация", "приложение", "техническое задание", "тз"
    ])

    if has_specification:
        # Спецификация должна содержать: предмет, цену, сроки, порядок исполнения,
        # условия доставки, допускаемые условия по оплате, гарантийный срок
        spec_required_elements = [
            "предмет", "цена", "срок", "стоимость"
        ]

        missing_elements = [
            elem for elem in spec_required_elements if elem not in text_lower]

        if len(missing_elements) > 2:  # Если отсутствует больше половины обязательных элементов
            violations.append({
                "description": f"В спецификации/приложении отсутствуют обязательные элементы: {', '.join(missing_elements)}",
                "found_text": "Спецификация неполная"
            })

    # 10. Проверка рамочного договора
    is_framework_contract = any(phrase in text_lower for phrase in [
        "рамочный договор", "генеральное соглашение", "договор на неопределенный объем"
    ])

    if is_framework_contract:
        # Рамочный договор должен содержать описание природы правоотношений
        # и возможность изменения лимитов через стандартные допы/спеки
        if "лимит" not in text_lower and "объем" not in text_lower:
            violations.append({
                "description": "В рамочном договоре не указаны лимиты или максимальные объемы",
                "found_text": "Отсутствуют лимиты рамочного договора"
            })

    # Determine overall status based on collected violations
    status = STATUS_STANDARD
    if violations:
        status = STATUS_NONSTANDARD

    return status, violations
