"""
Instagram

This module contains an example flow of a Instagram data donation study

Assumptions:
It handles DDPs in the english language with filetype JSON.

Configuration
-------------
The ``extraction`` function is driven by ``port_config.json``.  Generate one with::

    pnpm generate-config instagram

Each extractor function carries its own table config in a ``Table config::``
JSON block inside its docstring.  The generator reads those blocks and
assembles the JSON file.

Platform info::

    {
        "name": "Instagram",
        "filetypes": ["json"],
        "languages": ["en", "nl", "de", "pl", "tr", "ar", "ru", "it", "ro", "es", "sq"],
        "description": "Note that supported DDP language also includes Dutch and probably other languages as well. You get an english DDP regardless of the Dutch language setting. These data donation flows have not been tested yet, if you find anything wrong with them report to datadonation@uu.nl and they will be fixed!",
        "time_last_tested": "not yet implemented"
    }
"""

import functools
import logging
import re
from collections import Counter
from typing import Any, Callable, cast

import pandas as pd

import port.helpers.extraction_helpers as eh
import port.helpers.validate as validate
from port.helpers.extraction_helpers import ZipArchiveReader
from port.helpers.flow_builder import FlowBuilder

from port.helpers.validate import (
    DDPCategory,
    DDPFiletype,
    Language,
)
from port.api.d3i_props import ExtractionResult
from port.api.file_utils import SeekableBinaryReader
from port.helpers.table_extractor import (
    load_port_config,
    run_extraction,
)

logger = logging.getLogger(__name__)

DDP_CATEGORIES = [
    DDPCategory(
        id="json_en",
        ddp_filetype=DDPFiletype.JSON,
        language=Language.EN,
        known_files=[
            "secret_conversations.json",
            "personal_information.json",
            "account_privacy_changes.json",
            "account_based_in.json",
            "recently_deleted_content.json",
            "liked_posts.json",
            "stories.json",
            "profile_photos.json",
            "followers.json",
            "signup_information.json",
            "comments_allowed_from.json",
            "login_activity.json",
            "your_topics.json",
            "camera_information.json",
            "recent_follow_requests.json",
            "devices.json",
            "professional_information.json",
            "follow_requests_you've_received.json",
            "eligibility.json",
            "pending_follow_requests.json",
            "videos_watched.json",
            "account_searches.json",
            "profile_searches.json",
            "recent_searches.json",
            "word_or_phrase_searches.json",
            "followers_1.json",
            "saved_posts.json",
            "following.json",
            "posts_viewed.json",
            "post_comments_1.json",
            "recently_unfollowed_accounts.json",
            "post_comments.json",
            "account_information.json",
            "accounts_you're_not_interested_in.json",
            "liked_comments.json",
            "story_likes.json",
            "threads_viewed.json",
            "use_cross-app_messaging.json",
            "profile_changes.json",
            "reels.json",
        ],
    )
]


# ---------------------------------------------------------------------------
# No URLs that can lead to a person's account
#
# The export does not say whether an account is private, so URLs are removed
# wherever content or accounts of persons may be involved (profiles, posts,
# reels, comments, ...).  They are kept only where the other side is public by
# definition: ads, liked Facebook pages and external websites in the link
# history.
# ---------------------------------------------------------------------------


# ===========================================================================
# Study extraction (Teilnehmendeninformation)
#
# Only the data described in the participant information is extracted.
# Field names differ between export languages (they follow the language of the
# participant's profile), so fields are recognised by label lists in all
# supported languages and, where possible, by structure (e.g. the block that
# contains a username and a profile link is the owner of a post).
# ===========================================================================

_ST_LABELS = {
    "owner": [
        "Owner", "Eigenaar", "Eigentümer", "Inhaber", "Besitzer", "Właściciel", "Sahip", "Sahibi",
        "المالك", "Владелец", "Proprietario", "Proprietar", "Propietario", "Pronari",
        "Story author", "Verfasser der Story", "Auteur van verhaal", "Autor relacji", "Hikaye sahibi",
        "Автор истории", "Autore della storia", "Autorul poveștii", "Autor de la historia", "Autori i stories",
        "Author", "Verfasser/in", "Auteur", "Autor", "Yazar", "الكاتب", "Автор", "Autore", "Autori",
    ],
    "username": [
        "Username", "Gebruikersnaam", "Nutzername", "Benutzername", "Nazwa użytkownika",
        "Kullanıcı adı", "اسم المستخدم", "Имя пользователя", "Nome utente", "Nume de utilizator",
        "Nombre de usuario", "Emri i përdoruesit",
    ],
    "name": ["Name", "Naam", "Nazwa", "Ad", "İsim", "الاسم", "Имя", "Название", "Nome", "Nume", "Nombre", "Emri"],
    "url": ["URL", "Link", "Adres URL", "Bağlantı", "الرابط", "URL-адрес", "Ссылка", "Enlace", "Lidhja"],
    "caption": [
        "Caption", "Bijschrift", "Bildunterschrift", "Beschriftung", "Untertitel", "Podpis", "Açıklama",
        "Altyazı", "التسمية التوضيحية", "Подпись", "Didascalia", "Descriere", "Pie de foto", "Leyenda",
        "Përshkrimi", "Titulli",
    ],
    "title": ["Title", "Titel", "Tytuł", "Başlık", "العنوان", "Название", "Titolo", "Titlu", "Título", "Titulli"],
    "emoji": ["Emoji", "Emoji-Reaktion", "Emoji reaction", "إيموجي", "Эмодзи", "Emoji tepkisi"],
    "comment": [
        "Comment", "Opmerking", "Reactie", "Kommentar", "Komentarz", "Yorum", "تعليق", "Комментарий",
        "Commento", "Comentariu", "Comentario", "Koment",
    ],
    "media_owner": [
        "Media Owner", "Media-eigenaar", "Medieninhaber", "Medieneigentümer", "Właściciel mediów",
        "Medya sahibi", "مالك الوسائط", "Владелец медиафайла", "Proprietario del contenuto multimediale",
        "Proprietarul conținutului media", "Propietario del contenido multimedia", "Pronari i medias",
    ],
    "search": [
        "Search", "Search query", "Zoeken", "Zoekopdracht", "Suche", "Suchanfrage", "Suchbegriff",
        "Wyszukiwanie", "Zapytanie", "Arama", "Arama sorgusu", "بحث", "استعلام البحث", "Поиск",
        "Поисковый запрос", "Ricerca", "Query di ricerca", "Căutare", "Interogare de căutare",
        "Búsqueda", "Consulta de búsqueda", "Kërkimi", "Kërkesa e kërkimit",
    ],
    "time": ["Time", "Tijd", "Zeit", "Uhrzeit", "Godzina", "Czas", "Saat", "الوقت", "Время", "Ora", "Hora", "Koha"],
    "reaction": ["Reaction", "Reactie", "Reaktion", "Reakcja", "Tepki", "تفاعل", "Реакция", "Reazione",
                 "Reacție", "Reacción", "Reagimi"],
    "ad": ["Ad", "Advertentie", "Werbeanzeige", "Anzeige", "Reklama", "Reklam", "إعلان", "Реклама",
           "Inserzione", "Reclamă", "Anuncio", "Reklamë"],
}


def _st_norm(text) -> str:
    return re.sub(r"\s+", " ", _st_fix(text)).strip().casefold()


_ST_LABEL_SETS: dict = {}  # filled on first use


def _st_fix(text) -> str:
    """Repair Meta's double-encoded UTF-8 ("EigentÃ¼mer" -> "Eigentümer")."""
    if text is None:
        return ""
    text = str(text)
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _st_is(label, key: str) -> bool:
    if not _ST_LABEL_SETS:
        _ST_LABEL_SETS.update({k: {_st_norm(v) for v in vs} for k, vs in _ST_LABELS.items()})
    return _st_norm(label) in _ST_LABEL_SETS[key]


def _st_is_epoch(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _st_iso(ts, errors: Counter) -> str:
    return eh.epoch_to_iso(ts, errors=errors) if _st_is_epoch(ts) else ""


def _st_is_url(value) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith(("http://", "https://"))


def _st_is_ad_library_url(value) -> bool:
    return _st_is_url(value) and "/ads/library" in value


def _st_json_all(reader: ZipArchiveReader, errors: Counter, pattern: str) -> list:
    try:
        return [r.data for r in reader.json_all(pattern) if r.data is not None]
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return []


def _st_records(data) -> list:
    """Split a parsed JSON file into records."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if "label_values" in data or "string_map_data" in data or "string_list_data" in data:
        return [data]
    lists = [v for v in data.values() if isinstance(v, list)]
    if len(lists) == 1:
        return lists[0]
    return [data]


def _st_record_ts(record):
    """Timestamp of a record (record field, string_list_data, string_map_data or first timestamp_value)."""
    if not isinstance(record, dict):
        return None
    if _st_is_epoch(record.get("timestamp")):
        return record["timestamp"]
    for item in record.get("string_list_data") or []:
        if isinstance(item, dict) and _st_is_epoch(item.get("timestamp")):
            return item["timestamp"]
    smd = record.get("string_map_data")
    if isinstance(smd, dict):
        for label, item in smd.items():
            if isinstance(item, dict) and _st_is_epoch(item.get("timestamp")) and _st_is(label, "time"):
                return item["timestamp"]
        for item in smd.values():
            if isinstance(item, dict) and _st_is_epoch(item.get("timestamp")):
                return item["timestamp"]
    for entry in record.get("label_values") or []:
        if isinstance(entry, dict) and _st_is_epoch(entry.get("timestamp_value")):
            return entry["timestamp_value"]
    return None


_ST_TEXT_TIME_ZONE = "America/Los_Angeles"
_ST_TIME_TEXT = re.compile(r"^\s*(\d{10}|\d{13})(\.\d+)?\s*$")


def _st_parse_time_text(text):
    """Epoch seconds from a time written as text ("1730801234", "2026-11-05 10:23:12", ...), else None."""
    if not isinstance(text, str) or not re.search(r"\d", text):
        return None
    match = _ST_TIME_TEXT.match(text)
    if match:
        value = float(text)
        return value / 1000 if len(match.group(1)) == 13 else value
    if ":" not in text:
        return None  # a date needs a time of day here; avoids reading titles like "2024"
    try:
        parsed = pd.to_datetime(_st_fix(text), dayfirst="." in text)
        if pd.isna(parsed):
            return None
        if parsed.tzinfo is None:
            # Meta writes these times without a time zone, in US Pacific time
            # (checked against a real export: "1:58am" was late morning in Germany).
            parsed = parsed.tz_localize(_ST_TEXT_TIME_ZONE, ambiguous=True, nonexistent="shift_forward")
    except (ValueError, TypeError, OverflowError):
        return None
    return parsed.timestamp()


def _st_text_time(record: dict):
    """First time written as text in a record (e.g. website session start time in the link history)."""
    for _, entry in _st_top(record):
        value = entry.get("value")
        if _st_is_url(value):
            continue
        ts = _st_parse_time_text(value)
        if ts and ts > 0:
            return ts
    return None


def _st_top(record: dict) -> list:
    """Top-level (label, entry) pairs of a record (label_values or string_map_data)."""
    pairs = []
    for entry in record.get("label_values") or []:
        if isinstance(entry, dict):
            pairs.append((entry.get("label") or entry.get("title") or "", entry))
    smd = record.get("string_map_data")
    if isinstance(smd, dict):
        for label, entry in smd.items():
            if isinstance(entry, dict):
                pairs.append((label, entry))
    return pairs


def _st_top_value(record: dict, key: str) -> str:
    for label, entry in _st_top(record):
        if _st_is(label, key) and entry.get("value") not in (None, ""):
            return _st_fix(entry["value"])
    return ""


def _st_owner(node) -> tuple:
    """Find the owner/author block anywhere in *node*: returns (username, name, profile_url).

    A titled block counts as owner when its title is an owner label, or when
    it contains a username field (structure, independent of language).
    Hashtag blocks (only names, no username/link) are ignored.
    """
    if isinstance(node, list):
        for child in node:
            found = _st_owner(child)
            if any(found):
                return found
        return ("", "", "")
    if not isinstance(node, dict):
        return ("", "", "")
    members = node.get("dict") if isinstance(node.get("dict"), list) else None
    if members is not None and node.get("title") is not None:
        fields = []
        for member in members:
            if isinstance(member, dict) and isinstance(member.get("dict"), list):
                fields.extend(f for f in member["dict"] if isinstance(f, dict))
            elif isinstance(member, dict) and "label" in member:
                fields.append(member)
        has_username = any(_st_is(f.get("label"), "username") for f in fields)
        if _st_is(node.get("title"), "owner") and not has_username:
            # e.g. sticker reactions: "Owner" -> story (URL, caption) -> "Owner" -> account
            nested = _st_owner(members)
            if nested[0]:
                return nested
        if _st_is(node.get("title"), "owner") or has_username:
            username = name = url = ""
            for f in fields:
                value = f.get("value")
                if value in (None, ""):
                    continue
                if _st_is(f.get("label"), "username"):
                    username = username or _st_fix(value)
                elif _st_is_url(value):
                    url = url or value
                elif _st_is(f.get("label"), "name"):
                    name = name or _st_fix(value)
            if username or name or url:
                return (username, name, url)
    for key in ("label_values", "dict", "vec"):
        if key in node:
            found = _st_owner(node[key])
            if any(found):
                return found
    return ("", "", "")


def _st_content_url(record: dict) -> str:
    """Link of the content itself (not the owner's profile, not the Ad Library)."""
    candidates = []
    for label, entry in _st_top(record):
        for key in ("href", "value"):
            value = entry.get(key)
            if _st_is_url(value) and not _st_is_ad_library_url(value):
                candidates.append((0 if _st_is(label, "url") else 1, value))
    for item in record.get("string_list_data") or []:
        if isinstance(item, dict) and _st_is_url(item.get("href")):
            candidates.append((2, item["href"]))
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1] if candidates else ""


def _st_df(rows: list, columns: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame()


def _st_without_empty_rows(fn):
    """Drop rows whose cells are all empty or 0; empty tables are not shown."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        df = fn(*args, **kwargs)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        filled = df.apply(
            lambda row: any(str(v).strip() not in ("", "0", "nan", "None", "NaT") for v in row), axis=1
        )
        df = df[filled].reset_index(drop=True)
        return df if not df.empty else pd.DataFrame()
    return wrapper


_ST_OFF_META_EVENTS = {
    "PAGE_VIEW": "Seitenaufruf",
    "VIEW_CONTENT": "Inhalt angesehen",
    "SEARCH": "Suche",
    "ADD_TO_CART": "In den Warenkorb gelegt",
    "ADD_TO_WISHLIST": "Zur Wunschliste hinzugefügt",
    "INITIATE_CHECKOUT": "Bezahlvorgang begonnen",
    "ADD_PAYMENT_INFO": "Zahlungsinformationen hinzugefügt",
    "PURCHASE": "Kauf",
    "LEAD": "Kontaktanfrage",
    "COMPLETE_REGISTRATION": "Registrierung abgeschlossen",
    "SUBSCRIBE": "Abonniert",
    "START_TRIAL": "Testphase begonnen",
    "ACTIVATE_APP": "App geöffnet",
    "CUSTOM": "Benutzerdefiniertes Ereignis",
    "AD_REQUEST": "Werbeanfrage",
    "AD_IMPRESSION": "Werbeeinblendung",
    "AD_CLICK": "Werbeklick",
    "NOT_AUTHORIZED": "Nicht autorisiert",
}


def _st_off_meta_rows(data, errors: Counter) -> list:
    """Rows (website/app, event, time) from your_activity_off_meta_technologies.json; IDs are skipped."""
    rows = []
    for record in data if isinstance(data, list) else _st_records(data):
        if not isinstance(record, dict):
            continue
        website = _st_fix(record.get("title") or record.get("name") or "")
        for _, entry in _st_top(record):
            for event in entry.get("vec") or []:
                if not isinstance(event, dict):
                    continue
                name, ts = "", None
                for field in event.get("dict") or []:
                    if not isinstance(field, dict):
                        continue
                    if _st_is_epoch(field.get("timestamp_value")):
                        ts = ts or field["timestamp_value"]
                    elif _st_norm(field.get("label")) != "id" and field.get("value") not in (None, ""):
                        name = name or _st_fix(field["value"])
                rows.append((website, _ST_OFF_META_EVENTS.get(name, name), _st_iso(ts, errors)))
        for event in record.get("events") or []:  # older format
            if isinstance(event, dict):
                name = _st_fix(event.get("type", ""))
                rows.append((website, _ST_OFF_META_EVENTS.get(name, name), _st_iso(event.get("timestamp"), errors)))
    return rows


def _st_vec_values(data) -> list:
    """(list label, value) for every element of every ``vec`` list in a label_values file."""
    rows = []
    for record in _st_records(data):
        if not isinstance(record, dict):
            continue
        for label, entry in _st_top(record):
            for item in entry.get("vec") or []:
                if isinstance(item, dict) and item.get("value") not in (None, ""):
                    rows.append((_st_fix(label), _st_fix(item["value"])))
                elif isinstance(item, dict) and isinstance(item.get("dict"), list):
                    values = [_st_fix(f.get("value")) for f in item["dict"]
                              if isinstance(f, dict) and f.get("value") not in (None, "")]
                    if values:
                        rows.append((_st_fix(label), "; ".join(values)))
    return rows

# ---------------------------------------------------------------------------
# Instagram tables of the study
# ---------------------------------------------------------------------------

def _ig_files(reader: ZipArchiveReader, errors: Counter, folder_and_stem: str, *alternatives: str) -> list:
    """``<stem>.json`` and numbered variants, never from the Threads folder.

    Looks in the expected folder first; if Meta moved the file, the same file
    name is accepted in any folder. *alternatives* are older names of the file.
    """
    for candidate in (folder_and_stem,) + alternatives:
        stem = re.escape(candidate)
        found = _st_json_all(reader, errors, rf"^(?!.*threads)(?:.*/)?{stem}(?:_\d+)?\.json$")
        if not found:
            stem = re.escape(candidate.rsplit("/", 1)[-1])
            found = _st_json_all(reader, errors, rf"^(?!.*threads)(?:.*/)?{stem}(?:_\d+)?\.json$")
        if found:
            return found
    return []


def _ig_content_rows(reader, errors, folder_and_stem: str, with_text: bool = True, with_url: bool = True) -> list:
    """(account, [text], [url], date) for label_values files with an owner block."""
    rows = []
    for data in _ig_files(reader, errors, folder_and_stem):
        try:
            for record in _st_records(data):
                if not isinstance(record, dict):
                    continue
                username, name, _ = _st_owner(record)
                row = [username or name or _st_fix(record.get("title") or "")]
                if with_text:
                    row.append(_st_top_value(record, "caption") or _st_top_value(record, "title"))
                if with_url:
                    row.append(_st_content_url(record))
                row.append(_st_iso(_st_record_ts(record), errors))
                rows.append(tuple(row))
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    return rows


def instagram_posts_viewed_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Posts shown in your feed.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``Text``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows posts, videos and reels that were shown in your feed, with the account that posted them, their text (if available), the link and the time.",
          "source_file": "ads_information/ads_and_topics/posts_viewed.json, videos_watched.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "Text": "Text of the content (caption or post text), if the export contains it.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_feed",
          "title": {
            "en": "Posts shown in your feed",
            "nl": "Berichten in je feed",
            "de": "In Ihrem Feed angezeigte Beiträge",
            "pl": "Posty wyświetlone w Twoim kanale",
            "tr": "Akışında gösterilen gönderiler",
            "ar": "المنشورات المعروضة في موجزك",
            "ru": "Публикации в вашей ленте",
            "it": "Post mostrati nel tuo feed",
            "ro": "Postări afișate în fluxul tău",
            "es": "Publicaciones mostradas en tu feed",
            "sq": "Postimet e shfaqura në furnizimin tënd"
          },
          "description": {
            "en": "This table shows posts, videos and reels that were shown in your feed, with the account that posted them, their text (if available), the link and the time.",
            "nl": "Deze tabel toont berichten, video's en reels die in je feed werden getoond, met het account dat ze plaatste, de tekst (indien beschikbaar), de link en het tijdstip.",
            "de": "Diese Tabelle zeigt Beiträge, Videos und Reels, die in Ihrem Feed angezeigt wurden, mit dem Account, der sie veröffentlicht hat, ihrem Text (falls vorhanden), dem Link und dem Zeitpunkt.",
            "pl": "Ta tabela pokazuje posty, filmy i rolki wyświetlone w Twoim kanale wraz z kontem autora, tekstem (jeśli jest dostępny), linkiem i czasem.",
            "tr": "Bu tablo, akışında gösterilen gönderileri, videoları ve reels'leri; paylaşan hesap, metin (varsa), bağlantı ve zaman ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول المنشورات ومقاطع الفيديو والريلز التي ظهرت في موجزك، مع الحساب الناشر والنص (إن وجد) والرابط والوقت.",
            "ru": "В этой таблице показаны публикации, видео и Reels из вашей ленты с аккаунтом автора, текстом (если есть), ссылкой и временем.",
            "it": "Questa tabella mostra post, video e reel mostrati nel tuo feed, con l'account che li ha pubblicati, il testo (se disponibile), il link e l'ora.",
            "ro": "Acest tabel arată postările, videoclipurile și reel-urile afișate în fluxul tău, cu contul care le-a publicat, textul (dacă există), linkul și ora.",
            "es": "Esta tabla muestra publicaciones, vídeos y reels que aparecieron en tu feed, con la cuenta que los publicó, su texto (si está disponible), el enlace y la hora.",
            "sq": "Kjo tabelë tregon postimet, videot dhe reels që u shfaqën në furnizimin tënd, me llogarinë që i postoi, tekstin (nëse ka), lidhjen dhe kohën."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Account",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogaria"
            },
            "Text": {
              "en": "Text",
              "nl": "Tekst",
              "de": "Text",
              "pl": "Tekst",
              "tr": "Metin",
              "ar": "النص",
              "ru": "Текст",
              "it": "Testo",
              "ro": "Text",
              "es": "Texto",
              "sq": "Teksti"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          },
          "visualizations": [
            {
              "title": {
                "en": "The total number of Instagram posts you viewed over time",
                "nl": "Het totale aantal Instagram-berichten dat je in de loop van de tijd hebt bekeken",
                "de": "Die Gesamtzahl der Instagram-Beiträge, die Sie im Laufe der Zeit angesehen haben",
                "pl": "Łączna liczba postów na Instagramie, które wyświetliłeś/aś w czasie",
                "tr": "Zaman içinde görüntülediğin Instagram gönderilerinin toplam sayısı",
                "ar": "إجمالي عدد منشورات إنستغرام التي شاهدتها عبر الوقت",
                "ru": "Общее количество публикаций в Instagram, которые вы просмотрели с течением времени",
                "it": "Il numero totale di post di Instagram che hai visualizzato nel tempo",
                "ro": "Numărul total de postări de pe Instagram pe care le-ai vizualizat în timp",
                "es": "El número total de publicaciones de Instagram que viste a lo largo del tiempo",
                "sq": "Numri total i postimeve në Instagram që ke parë me kalimin e kohës"
              },
              "type": "bar",
              "group": {
                "column": "Date",
                "dateFormat": "auto",
                "label": "Datum"
              },
              "values": [
                {
                  "label": "Anzahl",
                  "aggregate": "count"
                }
              ]
            },
            {
              "title": {
                "en": "The total number of Instagram posts you have viewed per hour of the day",
                "nl": "Het totale aantal Instagram-berichten dat je per uur van de dag hebt bekeken",
                "de": "Die Gesamtzahl der Instagram-Beiträge, die Sie pro Stunde des Tages angesehen haben",
                "pl": "Łączna liczba postów na Instagramie, które wyświetliłeś/aś w poszczególnych godzinach doby",
                "tr": "Günün saatine göre görüntülediğin Instagram gönderilerinin toplam sayısı",
                "ar": "إجمالي عدد منشورات إنستغرام التي شاهدتها في كل ساعة من اليوم",
                "ru": "Общее количество публикаций в Instagram, которые вы просматривали по часам суток",
                "it": "Il numero totale di post di Instagram che hai visualizzato per ogni ora del giorno",
                "ro": "Numărul total de postări de pe Instagram pe care le-ai vizualizat pe oră din zi",
                "es": "El número total de publicaciones de Instagram que viste por hora del día",
                "sq": "Numri total i postimeve në Instagram që ke parë për çdo orë të ditës"
              },
              "type": "bar",
              "group": {
                "column": "Date",
                "dateFormat": "hour_cycle",
                "label": "Tageszeit"
              },
              "values": [
                {
                  "label": "Anzahl"
                }
              ]
            },
            {
              "title": {
                "en": "The 10 accounts whose posts you saw most often",
                "nl": "De 10 accounts waarvan je het vaakst berichten zag",
                "de": "Die 10 Accounts, deren Beiträge Sie am häufigsten gesehen haben",
                "pl": "10 kont, których posty widziałeś/aś najczęściej",
                "tr": "Gönderilerini en sık gördüğün 10 hesap",
                "ar": "الحسابات العشرة التي رأيت منشوراتها أكثر من غيرها",
                "ru": "10 аккаунтов, публикации которых вы видели чаще всего",
                "it": "I 10 account di cui hai visto più spesso i post",
                "ro": "Cele 10 conturi ale căror postări le-ai văzut cel mai des",
                "es": "Las 10 cuentas cuyas publicaciones viste con más frecuencia",
                "sq": "10 llogaritë, postimet e të cilave i ke parë më shpesh"
              },
              "type": "bar",
              "group": {
                "column": "Account",
                "label": "Account",
                "top": 10
              },
              "values": [
                {
                  "label": "Anzahl",
                  "aggregate": "count"
                }
              ]
            }
          ]
        }
    """
    rows = _ig_content_rows(reader, errors, "ads_and_topics/posts_viewed")
    rows += _ig_content_rows(reader, errors, "ads_and_topics/videos_watched")
    rows.sort(key=lambda r: r[3], reverse=True)
    return _st_df(rows, ["Account", "Text", "URL", "Date"])


def instagram_ads_viewed_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Ads shown to you.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``Text``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows ads that were shown to you, with the advertiser, the text (if available), the link and the time.",
          "source_file": "ads_information/ads_and_topics/ads_viewed.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "Text": "Text of the content (caption or post text), if the export contains it.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_ads_viewed",
          "title": {
            "en": "Ads shown to you",
            "nl": "Advertenties die je hebt gezien",
            "de": "Ihnen angezeigte Werbeanzeigen",
            "pl": "Wyświetlone reklamy",
            "tr": "Sana gösterilen reklamlar",
            "ar": "الإعلانات المعروضة لك",
            "ru": "Показанная вам реклама",
            "it": "Inserzioni che ti sono state mostrate",
            "ro": "Reclame afișate ție",
            "es": "Anuncios que se te mostraron",
            "sq": "Reklamat e shfaqura për ty"
          },
          "description": {
            "en": "This table shows ads that were shown to you, with the advertiser, the text (if available), the link and the time.",
            "nl": "Deze tabel toont advertenties die aan je werden getoond, met de adverteerder, de tekst (indien beschikbaar), de link en het tijdstip.",
            "de": "Diese Tabelle zeigt Werbeanzeigen, die Ihnen angezeigt wurden, mit der werbetreibenden Person oder Firma, dem Text (falls vorhanden), dem Link und dem Zeitpunkt.",
            "pl": "Ta tabela pokazuje wyświetlone Ci reklamy wraz z reklamodawcą, tekstem (jeśli jest dostępny), linkiem i czasem.",
            "tr": "Bu tablo, sana gösterilen reklamları; reklamveren, metin (varsa), bağlantı ve zaman ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول الإعلانات التي عُرضت لك، مع المعلن والنص (إن وجد) والرابط والوقت.",
            "ru": "В этой таблице показана реклама, которую вам показывали, с рекламодателем, текстом (если есть), ссылкой и временем.",
            "it": "Questa tabella mostra le inserzioni che ti sono state mostrate, con l'inserzionista, il testo (se disponibile), il link e l'ora.",
            "ro": "Acest tabel arată reclamele afișate ție, cu agentul de publicitate, textul (dacă există), linkul și ora.",
            "es": "Esta tabla muestra los anuncios que se te mostraron, con el anunciante, el texto (si está disponible), el enlace y la hora.",
            "sq": "Kjo tabelë tregon reklamat që të janë shfaqur, me reklamuesin, tekstin (nëse ka), lidhjen dhe kohën."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Account",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogaria"
            },
            "Text": {
              "en": "Text",
              "nl": "Tekst",
              "de": "Text",
              "pl": "Tekst",
              "tr": "Metin",
              "ar": "النص",
              "ru": "Текст",
              "it": "Testo",
              "ro": "Text",
              "es": "Texto",
              "sq": "Teksti"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          },
          "visualizations": [
            {
              "title": {
                "en": "The 10 advertisers whose ads you saw most often",
                "nl": "De 10 adverteerders van wie je het vaakst advertenties zag",
                "de": "Die 10 Werbetreibenden, deren Werbung Sie am häufigsten gesehen haben",
                "pl": "10 reklamodawców, których reklamy widziałeś/aś najczęściej",
                "tr": "Reklamlarını en sık gördüğün 10 reklamveren",
                "ar": "المعلنون العشرة الذين رأيت إعلاناتهم أكثر من غيرهم",
                "ru": "10 рекламодателей, чью рекламу вы видели чаще всего",
                "it": "I 10 inserzionisti di cui hai visto più spesso le inserzioni",
                "ro": "Cei 10 agenți de publicitate ale căror reclame le-ai văzut cel mai des",
                "es": "Los 10 anunciantes cuyos anuncios viste con más frecuencia",
                "sq": "10 reklamuesit, reklamat e të cilëve i ke parë më shpesh"
              },
              "type": "bar",
              "group": {
                "column": "Account",
                "label": "Werbetreibende",
                "top": 10
              },
              "values": [
                {
                  "label": "Anzahl",
                  "aggregate": "count"
                }
              ]
            }
          ]
        }
    """
    rows = _ig_content_rows(reader, errors, "ads_and_topics/ads_viewed")
    return _st_df(rows, ["Account", "Text", "URL", "Date"])


def instagram_stories_viewed_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Stories you viewed.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``Text``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows stories you viewed, with the account that posted them, their text (if available), the link and the time.",
          "source_file": "your_instagram_activity/story_interactions/stories_viewed.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "Text": "Text of the content (caption or post text), if the export contains it.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_stories_viewed",
          "title": {
            "en": "Stories you viewed",
            "nl": "Stories die je hebt bekeken",
            "de": "Angesehene Storys",
            "pl": "Obejrzane relacje",
            "tr": "Görüntülediğin hikayeler",
            "ar": "القصص التي شاهدتها",
            "ru": "Просмотренные истории",
            "it": "Storie che hai visualizzato",
            "ro": "Povești vizualizate",
            "es": "Historias que viste",
            "sq": "Stories që ke parë"
          },
          "description": {
            "en": "This table shows stories you viewed, with the account that posted them, their text (if available), the link and the time.",
            "nl": "Deze tabel toont stories die je hebt bekeken, met het account dat ze plaatste, de tekst (indien beschikbaar), de link en het tijdstip.",
            "de": "Diese Tabelle zeigt Storys, die Sie angesehen haben, mit dem Account, der sie veröffentlicht hat, ihrem Text (falls vorhanden), dem Link und dem Zeitpunkt.",
            "pl": "Ta tabela pokazuje obejrzane relacje wraz z kontem autora, tekstem (jeśli jest dostępny), linkiem i czasem.",
            "tr": "Bu tablo, görüntülediğin hikayeleri; paylaşan hesap, metin (varsa), bağlantı ve zaman ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول القصص التي شاهدتها، مع الحساب الناشر والنص (إن وجد) والرابط والوقت.",
            "ru": "В этой таблице показаны просмотренные истории с аккаунтом автора, текстом (если есть), ссылкой и временем.",
            "it": "Questa tabella mostra le storie che hai visualizzato, con l'account che le ha pubblicate, il testo (se disponibile), il link e l'ora.",
            "ro": "Acest tabel arată poveștile vizualizate, cu contul care le-a publicat, textul (dacă există), linkul și ora.",
            "es": "Esta tabla muestra las historias que viste, con la cuenta que las publicó, su texto (si está disponible), el enlace y la hora.",
            "sq": "Kjo tabelë tregon stories që ke parë, me llogarinë që i postoi, tekstin (nëse ka), lidhjen dhe kohën."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Account",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogaria"
            },
            "Text": {
              "en": "Text",
              "nl": "Tekst",
              "de": "Text",
              "pl": "Tekst",
              "tr": "Metin",
              "ar": "النص",
              "ru": "Текст",
              "it": "Testo",
              "ro": "Text",
              "es": "Texto",
              "sq": "Teksti"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = _ig_content_rows(reader, errors, "story_interactions/stories_viewed")
    return _st_df(rows, ["Account", "Text", "URL", "Date"])


def instagram_likes_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Your likes.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``Type``, ``Text``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows posts and comments you liked, with the account that posted them, their text (if available), the time and, for posts, the link.",
          "source_file": "your_instagram_activity/likes/liked_posts.json, liked_comments.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "Type": "Kind of content or interaction.",
            "Text": "Text of the content (caption or post text), if the export contains it.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_likes",
          "title": {
            "en": "Your likes",
            "nl": "Je vind-ik-leuks",
            "de": "Ihre „Gefällt mir“-Angaben",
            "pl": "Twoje polubienia",
            "tr": "Beğenilerin",
            "ar": "إعجاباتك",
            "ru": "Ваши отметки «Нравится»",
            "it": "I tuoi Mi piace",
            "ro": "Aprecierile tale",
            "es": "Tus Me gusta",
            "sq": "Pëlqimet e tua"
          },
          "description": {
            "en": "This table shows posts and comments you liked, with the account that posted them, their text (if available), the time and, for posts, the link.",
            "nl": "Deze tabel toont berichten en opmerkingen die je leuk vond, met het account dat ze plaatste, de tekst (indien beschikbaar), het tijdstip en bij berichten de link.",
            "de": "Diese Tabelle zeigt Beiträge und Kommentare, die Sie mit „Gefällt mir“ markiert haben, mit dem Account, der sie veröffentlicht hat, ihrem Text (falls vorhanden), dem Zeitpunkt und bei Beiträgen dem Link.",
            "pl": "Ta tabela pokazuje polubione posty i komentarze wraz z kontem autora, tekstem (jeśli jest dostępny), czasem oraz — w przypadku postów — linkiem.",
            "tr": "Bu tablo, beğendiğin gönderileri ve yorumları; paylaşan hesap, metin (varsa), zaman ve gönderiler için bağlantı ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول المنشورات والتعليقات التي أعجبتك، مع الحساب الناشر والنص (إن وجد) والوقت، والرابط بالنسبة للمنشورات.",
            "ru": "В этой таблице показаны понравившиеся вам публикации и комментарии с аккаунтом автора, текстом (если есть), временем и — для публикаций — ссылкой.",
            "it": "Questa tabella mostra i post e i commenti a cui hai messo Mi piace, con l'account che li ha pubblicati, il testo (se disponibile), l'ora e, per i post, il link.",
            "ro": "Acest tabel arată postările și comentariile apreciate, cu contul care le-a publicat, textul (dacă există), ora și, pentru postări, linkul.",
            "es": "Esta tabla muestra las publicaciones y comentarios que te gustaron, con la cuenta que los publicó, su texto (si está disponible), la hora y, en las publicaciones, el enlace.",
            "sq": "Kjo tabelë tregon postimet dhe komentet që ke pëlqyer, me llogarinë që i postoi, tekstin (nëse ka), kohën dhe për postimet lidhjen."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Account",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogaria"
            },
            "Type": {
              "en": "Type",
              "nl": "Type",
              "de": "Art",
              "pl": "Rodzaj",
              "tr": "Tür",
              "ar": "النوع",
              "ru": "Тип",
              "it": "Tipo",
              "ro": "Tip",
              "es": "Tipo",
              "sq": "Lloji"
            },
            "Text": {
              "en": "Text",
              "nl": "Tekst",
              "de": "Text",
              "pl": "Tekst",
              "tr": "Metin",
              "ar": "النص",
              "ru": "Текст",
              "it": "Testo",
              "ro": "Text",
              "es": "Texto",
              "sq": "Teksti"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = [
        (account, "Beitrag", text, url, date)
        for account, text, url, date in _ig_content_rows(reader, errors, "likes/liked_posts")
    ]
    for data in _ig_files(reader, errors, "likes/liked_comments"):
        try:
            for record in _st_records(data):
                if not isinstance(record, dict):
                    continue
                account = _st_fix(record.get("title") or "")
                if not account:
                    username, name, _ = _st_owner(record)
                    account = username or name
                rows.append((account, "Kommentar", "", "", _st_iso(_st_record_ts(record), errors)))
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    rows.sort(key=lambda r: r[4], reverse=True)
    return _st_df(rows, ["Account", "Type", "Text", "URL", "Date"])


# file stem -> kind of interaction (only the kind is kept, no answers or emoji texts of polls)
_IG_STORY_INTERACTIONS = [
    ("story_interactions/story_likes", "Gefällt mir"),
    ("story_interactions/emoji_story_reactions", None),
    ("story_interactions/story_reaction_sticker_reactions", None),
    ("story_interactions/emoji_sliders", "Emoji-Slider"),
    ("story_interactions/polls", "Umfrage"),
    ("story_interactions/quizzes", "Quiz"),
    ("story_interactions/questions", "Frage"),
    ("story_interactions/countdowns", "Countdown"),
]


def instagram_story_interactions_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Your interactions with stories.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``Type``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows how you interacted with stories (posts that disappear after 24 hours), with the account that posted the story, the kind of interaction and the time.",
          "source_file": "your_instagram_activity/story_interactions/*.json (likes, emoji reactions, stickers, sliders, polls, quizzes, questions, countdowns)",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "Type": "Kind of content or interaction.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_story_interactions",
          "title": {
            "en": "Your interactions with stories",
            "nl": "Je interacties met stories",
            "de": "Ihre Interaktionen mit Storys",
            "pl": "Twoje interakcje z relacjami",
            "tr": "Hikayelerle etkileşimlerin",
            "ar": "تفاعلاتك مع القصص",
            "ru": "Ваши взаимодействия с историями",
            "it": "Le tue interazioni con le storie",
            "ro": "Interacțiunile tale cu poveștile",
            "es": "Tus interacciones con historias",
            "sq": "Ndërveprimet e tua me stories"
          },
          "description": {
            "en": "This table shows how you interacted with stories (posts that disappear after 24 hours), with the account that posted the story, the kind of interaction and the time.",
            "nl": "Deze tabel toont hoe je met stories (berichten die na 24 uur verdwijnen) hebt gereageerd, met het account van de story, het soort interactie en het tijdstip.",
            "de": "Diese Tabelle zeigt Ihre Interaktionen mit Storys (Beiträge, die nach 24 Stunden verschwinden), mit dem Account, der die Story veröffentlicht hat, der Art der Interaktion und dem Zeitpunkt.",
            "pl": "Ta tabela pokazuje Twoje interakcje z relacjami (postami znikającymi po 24 godzinach) wraz z kontem autora, rodzajem interakcji i czasem.",
            "tr": "Bu tablo, hikayelerle (24 saat sonra kaybolan gönderiler) etkileşimlerini; hikayeyi paylaşan hesap, etkileşim türü ve zaman ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول تفاعلاتك مع القصص (منشورات تختفي بعد 24 ساعة)، مع الحساب الناشر ونوع التفاعل والوقت.",
            "ru": "В этой таблице показаны ваши взаимодействия с историями (публикациями, исчезающими через 24 часа) с аккаунтом автора, типом взаимодействия и временем.",
            "it": "Questa tabella mostra le tue interazioni con le storie (post che scompaiono dopo 24 ore), con l'account che ha pubblicato la storia, il tipo di interazione e l'ora.",
            "ro": "Acest tabel arată interacțiunile tale cu poveștile (postări care dispar după 24 de ore), cu contul care a publicat povestea, tipul interacțiunii și ora.",
            "es": "Esta tabla muestra tus interacciones con historias (publicaciones que desaparecen a las 24 horas), con la cuenta que publicó la historia, el tipo de interacción y la hora.",
            "sq": "Kjo tabelë tregon ndërveprimet e tua me stories (postime që zhduken pas 24 orësh), me llogarinë që e postoi, llojin e ndërveprimit dhe kohën."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Account",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogaria"
            },
            "Type": {
              "en": "Type",
              "nl": "Type",
              "de": "Art",
              "pl": "Rodzaj",
              "tr": "Tür",
              "ar": "النوع",
              "ru": "Тип",
              "it": "Tipo",
              "ro": "Tip",
              "es": "Tipo",
              "sq": "Lloji"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = []
    for stem, kind in _IG_STORY_INTERACTIONS:
        for data in _ig_files(reader, errors, stem):
            try:
                for record in _st_records(data):
                    if not isinstance(record, dict):
                        continue
                    username, name, _ = _st_owner(record)
                    if not username and not name:
                        username = _st_fix(record.get("title") or "")
                    rows.append((
                        username or name,
                        kind or _st_top_value(record, "emoji") or "Reaktion",
                        _st_iso(_st_record_ts(record), errors),
                    ))
            except Exception as e:
                logger.error("Exception caught: %s", e)
                errors[type(e).__name__] += 1
    rows.sort(key=lambda r: r[2], reverse=True)
    return _st_df(rows, ["Account", "Type", "Date"])


def instagram_saved_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Saved posts.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows posts you saved, with the account that posted them, the link and the time you saved them.",
          "source_file": "your_instagram_activity/saved/saved_posts.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_saved",
          "title": {
            "en": "Saved posts",
            "nl": "Opgeslagen berichten",
            "de": "Gespeicherte Beiträge",
            "pl": "Zapisane posty",
            "tr": "Kaydedilen gönderiler",
            "ar": "المنشورات المحفوظة",
            "ru": "Сохранённые публикации",
            "it": "Post salvati",
            "ro": "Postări salvate",
            "es": "Publicaciones guardadas",
            "sq": "Postimet e ruajtura"
          },
          "description": {
            "en": "This table shows posts you saved, with the account that posted them, the link and the time you saved them.",
            "nl": "Deze tabel toont berichten die je hebt opgeslagen, met het account dat ze plaatste, de link en het tijdstip van opslaan.",
            "de": "Diese Tabelle zeigt Beiträge, die Sie gespeichert haben, mit dem Account, der sie veröffentlicht hat, dem Link und dem Zeitpunkt der Speicherung.",
            "pl": "Ta tabela pokazuje zapisane posty wraz z kontem autora, linkiem i czasem zapisania.",
            "tr": "Bu tablo, kaydettiğin gönderileri; paylaşan hesap, bağlantı ve kaydetme zamanı ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول المنشورات التي حفظتها، مع الحساب الناشر والرابط ووقت الحفظ.",
            "ru": "В этой таблице показаны сохранённые публикации с аккаунтом автора, ссылкой и временем сохранения.",
            "it": "Questa tabella mostra i post che hai salvato, con l'account che li ha pubblicati, il link e l'ora del salvataggio.",
            "ro": "Acest tabel arată postările salvate, cu contul care le-a publicat, linkul și ora salvării.",
            "es": "Esta tabla muestra las publicaciones que guardaste, con la cuenta que las publicó, el enlace y la hora en que las guardaste.",
            "sq": "Kjo tabelë tregon postimet që ke ruajtur, me llogarinë që i postoi, lidhjen dhe kohën e ruajtjes."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Account",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogaria"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = _ig_content_rows(reader, errors, "saved/saved_posts", with_text=False)
    return _st_df(rows, ["Account", "URL", "Date"])


def instagram_comments_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Your comments.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Comment``, ``Account``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows comments you wrote, with the text of the comment, the account whose post you commented on and the time.",
          "source_file": "your_instagram_activity/comments/post_comments_*.json, reels_comments.json, hype.json",
          "columns": {
            "Comment": "Text of the participant's comment.",
            "Account": "Username (or name) of the account that posted the content.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_comments",
          "title": {
            "en": "Your comments",
            "nl": "Je opmerkingen",
            "de": "Ihre Kommentare",
            "pl": "Twoje komentarze",
            "tr": "Yorumların",
            "ar": "تعليقاتك",
            "ru": "Ваши комментарии",
            "it": "I tuoi commenti",
            "ro": "Comentariile tale",
            "es": "Tus comentarios",
            "sq": "Komentet e tua"
          },
          "description": {
            "en": "This table shows comments you wrote, with the text of the comment, the account whose post you commented on and the time.",
            "nl": "Deze tabel toont opmerkingen die je hebt geplaatst, met de tekst, het account op wiens bericht je reageerde en het tijdstip.",
            "de": "Diese Tabelle zeigt Ihre Kommentare, mit dem Text des Kommentars, dem Account, dessen Beitrag Sie kommentiert haben, und dem Zeitpunkt.",
            "pl": "Ta tabela pokazuje Twoje komentarze wraz z ich treścią, kontem, którego post skomentowałeś/aś, i czasem.",
            "tr": "Bu tablo, yazdığın yorumları; yorum metni, gönderisine yorum yaptığın hesap ve zaman ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول التعليقات التي كتبتها، مع نص التعليق والحساب الذي علّقت على منشوره والوقت.",
            "ru": "В этой таблице показаны ваши комментарии с текстом, аккаунтом, чью публикацию вы прокомментировали, и временем.",
            "it": "Questa tabella mostra i commenti che hai scritto, con il testo, l'account di cui hai commentato il post e l'ora.",
            "ro": "Acest tabel arată comentariile scrise de tine, cu textul, contul a cărui postare ai comentat-o și ora.",
            "es": "Esta tabla muestra los comentarios que escribiste, con el texto, la cuenta cuya publicación comentaste y la hora.",
            "sq": "Kjo tabelë tregon komentet që ke shkruar, me tekstin, llogarinë që ke komentuar dhe kohën."
          },
          "headers": {
            "Comment": {
              "en": "Comment",
              "nl": "Opmerking",
              "de": "Kommentar",
              "pl": "Komentarz",
              "tr": "Yorum",
              "ar": "التعليق",
              "ru": "Комментарий",
              "it": "Commento",
              "ro": "Comentariu",
              "es": "Comentario",
              "sq": "Komenti"
            },
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Account",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogaria"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = []
    for stem in ("comments/post_comments", "comments/reels_comments", "comments/hype"):
        for data in _ig_files(reader, errors, stem):
            try:
                for record in _st_records(data):
                    if not isinstance(record, dict):
                        continue
                    rows.append((
                        _st_top_value(record, "comment"),
                        _st_top_value(record, "media_owner"),
                        _st_iso(_st_record_ts(record), errors),
                    ))
            except Exception as e:
                logger.error("Exception caught: %s", e)
                errors[type(e).__name__] += 1
    rows.sort(key=lambda r: r[2], reverse=True)
    return _st_df(rows, ["Comment", "Account", "Date"])


def instagram_searches_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Your recent searches.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Search``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows what you searched for (e.g. profiles, words or hashtags), with the search text, the link of the result (if available) and the time.",
          "source_file": "logged_information/recent_searches/*.json",
          "columns": {
            "Search": "Search text (word, hashtag or profile name).",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_searches",
          "title": {
            "en": "Your recent searches",
            "nl": "Je recente zoekopdrachten",
            "de": "Ihre letzten Suchanfragen",
            "pl": "Twoje ostatnie wyszukiwania",
            "tr": "Son aramaların",
            "ar": "عمليات البحث الأخيرة",
            "ru": "Ваши недавние запросы",
            "it": "Le tue ricerche recenti",
            "ro": "Căutările tale recente",
            "es": "Tus búsquedas recientes",
            "sq": "Kërkimet e tua të fundit"
          },
          "description": {
            "en": "This table shows what you searched for (e.g. profiles, words or hashtags), with the search text, the link of the result (if available) and the time.",
            "nl": "Deze tabel toont waarnaar je hebt gezocht (bijv. profielen, woorden of hashtags), met de zoektekst, de link van het resultaat (indien beschikbaar) en het tijdstip.",
            "de": "Diese Tabelle zeigt, wonach Sie gesucht haben (z. B. Profile, Wörter oder Hashtags), mit dem Suchtext, dem Link des Suchergebnisses (falls vorhanden) und dem Zeitpunkt.",
            "pl": "Ta tabela pokazuje, czego szukałeś/aś (np. profili, słów lub hashtagów), wraz z tekstem wyszukiwania, linkiem wyniku (jeśli jest) i czasem.",
            "tr": "Bu tablo, neleri aradığını (ör. profiller, kelimeler veya hashtag'ler); arama metni, sonucun bağlantısı (varsa) ve zaman ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول ما بحثت عنه (مثل الملفات الشخصية أو الكلمات أو الوسوم)، مع نص البحث ورابط النتيجة (إن وجد) والوقت.",
            "ru": "В этой таблице показано, что вы искали (например, профили, слова или хештеги), с текстом запроса, ссылкой на результат (если есть) и временем.",
            "it": "Questa tabella mostra cosa hai cercato (ad es. profili, parole o hashtag), con il testo della ricerca, il link del risultato (se disponibile) e l'ora.",
            "ro": "Acest tabel arată ce ai căutat (de ex. profiluri, cuvinte sau hashtaguri), cu textul căutării, linkul rezultatului (dacă există) și ora.",
            "es": "Esta tabla muestra lo que buscaste (p. ej., perfiles, palabras o hashtags), con el texto de búsqueda, el enlace del resultado (si lo hay) y la hora.",
            "sq": "Kjo tabelë tregon çfarë ke kërkuar (p.sh. profile, fjalë ose hashtag), me tekstin e kërkimit, lidhjen e rezultatit (nëse ka) dhe kohën."
          },
          "headers": {
            "Search": {
              "en": "Search",
              "nl": "Zoekopdracht",
              "de": "Suchanfrage",
              "pl": "Wyszukiwanie",
              "tr": "Arama",
              "ar": "البحث",
              "ru": "Поиск",
              "it": "Ricerca",
              "ro": "Căutare",
              "es": "Búsqueda",
              "sq": "Kërkimi"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = []
    for stem in ("recent_searches/word_or_phrase_searches", "recent_searches/recent_searches",
                 "recent_searches/profile_searches", "recent_searches/tag_searches",
                 "recent_searches/account_searches"):
        for data in _ig_files(reader, errors, stem):
            try:
                for record in _st_records(data):
                    if not isinstance(record, dict):
                        continue
                    text = _st_top_value(record, "search") or _st_fix(record.get("title") or "")
                    if not text:
                        for item in record.get("string_list_data") or []:
                            text = text or _st_fix((item or {}).get("value") or "")
                    url = ""
                    for item in record.get("string_list_data") or []:
                        url = url or ((item or {}).get("href") if _st_is_url((item or {}).get("href")) else "")
                    rows.append((text, url, _st_iso(_st_record_ts(record), errors)))
            except Exception as e:
                logger.error("Exception caught: %s", e)
                errors[type(e).__name__] += 1
    rows = list(dict.fromkeys(rows))
    rows.sort(key=lambda r: r[2], reverse=True)
    return _st_df(rows, ["Search", "URL", "Date"])


def _ig_relation_rows(reader, errors, stem) -> list:
    rows = []
    for data in _ig_files(reader, errors, stem):
        try:
            for record in _st_records(data):
                if not isinstance(record, dict):
                    continue
                name, url = _st_fix(record.get("title") or ""), ""
                for item in record.get("string_list_data") or []:
                    item = item or {}
                    name = name or _st_fix(item.get("value") or "")
                    url = url or (item.get("href") if _st_is_url(item.get("href")) else "")
                rows.append((name, url, _st_iso(_st_record_ts(record), errors)))
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    return rows


def instagram_following_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Accounts you follow.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows the accounts, pages and profiles you follow and since when.",
          "source_file": "connections/followers_and_following/following.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_following",
          "title": {
            "en": "Accounts you follow",
            "nl": "Accounts die je volgt",
            "de": "Accounts, denen Sie folgen",
            "pl": "Konta, które obserwujesz",
            "tr": "Takip ettiğin hesaplar",
            "ar": "الحسابات التي تتابعها",
            "ru": "Аккаунты, на которые вы подписаны",
            "it": "Account che segui",
            "ro": "Conturi pe care le urmărești",
            "es": "Cuentas que sigues",
            "sq": "Llogaritë që ndjek"
          },
          "description": {
            "en": "This table shows the accounts, pages and profiles you follow and since when.",
            "nl": "Deze tabel toont de accounts, pagina's en profielen die je volgt en sinds wanneer.",
            "de": "Diese Tabelle zeigt die Accounts, Seiten und Profile, denen Sie folgen, und seit wann.",
            "pl": "Ta tabela pokazuje konta, strony i profile, które obserwujesz, oraz od kiedy.",
            "tr": "Bu tablo, takip ettiğin hesapları, sayfaları ve profilleri ve ne zamandan beri takip ettiğini gösterir.",
            "ar": "يعرض هذا الجدول الحسابات والصفحات والملفات الشخصية التي تتابعها ومنذ متى.",
            "ru": "В этой таблице показаны аккаунты, страницы и профили, на которые вы подписаны, и с какого времени.",
            "it": "Questa tabella mostra gli account, le pagine e i profili che segui e da quando.",
            "ro": "Acest tabel arată conturile, paginile și profilurile pe care le urmărești și de când.",
            "es": "Esta tabla muestra las cuentas, páginas y perfiles que sigues y desde cuándo.",
            "sq": "Kjo tabelë tregon llogaritë, faqet dhe profilet që ndjek dhe që kur."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Account",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogaria"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = _ig_relation_rows(reader, errors, "followers_and_following/following")
    return _st_df(rows, ["Account", "URL", "Date"])


def instagram_followers_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Accounts that follow you.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows the accounts that follow you, with the link to the account and since when.",
          "source_file": "connections/followers_and_following/followers_*.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_followers",
          "title": {
            "en": "Accounts that follow you",
            "nl": "Accounts die jou volgen",
            "de": "Accounts, die Ihnen folgen",
            "pl": "Konta, które Cię obserwują",
            "tr": "Seni takip eden hesaplar",
            "ar": "الحسابات التي تتابعك",
            "ru": "Ваши подписчики",
            "it": "Account che ti seguono",
            "ro": "Conturi care te urmăresc",
            "es": "Cuentas que te siguen",
            "sq": "Llogaritë që të ndjekin"
          },
          "description": {
            "en": "This table shows the accounts that follow you, with the link to the account and since when.",
            "nl": "Deze tabel toont de accounts die jou volgen, met de link naar het account en sinds wanneer.",
            "de": "Diese Tabelle zeigt die Accounts, die Ihnen folgen, mit dem Link zum Account und seit wann.",
            "pl": "Ta tabela pokazuje konta, które Cię obserwują, wraz z linkiem do konta i datą.",
            "tr": "Bu tablo, seni takip eden hesapları; hesap bağlantısı ve ne zamandan beri ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول الحسابات التي تتابعك، مع رابط الحساب ومنذ متى.",
            "ru": "В этой таблице показаны ваши подписчики со ссылкой на аккаунт и датой.",
            "it": "Questa tabella mostra gli account che ti seguono, con il link all'account e da quando.",
            "ro": "Acest tabel arată conturile care te urmăresc, cu linkul către cont și de când.",
            "es": "Esta tabla muestra las cuentas que te siguen, con el enlace a la cuenta y desde cuándo.",
            "sq": "Kjo tabelë tregon llogaritë që të ndjekin, me lidhjen e llogarisë dhe që kur."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Account",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogaria"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = _ig_relation_rows(reader, errors, "followers_and_following/followers")
    return _st_df(rows, ["Account", "URL", "Date"])


def instagram_hashtags_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Hashtags you follow.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Hashtag``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows the hashtags you follow and since when.",
          "source_file": "connections/followers_and_following/following_hashtags.json",
          "columns": {
            "Hashtag": "Name of the hashtag.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_hashtags",
          "title": {
            "en": "Hashtags you follow",
            "nl": "Hashtags die je volgt",
            "de": "Abonnierte Hashtags",
            "pl": "Obserwowane hashtagi",
            "tr": "Takip ettiğin hashtag'ler",
            "ar": "الوسوم التي تتابعها",
            "ru": "Хештеги, на которые вы подписаны",
            "it": "Hashtag che segui",
            "ro": "Hashtaguri pe care le urmărești",
            "es": "Hashtags que sigues",
            "sq": "Hashtag-ët që ndjek"
          },
          "description": {
            "en": "This table shows the hashtags you follow and since when.",
            "nl": "Deze tabel toont de hashtags die je volgt en sinds wanneer.",
            "de": "Diese Tabelle zeigt die Hashtags, die Sie abonniert haben, und seit wann.",
            "pl": "Ta tabela pokazuje obserwowane hashtagi i od kiedy.",
            "tr": "Bu tablo, takip ettiğin hashtag'leri ve ne zamandan beri takip ettiğini gösterir.",
            "ar": "يعرض هذا الجدول الوسوم التي تتابعها ومنذ متى.",
            "ru": "В этой таблице показаны хештеги, на которые вы подписаны, и с какого времени.",
            "it": "Questa tabella mostra gli hashtag che segui e da quando.",
            "ro": "Acest tabel arată hashtagurile pe care le urmărești și de când.",
            "es": "Esta tabla muestra los hashtags que sigues y desde cuándo.",
            "sq": "Kjo tabelë tregon hashtag-ët që ndjek dhe që kur."
          },
          "headers": {
            "Hashtag": {
              "en": "Hashtag",
              "nl": "Hashtag",
              "de": "Hashtag",
              "pl": "Hashtag",
              "tr": "Hashtag",
              "ar": "الوسم",
              "ru": "Хештег",
              "it": "Hashtag",
              "ro": "Hashtag",
              "es": "Hashtag",
              "sq": "Hashtag"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = _ig_relation_rows(reader, errors, "followers_and_following/following_hashtags")
    return _st_df(rows, ["Hashtag", "URL", "Date"])


def instagram_interests_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Interest categories assigned to you.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Topic``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows interests and topics that were assigned to you based on your activity and that influence which posts and ads you see.",
          "source_file": "preferences/your_topics/your_topics.json",
          "columns": {
            "Topic": "Name of the topic or interest.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_interests",
          "title": {
            "en": "Interest categories assigned to you",
            "nl": "Aan jou toegewezen interessecategorieën",
            "de": "Ihnen zugewiesene Interessenskategorien",
            "pl": "Przypisane Ci kategorie zainteresowań",
            "tr": "Sana atanan ilgi kategorileri",
            "ar": "فئات الاهتمامات المخصصة لك",
            "ru": "Присвоенные вам категории интересов",
            "it": "Categorie di interesse assegnate a te",
            "ro": "Categorii de interese atribuite ție",
            "es": "Categorías de intereses asignadas a ti",
            "sq": "Kategoritë e interesave që të janë caktuar"
          },
          "description": {
            "en": "This table shows interests and topics that were assigned to you based on your activity and that influence which posts and ads you see.",
            "nl": "Deze tabel toont interesses en onderwerpen die op basis van je activiteit aan je zijn toegewezen en bepalen welke berichten en advertenties je ziet.",
            "de": "Diese Tabelle zeigt Interessen und Themen, die Ihnen auf Basis Ihrer Aktivitäten zugewiesen wurden und mitbestimmen, welche Beiträge und Werbung Sie sehen.",
            "pl": "Ta tabela pokazuje zainteresowania i tematy przypisane Ci na podstawie Twojej aktywności, które wpływają na to, jakie posty i reklamy widzisz.",
            "tr": "Bu tablo, etkinliğine göre sana atanan ve hangi gönderileri ve reklamları gördüğünü etkileyen ilgi alanlarını ve konuları gösterir.",
            "ar": "يعرض هذا الجدول الاهتمامات والمواضيع التي خُصصت لك بناءً على نشاطك والتي تؤثر على المنشورات والإعلانات التي تراها.",
            "ru": "В этой таблице показаны интересы и темы, присвоенные вам на основе вашей активности и влияющие на то, какие публикации и рекламу вы видите.",
            "it": "Questa tabella mostra interessi e argomenti assegnati a te in base alla tua attività, che influenzano i post e le inserzioni che vedi.",
            "ro": "Acest tabel arată interesele și subiectele atribuite ție pe baza activității tale, care influențează ce postări și reclame vezi.",
            "es": "Esta tabla muestra los intereses y temas que se te asignaron según tu actividad y que influyen en las publicaciones y anuncios que ves.",
            "sq": "Kjo tabelë tregon interesat dhe temat që të janë caktuar sipas aktivitetit tënd dhe që ndikojnë se cilat postime dhe reklama sheh."
          },
          "headers": {
            "Topic": {
              "en": "Topic",
              "nl": "Onderwerp",
              "de": "Thema",
              "pl": "Temat",
              "tr": "Konu",
              "ar": "الموضوع",
              "ru": "Тема",
              "it": "Argomento",
              "ro": "Subiect",
              "es": "Tema",
              "sq": "Tema"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = []
    for data in _ig_files(reader, errors, "your_topics/your_topics"):
        try:
            for record in _st_records(data):
                if isinstance(record, str):
                    rows.append((_st_fix(record), ""))
                    continue
                if not isinstance(record, dict):
                    continue
                name = _st_top_value(record, "name")
                if name:
                    rows.append((name, _st_iso(_st_record_ts(record), errors)))
                for _, value in _st_vec_values(record):
                    rows.append((value, ""))
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    return _st_df(rows, ["Topic", "Date"])


def instagram_link_history_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Websites you visited in the app.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows websites you opened within the app (in-app browser), with the link and the time of your visit.",
          "source_file": "logged_information/link_history/link_history.json",
          "columns": {
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_link_history",
          "title": {
            "en": "Websites you visited in the app",
            "nl": "Websites die je in de app hebt bezocht",
            "de": "In der App besuchte Webseiten",
            "pl": "Strony odwiedzone w aplikacji",
            "tr": "Uygulamada ziyaret ettiğin web siteleri",
            "ar": "المواقع التي زرتها داخل التطبيق",
            "ru": "Сайты, посещённые в приложении",
            "it": "Siti web visitati nell'app",
            "ro": "Site-uri vizitate în aplicație",
            "es": "Sitios web que visitaste en la app",
            "sq": "Faqet që ke vizituar në aplikacion"
          },
          "description": {
            "en": "This table shows websites you opened within the app (in-app browser), with the link and the time of your visit.",
            "nl": "Deze tabel toont websites die je binnen de app hebt geopend (in-app browser), met de link en het tijdstip van je bezoek.",
            "de": "Diese Tabelle zeigt Webseiten, die Sie innerhalb der App (im In-App-Browser) besucht haben, mit dem Link und dem Zeitpunkt Ihres Besuchs.",
            "pl": "Ta tabela pokazuje strony otwarte w aplikacji (przeglądarka w aplikacji) wraz z linkiem i czasem wizyty.",
            "tr": "Bu tablo, uygulama içinde (uygulama içi tarayıcı) açtığın web sitelerini; bağlantı ve ziyaret zamanı ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول المواقع التي فتحتها داخل التطبيق (المتصفح الداخلي)، مع الرابط ووقت الزيارة.",
            "ru": "В этой таблице показаны сайты, открытые во встроенном браузере приложения, со ссылкой и временем посещения.",
            "it": "Questa tabella mostra i siti web aperti all'interno dell'app (browser in-app), con il link e l'ora della visita.",
            "ro": "Acest tabel arată site-urile deschise în aplicație (browserul din aplicație), cu linkul și ora vizitei.",
            "es": "Esta tabla muestra los sitios web que abriste dentro de la app (navegador integrado), con el enlace y la hora de la visita.",
            "sq": "Kjo tabelë tregon faqet që ke hapur brenda aplikacionit (shfletuesi i brendshëm), me lidhjen dhe kohën e vizitës."
          },
          "headers": {
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = []
    for data in _ig_files(reader, errors, "link_history/link_history"):
        try:
            for record in _st_records(data):
                if isinstance(record, dict):
                    rows.append((_st_content_url(record), _st_iso(_st_record_ts(record) or _st_text_time(record), errors)))
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    return _st_df(rows, ["URL", "Date"])


def instagram_off_meta_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """External apps and websites.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Website``, ``Event``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows activity outside the platform that other apps and websites shared with Meta (e.g. page views or purchases), with the app or website, the kind of activity and the time.",
          "source_file": "apps_and_websites_off_of_instagram/apps_and_websites/your_activity_off_meta_technologies.json",
          "columns": {
            "Website": "Name of the website or app that shared the activity.",
            "Event": "Type of activity, translated to German where known (e.g. PAGE_VIEW -> Seitenaufruf).",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_off_meta",
          "title": {
            "en": "External apps and websites",
            "nl": "Externe apps en websites",
            "de": "Externe Apps und Websites",
            "pl": "Zewnętrzne aplikacje i witryny",
            "tr": "Harici uygulamalar ve web siteleri",
            "ar": "التطبيقات والمواقع الخارجية",
            "ru": "Внешние приложения и сайты",
            "it": "App e siti web esterni",
            "ro": "Aplicații și site-uri externe",
            "es": "Aplicaciones y sitios web externos",
            "sq": "Aplikacione dhe faqe të jashtme"
          },
          "description": {
            "en": "This table shows activity outside the platform that other apps and websites shared with Meta (e.g. page views or purchases), with the app or website, the kind of activity and the time.",
            "nl": "Deze tabel toont activiteit buiten het platform die andere apps en websites met Meta hebben gedeeld (bijv. paginaweergaven of aankopen), met de app of website, het soort activiteit en het tijdstip.",
            "de": "Diese Tabelle zeigt Aktivitäten außerhalb der Plattform, die andere Apps und Websites mit Meta geteilt haben (z. B. Seitenaufrufe oder Käufe), mit der App bzw. Website, der Art der Aktivität und dem Zeitpunkt.",
            "pl": "Ta tabela pokazuje aktywność poza platformą, którą inne aplikacje i witryny udostępniły Meta (np. wyświetlenia stron lub zakupy), wraz z aplikacją lub witryną, rodzajem aktywności i czasem.",
            "tr": "Bu tablo, diğer uygulamaların ve web sitelerinin Meta ile paylaştığı platform dışı etkinlikleri (ör. sayfa görüntülemeleri veya satın almalar); uygulama ya da site, etkinlik türü ve zaman ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول النشاط خارج المنصة الذي شاركته تطبيقات ومواقع أخرى مع Meta (مثل مشاهدات الصفحات أو المشتريات)، مع التطبيق أو الموقع ونوع النشاط والوقت.",
            "ru": "В этой таблице показана активность вне платформы, которой другие приложения и сайты поделились с Meta (например, просмотры страниц или покупки), с приложением или сайтом, типом активности и временем.",
            "it": "Questa tabella mostra le attività al di fuori della piattaforma che altre app e siti web hanno condiviso con Meta (ad es. visualizzazioni di pagine o acquisti), con l'app o il sito, il tipo di attività e l'ora.",
            "ro": "Acest tabel arată activitatea din afara platformei pe care alte aplicații și site-uri au partajat-o cu Meta (de ex. vizualizări de pagini sau achiziții), cu aplicația sau site-ul, tipul activității și ora.",
            "es": "Esta tabla muestra la actividad fuera de la plataforma que otras aplicaciones y sitios web compartieron con Meta (p. ej., visitas a páginas o compras), con la aplicación o sitio, el tipo de actividad y la hora.",
            "sq": "Kjo tabelë tregon aktivitetin jashtë platformës që aplikacione dhe faqe të tjera ndanë me Meta (p.sh. shikime faqesh ose blerje), me aplikacionin ose faqen, llojin e aktivitetit dhe kohën."
          },
          "headers": {
            "Website": {
              "en": "Website or app",
              "nl": "Website of app",
              "de": "Website oder App",
              "pl": "Witryna lub aplikacja",
              "tr": "Web sitesi veya uygulama",
              "ar": "الموقع أو التطبيق",
              "ru": "Сайт или приложение",
              "it": "Sito web o app",
              "ro": "Site sau aplicație",
              "es": "Sitio web o aplicación",
              "sq": "Faqja ose aplikacioni"
            },
            "Event": {
              "en": "Activity",
              "nl": "Activiteit",
              "de": "Aktivität",
              "pl": "Aktywność",
              "tr": "Etkinlik",
              "ar": "النشاط",
              "ru": "Действие",
              "it": "Attività",
              "ro": "Activitate",
              "es": "Actividad",
              "sq": "Aktiviteti"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = []
    for data in _ig_files(reader, errors, "apps_and_websites/your_activity_off_meta_technologies"):
        try:
            rows += _st_off_meta_rows(data, errors)
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    rows.sort(key=lambda r: r[2], reverse=True)
    return _st_df(rows, ["Website", "Event", "Date"])


def instagram_advertisers_using_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Advertisers using your information.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Advertiser``, ``Usage``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows advertisers that used information about you, and how they used it.",
          "source_file": "ads_information/instagram_ads_and_businesses/advertisers_using_your_activity_or_information.json",
          "columns": {
            "Advertiser": "Name of the advertiser.",
            "Usage": "How the advertiser used the participant's information, as named in the export."
          }
        }

    Table config::

        {
          "id": "instagram_advertisers_using",
          "title": {
            "en": "Advertisers using your information",
            "nl": "Adverteerders die je informatie gebruiken",
            "de": "Werbetreibende, die Ihre Informationen nutzen",
            "pl": "Reklamodawcy korzystający z Twoich informacji",
            "tr": "Bilgilerini kullanan reklamverenler",
            "ar": "المعلنون الذين يستخدمون معلوماتك",
            "ru": "Рекламодатели, использующие вашу информацию",
            "it": "Inserzionisti che usano le tue informazioni",
            "ro": "Agenți de publicitate care îți folosesc informațiile",
            "es": "Anunciantes que usan tu información",
            "sq": "Reklamuesit që përdorin informacionin tënd"
          },
          "description": {
            "en": "This table shows advertisers that used information about you, and how they used it.",
            "nl": "Deze tabel toont adverteerders die informatie over jou hebben gebruikt, en hoe.",
            "de": "Diese Tabelle zeigt Werbetreibende, die Informationen über Sie genutzt haben, und auf welche Weise.",
            "pl": "Ta tabela pokazuje reklamodawców, którzy korzystali z informacji o Tobie, i w jaki sposób.",
            "tr": "Bu tablo, hakkındaki bilgileri kullanan reklamverenleri ve bunları nasıl kullandıklarını gösterir.",
            "ar": "يعرض هذا الجدول المعلنين الذين استخدموا معلومات عنك وكيفية استخدامها.",
            "ru": "В этой таблице показаны рекламодатели, использовавшие информацию о вас, и способ её использования.",
            "it": "Questa tabella mostra gli inserzionisti che hanno usato informazioni su di te e in che modo.",
            "ro": "Acest tabel arată agenții de publicitate care au folosit informații despre tine și modul în care le-au folosit.",
            "es": "Esta tabla muestra los anunciantes que usaron información sobre ti y de qué forma.",
            "sq": "Kjo tabelë tregon reklamuesit që kanë përdorur informacion për ty dhe si e kanë përdorur."
          },
          "headers": {
            "Advertiser": {
              "en": "Advertiser",
              "nl": "Adverteerder",
              "de": "Werbetreibende*r",
              "pl": "Reklamodawca",
              "tr": "Reklamveren",
              "ar": "المعلن",
              "ru": "Рекламодатель",
              "it": "Inserzionista",
              "ro": "Agent de publicitate",
              "es": "Anunciante",
              "sq": "Reklamuesi"
            },
            "Usage": {
              "en": "Information used",
              "nl": "Gebruikte informatie",
              "de": "Genutzte Information",
              "pl": "Wykorzystane informacje",
              "tr": "Kullanılan bilgi",
              "ar": "المعلومات المستخدمة",
              "ru": "Использованная информация",
              "it": "Informazioni utilizzate",
              "ro": "Informații utilizate",
              "es": "Información utilizada",
              "sq": "Informacioni i përdorur"
            }
          }
        }
    """
    rows = []
    for data in _ig_files(reader, errors, "instagram_ads_and_businesses/advertisers_using_your_activity_or_information"):
        try:
            rows += [(value, label) for label, value in _st_vec_values(data)]
            for record in _st_records(data):  # older format
                if isinstance(record, dict) and record.get("advertiser_name"):
                    rows.append((_st_fix(record["advertiser_name"]), ""))
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    return _st_df(rows, ["Advertiser", "Usage"])


def instagram_other_categories_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Categories used to reach you with ads.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Category``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows categories Meta assigned to you that advertisers can use to reach you.",
          "source_file": "ads_information/instagram_ads_and_businesses/other_categories_used_to_reach_you.json",
          "columns": {
            "Category": "Category used to reach the participant with ads."
          }
        }

    Table config::

        {
          "id": "instagram_other_categories",
          "title": {
            "en": "Categories used to reach you with ads",
            "nl": "Categorieën om je met advertenties te bereiken",
            "de": "Kategorien, über die Sie mit Werbung erreicht werden",
            "pl": "Kategorie używane do docierania do Ciebie z reklamami",
            "tr": "Sana reklamla ulaşmak için kullanılan kategoriler",
            "ar": "الفئات المستخدمة للوصول إليك بالإعلانات",
            "ru": "Категории для показа вам рекламы",
            "it": "Categorie usate per raggiungerti con le inserzioni",
            "ro": "Categorii folosite pentru a ajunge la tine cu reclame",
            "es": "Categorías usadas para llegar a ti con anuncios",
            "sq": "Kategoritë e përdorura për të të arritur me reklama"
          },
          "description": {
            "en": "This table shows categories Meta assigned to you that advertisers can use to reach you.",
            "nl": "Deze tabel toont categorieën die Meta aan je heeft toegewezen en die adverteerders kunnen gebruiken om je te bereiken.",
            "de": "Diese Tabelle zeigt Kategorien, die Meta Ihnen zugewiesen hat und über die Werbetreibende Sie erreichen können.",
            "pl": "Ta tabela pokazuje kategorie przypisane Ci przez Meta, których reklamodawcy mogą używać, aby do Ciebie dotrzeć.",
            "tr": "Bu tablo, Meta'nın sana atadığı ve reklamverenlerin sana ulaşmak için kullanabileceği kategorileri gösterir.",
            "ar": "يعرض هذا الجدول الفئات التي خصصتها لك Meta والتي يمكن للمعلنين استخدامها للوصول إليك.",
            "ru": "В этой таблице показаны категории, присвоенные вам Meta, которые рекламодатели могут использовать, чтобы достучаться до вас.",
            "it": "Questa tabella mostra le categorie che Meta ti ha assegnato e che gli inserzionisti possono usare per raggiungerti.",
            "ro": "Acest tabel arată categoriile atribuite ție de Meta pe care agenții de publicitate le pot folosi pentru a ajunge la tine.",
            "es": "Esta tabla muestra las categorías que Meta te asignó y que los anunciantes pueden usar para llegar a ti.",
            "sq": "Kjo tabelë tregon kategoritë që Meta të ka caktuar dhe që reklamuesit mund t'i përdorin për të të arritur."
          },
          "headers": {
            "Category": {
              "en": "Category",
              "nl": "Categorie",
              "de": "Kategorie",
              "pl": "Kategoria",
              "tr": "Kategori",
              "ar": "الفئة",
              "ru": "Категория",
              "it": "Categoria",
              "ro": "Categorie",
              "es": "Categoría",
              "sq": "Kategoria"
            }
          }
        }
    """
    rows = []
    for data in _ig_files(reader, errors, "instagram_ads_and_businesses/other_categories_used_to_reach_you"):
        try:
            rows += [(value,) for _, value in _st_vec_values(data)]
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    return _st_df(rows, ["Category"])


def instagram_login_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Login and logout history.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Event``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows when you logged in, logged out or registered. Only the times are stored, no login details, IP addresses, device identifiers or locations.",
          "source_file": "security_and_login_information/login_and_profile_creation/login_activity.json, logout_activity.json, signup_details.json (times only)",
          "columns": {
            "Event": "Type of activity, translated to German where known (e.g. PAGE_VIEW -> Seitenaufruf).",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "instagram_login",
          "title": {
            "en": "Login and logout history",
            "nl": "Aanmeld- en afmeldgeschiedenis",
            "de": "Login- und Logout-Verlauf",
            "pl": "Historia logowań i wylogowań",
            "tr": "Giriş ve çıkış geçmişi",
            "ar": "سجل تسجيل الدخول والخروج",
            "ru": "История входов и выходов",
            "it": "Cronologia di accessi e disconnessioni",
            "ro": "Istoricul conectărilor și deconectărilor",
            "es": "Historial de inicios y cierres de sesión",
            "sq": "Historiku i hyrjeve dhe daljeve"
          },
          "description": {
            "en": "This table shows when you logged in, logged out or registered. Only the times are stored, no login details, IP addresses, device identifiers or locations.",
            "nl": "Deze tabel toont wanneer je je hebt aangemeld, afgemeld of geregistreerd. Alleen de tijdstippen worden opgeslagen, geen inloggegevens, IP-adressen, apparaat-ID's of locaties.",
            "de": "Diese Tabelle zeigt, wann Sie sich angemeldet, abgemeldet oder registriert haben. Gespeichert werden nur die Zeitpunkte, keine Anmeldeinformationen, IP-Adressen, Geräte-Kennungen oder Standorte.",
            "pl": "Ta tabela pokazuje, kiedy się logowałeś/aś, wylogowywałeś/aś lub rejestrowałeś/aś. Zapisywane są tylko czasy, bez danych logowania, adresów IP, identyfikatorów urządzeń ani lokalizacji.",
            "tr": "Bu tablo, ne zaman giriş yaptığını, çıkış yaptığını veya kaydolduğunu gösterir. Yalnızca zamanlar saklanır; giriş bilgileri, IP adresleri, cihaz kimlikleri veya konumlar saklanmaz.",
            "ar": "يعرض هذا الجدول متى سجّلت الدخول أو الخروج أو أنشأت حسابك. تُحفظ الأوقات فقط، دون بيانات الدخول أو عناوين IP أو معرّفات الأجهزة أو المواقع.",
            "ru": "В этой таблице показано, когда вы входили, выходили или регистрировались. Сохраняется только время, без данных для входа, IP-адресов, идентификаторов устройств и местоположений.",
            "it": "Questa tabella mostra quando hai effettuato l'accesso, la disconnessione o la registrazione. Vengono salvati solo gli orari, nessuna credenziale, indirizzo IP, identificativo del dispositivo o posizione.",
            "ro": "Acest tabel arată când te-ai conectat, deconectat sau înregistrat. Se salvează doar orele, fără date de autentificare, adrese IP, identificatori de dispozitiv sau locații.",
            "es": "Esta tabla muestra cuándo iniciaste sesión, la cerraste o te registraste. Solo se guardan las horas, sin datos de acceso, direcciones IP, identificadores de dispositivo ni ubicaciones.",
            "sq": "Kjo tabelë tregon kur je identifikuar, ke dalë ose je regjistruar. Ruhen vetëm oraret, pa të dhëna hyrjeje, adresa IP, identifikues pajisjesh ose vendndodhje."
          },
          "headers": {
            "Event": {
              "en": "Activity",
              "nl": "Activiteit",
              "de": "Aktivität",
              "pl": "Aktywność",
              "tr": "Etkinlik",
              "ar": "النشاط",
              "ru": "Действие",
              "it": "Attività",
              "ro": "Activitate",
              "es": "Actividad",
              "sq": "Aktiviteti"
            },
            "Date": {
              "en": "Date and time",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data i godzina",
              "tr": "Tarih ve saat",
              "ar": "التاريخ والوقت",
              "ru": "Дата и время",
              "it": "Data e ora",
              "ro": "Data și ora",
              "es": "Fecha y hora",
              "sq": "Data dhe ora"
            }
          }
        }
    """
    rows = []
    sources = (
        ("login_and_profile_creation/login_activity", "Login"),
        ("login_and_profile_creation/logout_activity", "Logout"),
        ("login_and_profile_creation/signup_details", "Registrierung"),
    )
    for stem, kind in sources:
        for data in _ig_files(reader, errors, stem):
            try:
                for record in _st_records(data):
                    if not isinstance(record, dict):
                        continue
                    # Only the time; IP address, cookie, device, e-mail etc. are not read.
                    ts = None
                    for label, entry in _st_top(record):
                        if _st_is(label, "time") and _st_is_epoch(entry.get("timestamp")):
                            ts = entry["timestamp"]
                    rows.append((kind, _st_iso(ts or _st_record_ts(record), errors)))
            except Exception as e:
                logger.error("Exception caught: %s", e)
                errors[type(e).__name__] += 1
    rows.sort(key=lambda r: r[1], reverse=True)
    return _st_df(rows, ["Event", "Date"])


# Extractor registry & platform info
# ---------------------------------------------------------------------------

#: Mapping from the string names used in port_config.json to actual extractor functions.
#: Only the data described in the participant information; the order is the table order.
EXTRACTOR_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "instagram_posts_viewed_to_df": instagram_posts_viewed_to_df,
    "instagram_ads_viewed_to_df": instagram_ads_viewed_to_df,
    "instagram_stories_viewed_to_df": instagram_stories_viewed_to_df,
    "instagram_likes_to_df": instagram_likes_to_df,
    "instagram_story_interactions_to_df": instagram_story_interactions_to_df,
    "instagram_saved_to_df": instagram_saved_to_df,
    "instagram_comments_to_df": instagram_comments_to_df,
    "instagram_searches_to_df": instagram_searches_to_df,
    "instagram_following_to_df": instagram_following_to_df,
    "instagram_followers_to_df": instagram_followers_to_df,
    "instagram_hashtags_to_df": instagram_hashtags_to_df,
    "instagram_interests_to_df": instagram_interests_to_df,
    "instagram_link_history_to_df": instagram_link_history_to_df,
    "instagram_off_meta_to_df": instagram_off_meta_to_df,
    "instagram_advertisers_using_to_df": instagram_advertisers_using_to_df,
    "instagram_other_categories_to_df": instagram_other_categories_to_df,
    "instagram_login_to_df": instagram_login_to_df,
}
# Empty tables (no rows, or only empty cells) are never shown to participants.
EXTRACTOR_REGISTRY = {name: _st_without_empty_rows(fn) for name, fn in EXTRACTOR_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Main extraction & flow
# ---------------------------------------------------------------------------

def extraction(
    instagram_zip: SeekableBinaryReader,
    validation,
) -> ExtractionResult:
    """Extract data from an Instagram DDP zip and return consent-form tables.

    Parameters
    ----------
    instagram_zip:
        Seekable binary reader over the Instagram DDP zip — the upload
        adapter itself, never a path (ADR-0026).
    validation:
        Validation result object whose ``archive_members`` attribute is passed
        to ``ZipArchiveReader``.
    """
    config = load_port_config(EXTRACTOR_REGISTRY, "instagram")
    errors: Counter = Counter()
    reader = ZipArchiveReader(instagram_zip, validation.archive_members, errors)
    return run_extraction(reader, errors, config)


class InstagramFlow(FlowBuilder):
    """Flow implementation for the Instagram data donation study.

    Parameters
    ----------
    session_id:
        Unique identifier for the current participant session.
    """

    def __init__(self, session_id: str):
        super().__init__(session_id, "Instagram")

    def validate_file(self, file):
        return validate.validate_zip(DDP_CATEGORIES, file)

    def extract_data(self, file_value, validation):
        return extraction(file_value, validation)


def process(session_id):
    flow = InstagramFlow(session_id)
    return flow.start_flow()