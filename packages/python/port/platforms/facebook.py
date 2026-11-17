"""
Facebook

This module contains an example flow of a Facebook data donation study

Assumptions:
It handles DDPs in the english language with filetype JSON.

Configuration
-------------
The ``extraction`` function is driven by ``port_config.json``.  Generate one with::

    pnpm generate-config facebook

Each extractor function carries its own table config in a ``Table config::``
JSON block inside its docstring.  The generator reads those blocks and
assembles the JSON file.

Platform info::

    {
        "name": "Facebook",
        "filetypes": ["json"],
        "languages": ["en", "nl", "de", "pl", "tr", "ar", "ru", "it", "ro", "es", "sq"],
        "description": "Handles DDPs in English. These data donation flows have not been tested yet, if you find anything wrong with them report to datadonation@uu.nl and they will be fixed!",
        "time_last_tested": "not yet implemented"
    }
"""

import functools
import logging
import re
from collections import Counter
from typing import Callable, cast 

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
        # Present in your real Facebook exports; absent from the real Instagram export
        "your_posts__check_ins__photos_and_videos_1.json",  # nearly universal — anyone who has posted
        "pages_you've_liked.json",
        "pages_and_profiles_you_follow.json",
        "who_you've_followed.json",
        "time_spent_on_facebook.json",
        "likes_and_reactions.json", "likes_and_reactions_1.json",
        "your_page_or_groups_badges.json",
        "payment_history.json",
        "fundraiser_posts_you_likely_viewed.json",
        "your_fundraiser_donations_information.json",
        "your_transaction_survey_information.json",
        "your_marketplace_device_history.json",
        "facebook_new_user_guide.json",
        "your_information_download_requests.json",
        "your_facebook_story_preferences.json",
        "reels_preferences.json",
        "your_camera_roll_controls.json",
        "device_navigation_bar_information.json",
        "navigation_bar_shortcut_history.json",
        ]
    ),
]


def story_views_in_past_7_days_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Number of stories the participant viewed in the past 7 days.

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
        Columns: ``Count``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Number of stories the participant viewed in the past 7 days.",
          "source_file": "ads_information/story_views_in_past_7_days.json",
          "columns": {
            "Count": "Number reported by Meta in this file."
          }
        }

    Table config::

        {
          "id": "facebook_story_views_in_past_7_days",
          "title": {
            "en": "Story views in the past 7 days",
            "nl": "Bekeken stories in de afgelopen 7 dagen",
            "de": "Story-Aufrufe der letzten 7 Tage",
            "pl": "Wyświetlenia relacji w ostatnich 7 dniach",
            "tr": "Son 7 gündeki hikaye görüntülemeleri",
            "ar": "مشاهدات القصص خلال آخر 7 أيام",
            "ru": "Просмотры историй за последние 7 дней",
            "it": "Storie visualizzate negli ultimi 7 giorni",
            "ro": "Vizualizări de povești în ultimele 7 zile",
            "es": "Historias vistas en los últimos 7 días",
            "sq": "Shikimet e stories në 7 ditët e fundit"
          },
          "description": {
            "en": "This table shows how many stories you viewed on Facebook in the past 7 days.",
            "nl": "Deze tabel toont hoeveel stories je de afgelopen 7 dagen op Facebook hebt bekeken.",
            "de": "Diese Tabelle zeigt, wie viele Storys Sie in den letzten 7 Tagen auf Facebook angesehen haben.",
            "pl": "Ta tabela pokazuje, ile relacji wyświetliłeś/aś na Facebooku w ostatnich 7 dniach.",
            "tr": "Bu tablo, son 7 günde Facebook'ta kaç hikaye görüntülediğini gösterir.",
            "ar": "يعرض هذا الجدول عدد القصص التي شاهدتها على فيسبوك خلال آخر 7 أيام.",
            "ru": "В этой таблице показано, сколько историй вы просмотрели на Facebook за последние 7 дней.",
            "it": "Questa tabella mostra quante storie hai visualizzato su Facebook negli ultimi 7 giorni.",
            "ro": "Acest tabel arată câte povești ai vizualizat pe Facebook în ultimele 7 zile.",
            "es": "Esta tabla muestra cuántas historias viste en Facebook en los últimos 7 días.",
            "sq": "Kjo tabelë tregon sa stories ke parë në Facebook në 7 ditët e fundit."
          },
          "headers": {
            "Count": {
              "en": "Number of stories",
              "nl": "Aantal stories",
              "de": "Anzahl der Storys",
              "pl": "Liczba relacji",
              "tr": "Hikaye sayısı",
              "ar": "عدد القصص",
              "ru": "Количество историй",
              "it": "Numero di storie",
              "ro": "Număr de povești",
              "es": "Número de historias",
              "sq": "Numri i stories"
            }
          }
        }
    """
    return _count_df(reader, errors, "ads_information/story_views_in_past_7_days.json")


    


# ---------------------------------------------------------------------------
# Author and content type from Meta's descriptive sentences
#
# Meta describes content with localised sentences such as "Xs Beitrag: ...",
# "X's post", "X gefällt Xs Kommentar." or "Publicación de X".
# The nouns below identify the kind of content in each language; the
# detected type is returned as a fixed English value (Post, Comment, Reel,
# Video, Photo, Link, Story) so that it is identical across export languages.
# English and German are verified against real exports; the other languages
# are best-effort.
# ---------------------------------------------------------------------------

_CONTENT_TYPE_NOUNS = {
    "Post": ["post", "posts", "beitrag", "beiträge", "bericht", "berichten", "publicación", "publicacion",
             "publicação", "publication", "postarea", "postare", "postimi", "postim", "gönderi",
             "gönderisi", "gönderisine", "публикация", "публикацию", "публикации", "запись", "منشور"],
    "Comment": ["comment", "kommentar", "opmerking", "reactie", "comentario", "comentário", "commento",
                "commentaire", "comentariul", "comentariu", "komentarz", "yorum", "yorumu", "yorumuna",
                "комментарий", "комментарию", "komenti", "koment", "تعليق"],
    "Reel": ["reel", "reels"],
    "Video": ["video", "videos", "vídeo", "videoclipul", "videoja", "videosu", "videosuna", "видео", "فيديو"],
    "Photo": ["photo", "foto", "fotos", "bild", "zdjęcie", "fotoğraf", "fotoğrafı", "фото",
              "фотографию", "fotografia", "fotografie", "صورة"],
    "Link": ["link", "enlace", "lien", "collegamento", "bağlantı", "ссылка", "رابط", "lidhja"],
    "Story": ["story", "storys", "historia", "storia", "povestea", "hikaye", "история"],
}
_NOUN_TO_TYPE = {noun: ctype for ctype, nouns in _CONTENT_TYPE_NOUNS.items() for noun in nouns}
_NOUN_PATTERN = "|".join(sorted(map(re.escape, _NOUN_TO_TYPE), key=len, reverse=True))
_OWNER_PREPOSITIONS = ["de", "di", "del", "della", "dello", "da", "do", "du", "des", "van", "von",
                       "lui", "al", "a", "i", "e", "od", "użytkownika"]

# "<owner>s Beitrag", "<owner>'s post", "<owner>' Beitrag", "<owner> Kommentar" (owner ending in s/x/z)
_POSSESSIVE_RE = re.compile(
    rf"^(?P<owner>.+?)(?:['’]s|['’]|s)?\s+(?P<noun>{_NOUN_PATTERN})$", re.IGNORECASE
)
# "Publicación de <owner>", "Post di <owner>", "Bericht van <owner>"
_PREFIX_RE = re.compile(
    rf"^(?P<noun>{_NOUN_PATTERN})\s+(?:{'|'.join(_OWNER_PREPOSITIONS)})\s+(?P<owner>.+)$", re.IGNORECASE
)

# Verb phrases around the owner in reaction titles ("X gefällt X's Beitrag.",
# "X hat auf Xs Beitrag reagiert.", "X likes X's post.").
_REACTION_VERB_PREFIXES = sorted([
    "gefällt", "hat auf", "reagierte auf", "likes", "liked", "loves", "reacted to",
    "vindt", "heeft gereageerd op", "reageerde op", "le gusta", "reaccionó a", "ha reagito a",
    "a reacționat la", "reagowała na", "reagował na",
], key=len, reverse=True)
_REACTION_VERB_SUFFIXES = ["reagiert", "leuk", "gereageerd"]


def _parse_content_sentence(text) -> tuple[str, str, str]:
    """Split a Meta content sentence into ``(owner, content_type, text)``.

    ``"Xs Beitrag: Schöner Tag"`` -> ``("X", "Post", "Schöner Tag")``.
    When no known pattern matches, owner and type are empty and the whole
    sentence is returned as text.

    German possessives drop the genitive "s", which is ambiguous for names
    that end in "s" themselves (Meta writes "Jonas Beitrag"); those names
    lose their final "s".
    """
    text = _clean_text(text) if text is not None else ""
    if not text:
        return "", "", ""
    head, _, rest = text.partition(": ")
    head = head.strip().rstrip(" :.")
    for pattern in (_POSSESSIVE_RE, _PREFIX_RE):
        match = pattern.match(head)
        if match:
            return (
                match.group("owner").strip(),
                _NOUN_TO_TYPE[match.group("noun").lower()],
                rest.strip(),
            )
    return "", "", text


def _content_type_from_url(url: str) -> str:
    """Infer the content type from a Facebook URL (empty when unclear)."""
    u = (url or "").lower()
    if "comment_id=" in u:
        return "Comment"
    if "/reel/" in u or "/reels/" in u:
        return "Reel"
    if "/videos/" in u or "/watch" in u or "video.php" in u:
        return "Video"
    if "/photo" in u:
        return "Photo"
    if "/posts/" in u or "/permalink" in u or "story_fbid" in u or "story.php" in u:
        return "Post"
    return ""


_CONTENT_TYPE_LABELS = {
    "Post": "Beitrag",
    "Comment": "Kommentar",
    "Reel": "Reel",
    "Video": "Video",
    "Photo": "Foto",
    "Link": "Link",
    "Story": "Story",
}


def _content_type_label(content_type: str) -> str:
    """Display value for a detected content type (German, as in the study)."""
    return _CONTENT_TYPE_LABELS.get(content_type, content_type)


def _combine_content_type(url_type: str, sentence_type: str) -> str:
    """URL types that are more specific than Meta's wording win (a Reel is often called "Beitrag")."""
    if url_type in ("Comment", "Reel", "Video", "Photo"):
        return _content_type_label(url_type)
    return _content_type_label(sentence_type or url_type)


def _parse_reaction_title(title, actor) -> tuple[str, str]:
    """Return ``(owner, content_type)`` from an older-format reaction title."""
    text = _clean_text(title) if title else ""
    actor = _clean_text(actor) if actor else ""
    if actor and text.startswith(actor):
        text = text[len(actor):].strip()
    lowered = text.lower()
    for prefix in _REACTION_VERB_PREFIXES:
        if lowered.startswith(prefix + " "):
            text = text[len(prefix):].strip()
            break
    text = text.rstrip(" .!")
    for suffix in _REACTION_VERB_SUFFIXES:
        if text.lower().endswith(" " + suffix):
            text = text[: -len(suffix)].strip()
            break
    owner, ctype, _ = _parse_content_sentence(text)
    if not ctype:
        for word in reversed(re.findall(r"[\w'’]+", text)):
            if word.lower() in _NOUN_TO_TYPE:
                ctype = _NOUN_TO_TYPE[word.lower()]
                break
    return owner, ctype


# ---------------------------------------------------------------------------
# Participant's own name in activity titles
# ---------------------------------------------------------------------------

OWN_NAME_PLACEHOLDER = "Sie selbst"

# Words that follow the actor's name in Meta's activity titles
# ("X X hat einen Link gespeichert.", "X X saved a link.").
_ACTOR_MARKERS = (
    " hat ", " has ", " saved ", " heeft ", " ha ", " a salvat", " a ", " zapisał",
    " guardó ", " salvó ", " сохранил", " ruajti", " kaydetti",
)


def _own_name_in_titles(titles: list[str]) -> str:
    """Return the participant's name as it appears at the start of activity titles.

    The name is the text before the first marker word; it is accepted when it
    is the same in at least half of the titles (it is always the same person).
    """
    candidates: Counter = Counter()
    for title in titles:
        positions = [title.find(m) for m in _ACTOR_MARKERS if title.find(m) > 0]
        if positions:
            candidates[title[: min(positions)].strip()] += 1
    if not candidates:
        return ""
    name, count = candidates.most_common(1)[0]
    return name if count >= max(1, len(titles) / 2) else ""


# ---------------------------------------------------------------------------
# Category extractors (settings, security, interactions)
#
# These tables bundle many small Facebook files into a handful of tables so
# participants are not shown dozens of near-empty tables.  The files are
# heterogeneous: most use the localized ``label_values`` structure (labels are
# in the participant's Facebook language), some use plain English-key dicts.
# The helpers below therefore walk the structure generically instead of
# relying on language-specific labels, and drop anything that looks like an
# identifier (IP address, e-mail, phone number, cookie, user agent).
# ---------------------------------------------------------------------------


#: Meta leaves some labels untranslated (English) in otherwise German exports.
#: Known ones are mapped to German here; unknown strings are kept as exported.
_META_ENGLISH_TO_GERMAN = {
    "Hide an Ad": "Werbeanzeige verborgen",
    "Report an Ad": "Werbeanzeige gemeldet",
    "Added interests": "Hinzugefügte Interessen",
    "Reported interests": "Gemeldete Interessen",
    "Sensitive topics opted out of": "Abgewählte sensible Themen",
    "Sensitive topics opted in to": "Ausgewählte sensible Themen",
    "Datafile Custom Audience inclusion opt-out": "Datendatei Custom Audience: Einbeziehung deaktiviert",
    "Datafile Custom Audience exclusion opt-out": "Datendatei Custom Audience: Ausschluss deaktiviert",
    "Combined number of times you've used these privacy settings":
        "Gesamtzahl der Verwendungen dieser Datenschutzeinstellungen",
    "The level of permission Facebook has to use your Camera Roll data for each feature":
        "Berechtigungsstufe, die Facebook für die Nutzung deiner Kamerarollen-Daten pro Funktion hat",
    "Dismiss Click": "Schließen-Klick",
    "Feed Comments": "Feed-Kommentare",
    "Groups": "Gruppen",
    "Notes": "Notizen",
    "Other updates from Facebook": "Weitere Updates von Facebook",
    "Pages": "Seiten",
    "Photos": "Fotos",
    "Translations": "Übersetzungen",
    "Events": "Veranstaltungen",
    "Audited translations updates": "Updates zu geprüften Übersetzungen",
    "Translation task activity": "Aktivität bei Übersetzungsaufgaben",
    "You being tagged in a video": "Videos, auf denen du markiert wirst",
}
_RE_IN_APP_MESSAGE = re.compile(r"^In-app Message\b", re.IGNORECASE)


def _clean_text(value) -> str:
    if isinstance(value, bool):
        # Match the Richtig / Falsch wording Meta uses for booleans in German exports.
        return "Richtig" if value else "Falsch"
    text = eh.fix_latin1_string(str(value)).strip()
    text = _META_ENGLISH_TO_GERMAN.get(text, text)
    return _RE_IN_APP_MESSAGE.sub("In-App-Nachricht", text)


# ---------------------------------------------------------------------------
# No URLs that can lead to a person's account
#
# The export does not say whether an account is private, so URLs are removed
# wherever content or accounts of persons may be involved (profiles, posts,
# reels, comments, ...).  They are kept only where the other side is public by
# definition: ads, liked Facebook pages and external websites in the link
# history.
# ---------------------------------------------------------------------------


def _count_df(reader: ZipArchiveReader, errors: Counter, path: str) -> pd.DataFrame:
    """Return a one-cell ``Count`` table with the first number found in *path*."""
    data = _st_read(reader, errors, path)
    if data is None:
        return pd.DataFrame()
    queue = [data]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            value = node.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return pd.DataFrame({"Count": [int(value)]})
            if isinstance(value, str):
                digits = re.findall(r"\d+", value)
                if digits:
                    return pd.DataFrame({"Count": [int("".join(digits))]})
            queue.extend(v for k, v in node.items() if "timestamp" not in str(k).lower())
        elif isinstance(node, list):
            queue.extend(node)
    return pd.DataFrame()


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


def _st_by_name(name: str) -> str:
    """Regex for a file name in any folder (Threads excluded)."""
    return rf"^(?!.*threads)(?:.*/)?{re.escape(name)}$"


def _st_read(reader: ZipArchiveReader, errors: Counter, path: str, *alternatives: str):
    """Read *path*; else one of *alternatives* (older names); else the same file name in any folder.

    Meta moves files between folders from time to time, so the last step finds
    a file whose folder changed, as long as its name is unique in the export.
    """
    candidates = (path,) + alternatives
    for candidate in candidates:
        try:
            result = reader.json(candidate)
        except Exception as e:  # unreadable file
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
            continue
        if result.found:
            return result.data
    for candidate in candidates:
        found = _st_json_all(reader, errors, _st_by_name(candidate.rsplit("/", 1)[-1]))
        if len(found) == 1:
            return found[0]
    return None


def _st_read_numbered(reader: ZipArchiveReader, errors: Counter, folder_and_stem: str, *alternatives: str) -> list:
    """Read ``<stem>.json`` and all numbered variants ``<stem>_1.json``, ``<stem>_2.json`` ...

    *alternatives* are older names of the same file (without ``.json``).
    """
    for candidate in (folder_and_stem,) + alternatives:
        stem = candidate.rsplit("/", 1)[-1]
        found = _st_json_all(reader, errors, rf"^(?!.*threads)(?:.*/)?{re.escape(stem)}(?:_\d+)?\.json$")
        if found:
            return found
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
        if df.empty:
            return pd.DataFrame()
        # Columns without any value in this export (e.g. no texts) are not shown.
        empty_columns = [
            c for c in df.columns
            if all(str(v).strip() in ("", "nan", "None", "NaT") for v in df[c])
        ]
        return df.drop(columns=empty_columns)
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
    # Ads that Meta shows in other apps (Meta Audience Network):
    "AD_REQUEST": "App hat Werbung von Meta angefordert",
    "AD_IMPRESSION": "Werbung von Meta in der App angezeigt",
    "AD_CLICK": "Werbung von Meta in der App angeklickt",
    "SCHEDULE": "Termin vereinbart",
    "CONTACT": "Kontakt aufgenommen",
    "CUSTOMIZE_PRODUCT": "Produkt angepasst",
    "DONATE": "Spende",
    "FIND_LOCATION": "Standort gesucht",
    "SUBMIT_APPLICATION": "Antrag/Bewerbung abgeschickt",
    "RATE": "Bewertung abgegeben",
    "ACHIEVE_LEVEL": "Level erreicht (Spiel)",
    "UNLOCK_ACHIEVEMENT": "Erfolg freigeschaltet (Spiel)",
    "COMPLETE_TUTORIAL": "Tutorial abgeschlossen",
    "SPENT_CREDITS": "Guthaben ausgegeben",
    "OTHER": "Sonstiges",
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
# Facebook tables of the study
# ---------------------------------------------------------------------------

# Words around the names in Facebook activity sentences
# ("A hat Bs Beitrag kommentiert.", "A commented on B's post.").
_FB_TITLE_LEAD_WORDS = {
    "hat", "has", "heeft", "ha", "har", "on", "op", "auf", "to", "sur", "su", "na", "zu",
    "commented", "kommentiert", "replied", "geantwortet", "antwortete", "reacted", "reagiert",
    "reagierte", "saved", "gespeichert", "liked", "likes", "loves", "gefällt", "gereageerd",
    "reageerde", "opgeslagen", "gaf", "gegeven", "commentaar", "comentó", "comentou", "commentato",
    "commenté", "comentat", "skomentował", "skomentowała", "zareagował", "zareagowała", "zapisał",
    "zapisała", "guardó", "salvó", "salvato", "salvat", "yorum", "yaptı", "kaydetti",
    "прокомментировал", "прокомментировала", "отреагировал", "отреагировала", "сохранил",
    "сохранила", "komentoi", "ruajti", "reagoi", "a",
}
_FB_ARTICLES = {
    "der", "die", "das", "den", "dem", "des", "einen", "eine", "ein", "einem", "einer", "the", "a", "an",
    "de", "het", "een", "la", "el", "le", "lo", "il", "un", "una", "uno", "o", "os", "as", "les",
    "los", "las", "al", "del", "un", "o",
}
_FB_OWN_WORDS = {
    "own", "eigenen", "eigene", "eigener", "eigenes", "eigen", "propio", "propia", "proprio",
    "propria", "propre", "propriul", "własny", "własnego", "własnym", "własną", "kendi",
    "свой", "своей", "свою", "своего", "saját",
}
_FB_POSSESSIVE_PRONOUNS = {
    "his", "her", "their", "seinen", "ihren", "seine", "ihre", "seinem", "ihrem", "zijn", "haar",
    "hun", "sus", "suo", "sua", "son", "sa", "ses", "swój", "swojego", "swoim", "swoją",
}


def _fb_strip_actor(title: str, actor: str) -> str:
    """Remove the participant's name at the start of an activity sentence.

    When the name is unknown or written differently, everything before the
    first verb ("hat", "has", "heeft", ...) is removed, so the participant's
    name never ends up in the account column.
    """
    title = _st_fix(title).strip()
    if actor and title.startswith(actor):
        return title[len(actor):].strip()
    positions = [title.find(m) for m in _ACTOR_MARKERS + (" gefällt ", " likes ", " reacted ") if title.find(m) > 0]
    if positions:
        return title[min(positions):].strip()
    return title


def _fb_title_owner(title, actor: str) -> str:
    """Name of the person/page whose content an activity sentence is about.

    ``"A hat Bs Beitrag kommentiert."`` -> ``"B"``; own content -> ``OWN_NAME_PLACEHOLDER``.
    """
    rest = _fb_strip_actor(title, actor).rstrip(" .!")
    # "mit „Gefällt mir“ markiert" / "with ‘Like’"
    rest = re.sub(r"\s+(?:mit|with|met)\s+[„“\"'‘’«»].*$", "", rest).strip()
    if not rest:
        return ""
    words = rest.split()
    if any(w.casefold() in _FB_OWN_WORDS for w in words):
        return OWN_NAME_PLACEHOLDER
    while words and words[0].casefold() in _FB_TITLE_LEAD_WORDS:
        words.pop(0)
    if len(words) > 1 and words[0].casefold() in _FB_POSSESSIVE_PRONOUNS and words[1].casefold() in _NOUN_TO_TYPE:
        return OWN_NAME_PLACEHOLDER  # "X hat ihren Beitrag gespeichert." / "X saved his post."
    while words and words[-1].casefold() in _FB_TITLE_LEAD_WORDS:
        words.pop()
    if len(words) > 1 and words[0].casefold() in _FB_ARTICLES and words[1].casefold() in _NOUN_TO_TYPE:
        words.pop(0)  # "la publicación de B"
    owner, _, _ = _parse_content_sentence(" ".join(words))
    # "ein Video von Y Zs Beitrag" -> "Y Z"
    of_match = re.match(r"^(?:\S+\s+)?(?:von|of|van|de|del|di|du|da)\s+(.+)$", owner, re.I)
    if of_match and owner.split()[0].casefold() in set(_NOUN_TO_TYPE) | _FB_ARTICLES:
        owner = of_match.group(1)
    if owner.casefold() in _FB_ARTICLES | _FB_TITLE_LEAD_WORDS:
        return ""  # "einen Beitrag", "a link": nobody named
    return owner


def _fb_actor(titles: list) -> str:
    """The participant's own name at the start of activity sentences (not shown in tables)."""
    return _own_name_in_titles([_st_fix(t) for t in titles if t])


def _fb_label_values_entries(data) -> list:
    """All (label, entry) pairs below a label_values file, including vec/dict members."""
    found = []

    def walk(node, section=""):
        if isinstance(node, list):
            for child in node:
                walk(child, section)
        elif isinstance(node, dict):
            if "label_values" in node:
                walk(node["label_values"], section)
                return
            if isinstance(node.get("vec"), list):
                walk(node["vec"], _st_fix(node.get("label") or node.get("title") or section))
            elif isinstance(node.get("dict"), list):
                found.append((section, node["dict"]))
    walk(data)
    return found


def facebook_feed_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
          "source_file": "logged_information/interactions/content_that_has_been_shown_to_you_in_your_feed.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "Text": "Text of the content (caption or post text), if the export contains it.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_feed",
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
                "en": "The total number of Facebook posts you viewed over time",
                "nl": "Het totale aantal Facebook-berichten dat je in de loop van de tijd hebt bekeken",
                "de": "Die Gesamtzahl der Facebook-Beiträge, die Sie im Laufe der Zeit angesehen haben",
                "pl": "Łączna liczba postów na Facebookie, które wyświetliłeś/aś w czasie",
                "tr": "Zaman içinde görüntülediğin Facebook gönderilerinin toplam sayısı",
                "ar": "إجمالي عدد منشورات فيسبوك التي شاهدتها عبر الوقت",
                "ru": "Общее количество публикаций в Facebook, которые вы просмотрели с течением времени",
                "it": "Il numero totale di post di Facebook che hai visualizzato nel tempo",
                "ro": "Numărul total de postări de pe Facebook pe care le-ai vizualizat în timp",
                "es": "El número total de publicaciones de Facebook que viste a lo largo del tiempo",
                "sq": "Numri total i postimeve në Facebook që ke parë me kalimin e kohës"
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
                "en": "The total number of Facebook posts you have viewed per hour of the day",
                "nl": "Het totale aantal Facebook-berichten dat je per uur van de dag hebt bekeken",
                "de": "Die Gesamtzahl der Facebook-Beiträge, die Sie pro Stunde des Tages angesehen haben",
                "pl": "Łączna liczba postów na Facebookie, które wyświetliłeś/aś w poszczególnych godzinach doby",
                "tr": "Günün saatine göre görüntülediğin Facebook gönderilerinin toplam sayısı",
                "ar": "إجمالي عدد منشورات فيسبوك التي شاهدتها في كل ساعة من اليوم",
                "ru": "Общее количество публикаций в Facebook, которые вы просматривали по часам суток",
                "it": "Il numero totale di post di Facebook che hai visualizzato per ogni ora del giorno",
                "ro": "Numărul total de postări de pe Facebook pe care le-ai vizualizat pe oră din zi",
                "es": "El número total de publicaciones de Facebook que viste por hora del día",
                "sq": "Numri total i postimeve në Facebook që ke parë për çdo orë të ditës"
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
    rows = []
    data = _st_read(reader, errors, "interactions/content_that_has_been_shown_to_you_in_your_feed.json")
    if data is None:
        return pd.DataFrame()
    try:
        for _, fields in _fb_label_values_entries(data):
            sentence, url, ts = "", "", None
            for field in fields:
                if not isinstance(field, dict):
                    continue
                if _st_is_epoch(field.get("timestamp_value")):
                    ts = ts or field["timestamp_value"]
                elif _st_is_url(field.get("href")) or _st_is_url(field.get("value")):
                    url = url or field.get("href") or field.get("value")
                elif field.get("value") not in (None, ""):
                    sentence = sentence or _st_fix(field["value"])
            owner, _, text = _parse_content_sentence(sentence)
            rows.append((owner, text, url, _st_iso(ts, errors)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return _st_df(rows, ["Account", "Text", "URL", "Date"])


def _fb_ad_rows(data, errors: Counter) -> list:
    rows = []
    for record in _st_records(data):
        if not isinstance(record, dict):
            continue
        name, url = "", ""
        for label, entry in _st_top(record):
            value, href = entry.get("value"), entry.get("href")
            if _st_is_ad_library_url(value) or _st_is_ad_library_url(href):
                continue
            if value not in (None, "") and not _st_is_url(value) and not name:
                name = _st_fix(value)
                url = url or (href if _st_is_url(href) else "")
            elif not url and (_st_is_url(href) or _st_is_url(value)):
                url = href if _st_is_url(href) else value
        if not name:
            name = _st_fix(record.get("title") or "")
        rows.append((name, url, _st_iso(_st_record_ts(record), errors)))
    return rows


def facebook_ads_viewed_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
        Columns: ``Advertiser``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows ads that were shown to you, with the advertiser, the text (if available), the link and the time.",
          "source_file": "ads_information/ads_viewed.json (older exports: logged_information/interactions/ads.json)",
          "columns": {
            "Advertiser": "Name of the advertiser.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_ads_viewed",
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
                "column": "Advertiser",
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
    rows = []
    for path in ("ads_information/ads_viewed.json", "interactions/ads.json"):
        data = _st_read(reader, errors, path)
        if data is None:
            continue
        try:
            rows.extend(_fb_ad_rows(data, errors))
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    rows = list(dict.fromkeys(rows))
    return _st_df(rows, ["Advertiser", "URL", "Date"])


def facebook_likes_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
        Columns: ``Account``, ``Type``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows posts and comments you liked, with the account that posted them, the kind of content, the time and, for posts, the link.",
          "source_file": "your_facebook_activity/comments_and_reactions/likes_and_reactions*.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "Type": "Kind of content or interaction.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_likes",
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
            "en": "This table shows posts and comments you liked, with the account that posted them, the kind of content, the time and, for posts, the link.",
            "nl": "Deze tabel toont berichten en opmerkingen die je leuk vond, met het account dat ze plaatste, het soort inhoud, het tijdstip en bij berichten de link.",
            "de": "Diese Tabelle zeigt Beiträge und Kommentare, die Sie mit „Gefällt mir“ markiert haben, mit dem Account, der sie veröffentlicht hat, der Art des Inhalts, dem Zeitpunkt und bei Beiträgen dem Link.",
            "pl": "Ta tabela pokazuje polubione posty i komentarze wraz z kontem autora, rodzajem treści, czasem oraz — w przypadku postów — linkiem.",
            "tr": "Bu tablo, beğendiğin gönderileri ve yorumları; paylaşan hesap, içerik türü, zaman ve gönderiler için bağlantı ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول المنشورات والتعليقات التي أعجبتك، مع الحساب الناشر ونوع المحتوى والوقت، والرابط بالنسبة للمنشورات.",
            "ru": "В этой таблице показаны понравившиеся вам публикации и комментарии с аккаунтом автора, типом контента, временем и — для публикаций — ссылкой.",
            "it": "Questa tabella mostra i post e i commenti a cui hai messo Mi piace, con l'account che li ha pubblicati, il tipo di contenuto, l'ora e, per i post, il link.",
            "ro": "Acest tabel arată postările și comentariile apreciate, cu contul care le-a publicat, tipul conținutului, ora și, pentru postări, linkul.",
            "es": "Esta tabla muestra las publicaciones y comentarios que te gustaron, con la cuenta que los publicó, el tipo de contenido, la hora y, en las publicaciones, el enlace.",
            "sq": "Kjo tabelë tregon postimet dhe komentet që ke pëlqyer, me llogarinë që i postoi, llojin e përmbajtjes, kohën dhe për postimet lidhjen."
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
    files = _st_read_numbered(reader, errors, "comments_and_reactions/likes_and_reactions", "posts_and_comments")
    try:
        old_titles = [
            r.get("title", "") for d in files for r in _st_records(d)
            if isinstance(r, dict) and "label_values" not in r
        ]
        actor = _fb_actor(old_titles)
        # Both formats can describe the same like; the current format has more detail.
        current_times = {
            r.get("timestamp") for d in files for r in _st_records(d)
            if isinstance(r, dict) and "label_values" in r
        }
        for data in files:
            for record in _st_records(data):
                if not isinstance(record, dict):
                    continue
                ts = _st_iso(_st_record_ts(record), errors)
                if "label_values" in record:  # current format
                    url = _st_content_url(record)
                    username, name, _ = _st_owner(record)
                    owner = username or name or _st_top_value(record, "name")
                    text = _st_top_value(record, "title")
                    sentence_owner, sentence_type, sentence_text = _parse_content_sentence(text)
                    if sentence_owner:
                        owner, text = owner or sentence_owner, sentence_text
                    ctype = _combine_content_type(_content_type_from_url(url), sentence_type)
                else:  # older format: only a sentence ("A gefällt Bs Beitrag.")
                    if record.get("timestamp") in current_times:
                        continue
                    url, text = "", ""
                    record_actor = actor
                    for item in record.get("data") or []:
                        record_actor = _st_fix(((item or {}).get("reaction") or {}).get("actor") or "") or record_actor
                    title = _st_fix(record.get("title"))
                    owner, sentence_type = _parse_reaction_title(title, record_actor)
                    if (not owner or owner.casefold() in _FB_ARTICLES
                            or (record_actor and record_actor in owner)
                            or any(m in f" {owner} " for m in _ACTOR_MARKERS)):
                        owner = _fb_title_owner(title, record_actor)
                    ctype = _content_type_label(sentence_type)
                if ctype != "Beitrag" and ctype not in ("Reel", "Video", "Foto"):
                    url = ""  # the participant information names links for posts only
                rows.append((owner, ctype, url, ts))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    rows = list(dict.fromkeys(rows))
    rows.sort(key=lambda r: r[3], reverse=True)
    return _st_df(rows, ["Account", "Type", "URL", "Date"])


def facebook_story_interactions_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
          "source_file": "your_facebook_activity/stories/story_reactions.json, older_story_reactions.json, stories_you_reacted_to_with_a_sticker.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "Type": "Kind of content or interaction.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_story_interactions",
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
    sources = (
        ("stories/story_reactions.json", "Reaktion"),
        ("stories/older_story_reactions.json", "Reaktion"),
        ("stories/stories_you_reacted_to_with_a_sticker.json", "Sticker"),
    )
    for path, default_type in sources:
        data = _st_read(reader, errors, path)
        if data is None:
            continue
        try:
            for record in _st_records(data):
                if not isinstance(record, dict):
                    continue
                username, name, _ = _st_owner(record)
                kind = _st_top_value(record, "emoji") or _st_top_value(record, "reaction") or default_type
                rows.append((username or name, kind, _st_iso(_st_record_ts(record), errors)))
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    rows.sort(key=lambda r: r[2], reverse=True)
    return _st_df(rows, ["Account", "Type", "Date"])


def facebook_saved_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
          "summary": "This table shows posts and other items you saved, with the account that posted them (if known), the link (if available) and the time you saved them.",
          "source_file": "your_facebook_activity/saved_items_and_collections/your_saved_items.json",
          "columns": {
            "Account": "Username (or name) of the account that posted the content.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_saved",
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
            "en": "This table shows posts and other items you saved, with the account that posted them (if known), the link (if available) and the time you saved them.",
            "nl": "Deze tabel toont berichten en andere items die je hebt opgeslagen, met het account dat ze plaatste (indien bekend), de link (indien beschikbaar) en het tijdstip van opslaan.",
            "de": "Diese Tabelle zeigt Beiträge und andere Inhalte, die Sie gespeichert haben, mit dem Account, der sie veröffentlicht hat (falls bekannt), dem Link (falls vorhanden) und dem Zeitpunkt der Speicherung.",
            "pl": "Ta tabela pokazuje zapisane posty i inne elementy wraz z kontem autora (jeśli jest znane), linkiem (jeśli jest dostępny) i czasem zapisania.",
            "tr": "Bu tablo, kaydettiğin gönderileri ve diğer öğeleri; paylaşan hesap (biliniyorsa), bağlantı (varsa) ve kaydetme zamanı ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول المنشورات والعناصر الأخرى التي حفظتها، مع الحساب الناشر (إن كان معروفًا) والرابط (إن وجد) ووقت الحفظ.",
            "ru": "В этой таблице показаны сохранённые публикации и другие материалы с аккаунтом автора (если известен), ссылкой (если есть) и временем сохранения.",
            "it": "Questa tabella mostra i post e gli altri elementi che hai salvato, con l'account che li ha pubblicati (se noto), il link (se disponibile) e l'ora del salvataggio.",
            "ro": "Acest tabel arată postările și alte elemente salvate, cu contul care le-a publicat (dacă este cunoscut), linkul (dacă există) și ora salvării.",
            "es": "Esta tabla muestra las publicaciones y otros elementos que guardaste, con la cuenta que los publicó (si se conoce), el enlace (si lo hay) y la hora en que los guardaste.",
            "sq": "Kjo tabelë tregon postimet dhe elementet e tjera që ke ruajtur, me llogarinë që i postoi (nëse dihet), lidhjen (nëse ka) dhe kohën e ruajtjes."
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
    data = _st_read(reader, errors, "saved_items_and_collections/your_saved_items.json")
    if data is None:
        return pd.DataFrame()
    rows = []
    try:
        records = [r for r in _st_records(data) if isinstance(r, dict)]
        actor = _fb_actor([r.get("title", "") for r in records])
        for record in records:
            owner = _fb_title_owner(record.get("title", ""), actor)
            context_name, url = "", ""
            for attachment in record.get("attachments") or []:
                for item in (attachment or {}).get("data") or []:
                    context = (item or {}).get("external_context") or {}
                    context_name = context_name or _st_fix(context.get("name") or "")
                    url = url or (context.get("url") if _st_is_url(context.get("url")) else "")
            rows.append((owner or context_name, url, _st_iso(record.get("timestamp"), errors)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return _st_df(rows, ["Account", "URL", "Date"])


def facebook_comments_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
          "source_file": "your_facebook_activity/comments_and_reactions/comments.json",
          "columns": {
            "Comment": "Text of the participant's comment.",
            "Account": "Username (or name) of the account that posted the content.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_comments",
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
    data = _st_read(reader, errors, "comments_and_reactions/comments.json")
    if data is None:
        return pd.DataFrame()
    rows = []
    try:
        records = [r for r in _st_records(data) if isinstance(r, dict)]
        common_actor = _fb_actor([r.get("title", "") for r in records])
        for record in records:
            text, author = "", ""
            for item in record.get("data") or []:
                comment = (item or {}).get("comment") or {}
                text = text or _st_fix(comment.get("comment") or "")
                author = author or _st_fix(comment.get("author") or "")
            owner = _fb_title_owner(record.get("title", ""), author or common_actor)
            if not text:
                # Comment without text: only say what kind of attachment it was.
                # Attachment contents (links, photos with upload IP) are never read.
                kinds = []
                for attachment in record.get("attachments") or []:
                    for item in (attachment or {}).get("data") or []:
                        item = item or {}
                        media = item.get("media") or {}
                        if "ai_stickers" in media or "sticker" in str(media.get("uri", "")).lower():
                            kinds.append("[Sticker]")
                        elif media:
                            kinds.append("[Foto/Video]")
                        elif item.get("external_context"):
                            kinds.append("[Link]")
                text = " ".join(dict.fromkeys(kinds))
            rows.append((text, owner, _st_iso(record.get("timestamp"), errors)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return _st_df(rows, ["Comment", "Account", "Date"])


def facebook_searches_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
        Columns: ``Search``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows what you searched for (e.g. profiles, words or hashtags), with the search text, the link of the result (if available) and the time.",
          "source_file": "logged_information/search/your_search_history.json",
          "columns": {
            "Search": "Search text (word, hashtag or profile name).",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_searches",
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
    data = _st_read(reader, errors, "search/your_search_history.json")
    if data is None:
        return pd.DataFrame()
    rows = []
    try:
        for record in _st_records(data):
            if not isinstance(record, dict):
                continue
            text = ""
            for item in record.get("data") or []:
                text = text or _st_fix((item or {}).get("text") or "")
            if not text:
                for attachment in record.get("attachments") or []:
                    for item in (attachment or {}).get("data") or []:
                        text = text or _st_fix((item or {}).get("text") or "")
            rows.append((text.strip('"“”'), _st_iso(record.get("timestamp"), errors)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return _st_df(rows, ["Search", "Date"])


def _fb_name_rows(reader, errors, path, *alternatives) -> list:
    data = _st_read(reader, errors, path, *alternatives)
    rows = []
    if data is None:
        return rows
    try:
        for record in _st_records(data):
            if not isinstance(record, dict):
                continue
            name = record.get("name") or ""
            for item in record.get("data") or []:
                name = name or (item or {}).get("name") or ""
            name = _st_fix(name or record.get("title") or "")
            rows.append((name, _st_iso(record.get("timestamp"), errors)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return rows


def facebook_friends_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Your friends.

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
        Columns: ``Name``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows your Facebook friends and since when you are connected.",
          "source_file": "connections/friends/your_friends.json",
          "columns": {
            "Name": "Name of the person, page or account.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_friends",
          "title": {
            "en": "Your friends",
            "nl": "Je vrienden",
            "de": "Ihre Freund*innen",
            "pl": "Twoi znajomi",
            "tr": "Arkadaşların",
            "ar": "أصدقاؤك",
            "ru": "Ваши друзья",
            "it": "I tuoi amici",
            "ro": "Prietenii tăi",
            "es": "Tus amigos",
            "sq": "Miqtë e tu"
          },
          "description": {
            "en": "This table shows your Facebook friends and since when you are connected.",
            "nl": "Deze tabel toont je Facebook-vrienden en sinds wanneer jullie verbonden zijn.",
            "de": "Diese Tabelle zeigt Ihre Facebook-Freund*innen und seit wann Sie verbunden sind.",
            "pl": "Ta tabela pokazuje Twoich znajomych na Facebooku i od kiedy jesteście połączeni.",
            "tr": "Bu tablo, Facebook arkadaşlarını ve ne zamandan beri bağlantılı olduğunuzu gösterir.",
            "ar": "يعرض هذا الجدول أصدقاءك على فيسبوك ومنذ متى أنتم متصلون.",
            "ru": "В этой таблице показаны ваши друзья на Facebook и с какого времени вы связаны.",
            "it": "Questa tabella mostra i tuoi amici di Facebook e da quando siete collegati.",
            "ro": "Acest tabel arată prietenii tăi de pe Facebook și de când sunteți conectați.",
            "es": "Esta tabla muestra tus amigos de Facebook y desde cuándo estáis conectados.",
            "sq": "Kjo tabelë tregon miqtë e tu në Facebook dhe që kur jeni të lidhur."
          },
          "headers": {
            "Name": {
              "en": "Name",
              "nl": "Naam",
              "de": "Name",
              "pl": "Nazwa",
              "tr": "Ad",
              "ar": "الاسم",
              "ru": "Имя",
              "it": "Nome",
              "ro": "Nume",
              "es": "Nombre",
              "sq": "Emri"
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
    return _st_df(_fb_name_rows(reader, errors, "friends/your_friends.json", "friends.json"), ["Name", "Date"])


def facebook_following_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
        Columns: ``Name``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows the accounts, pages and profiles you follow and since when.",
          "source_file": "connections/followers/who_you've_followed.json, your_facebook_activity/pages/pages_and_profiles_you_follow.json",
          "columns": {
            "Name": "Name of the person, page or account.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_following",
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
            "Name": {
              "en": "Name",
              "nl": "Naam",
              "de": "Name",
              "pl": "Nazwa",
              "tr": "Ad",
              "ar": "الاسم",
              "ru": "Имя",
              "it": "Nome",
              "ro": "Nume",
              "es": "Nombre",
              "sq": "Emri"
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
    rows = _fb_name_rows(reader, errors, "who_you've_followed.json", "following.json")
    rows += _fb_name_rows(reader, errors, "pages/pages_and_profiles_you_follow.json")
    rows = list(dict.fromkeys(rows))
    rows.sort(key=lambda r: r[1], reverse=True)
    return _st_df(rows, ["Name", "Date"])


def facebook_interests_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
        Columns: ``Topic``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows interests and topics that were assigned to you based on your activity and that influence which posts and ads you see.",
          "source_file": "logged_information/other_logged_information/ads_interests.json",
          "columns": {
            "Topic": "Name of the topic or interest."
          }
        }

    Table config::

        {
          "id": "facebook_interests",
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
            }
          }
        }
    """
    data = _st_read(reader, errors, "other_logged_information/ads_interests.json")
    rows = []
    try:
        for record in _st_records(data) if data is not None else []:
            if isinstance(record, str):
                rows.append((_st_fix(record),))
            elif isinstance(record, dict):
                name = record.get("name") or _st_top_value(record, "name")
                if name:
                    rows.append((_st_fix(name),))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return _st_df(list(dict.fromkeys(rows)), ["Topic"])


def facebook_link_history_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
          "source_file": "your_facebook_activity/other_activity/link_history.json",
          "columns": {
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_link_history",
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
    data = _st_read(reader, errors, "other_activity/link_history.json")
    if data is None:
        return pd.DataFrame()
    rows = []
    try:
        for record in _st_records(data):
            if isinstance(record, dict):
                rows.append((_st_content_url(record), _st_iso(_st_record_ts(record) or _st_text_time(record), errors)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return _st_df(rows, ["URL", "Date"])


def facebook_off_meta_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
          "source_file": "apps_and_websites_off_of_facebook/your_activity_off_meta_technologies.json",
          "columns": {
            "Website": "Name of the website or app that shared the activity.",
            "Event": "Type of activity, translated to German where known (e.g. PAGE_VIEW -> Seitenaufruf).",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_off_meta",
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
    data = _st_read(reader, errors, "apps_and_websites_off_of_facebook/your_activity_off_meta_technologies.json")
    if data is None:
        return pd.DataFrame()
    try:
        rows = _st_off_meta_rows(data, errors)
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        rows = []
    rows.sort(key=lambda r: r[2], reverse=True)
    return _st_df(rows, ["Website", "Event", "Date"])


def facebook_advertisers_using_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
          "source_file": "ads_information/advertisers_using_your_activity_or_information.json",
          "columns": {
            "Advertiser": "Name of the advertiser.",
            "Usage": "How the advertiser used the participant's information, as named in the export."
          }
        }

    Table config::

        {
          "id": "facebook_advertisers_using",
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
    data = _st_read(reader, errors, "advertisers_using_your_activity_or_information.json")
    if data is None:
        return pd.DataFrame()
    rows = []
    try:
        rows = [(value, label) for label, value in _st_vec_values(data)]
        for record in _st_records(data):  # older format: custom_audiences_all_types_v2
            if isinstance(record, dict) and record.get("advertiser_name"):
                kinds = [k for k in ("has_data_file_custom_audience", "has_remarketing_custom_audience",
                                     "has_in_person_store_visit") if record.get(k)]
                rows.append((_st_fix(record["advertiser_name"]), ", ".join(kinds)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return _st_df(rows, ["Advertiser", "Usage"])


_ST_ACTION_LABELS = [
    "Action", "Handlung", "Aktion", "Actie", "Działanie", "İşlem", "Eylem", "الإجراء", "Действие",
    "Azione", "Acțiune", "Acción", "Veprimi",
]


def facebook_advertisers_interacted_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Your interactions with advertisers.

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
        Columns: ``Event``, ``Advertiser``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows your interactions with ads of companies and organisations, with the action, the ad, the link and the time.",
          "source_file": "ads_information/advertisers_you've_interacted_with.json",
          "columns": {
            "Event": "Type of activity, translated to German where known (e.g. PAGE_VIEW -> Seitenaufruf).",
            "Advertiser": "Name of the advertiser.",
            "URL": "Link to the content. For content of private accounts the link cannot be opened by others.",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_advertisers_interacted",
          "title": {
            "en": "Your interactions with advertisers",
            "nl": "Je interacties met adverteerders",
            "de": "Ihre Interaktionen mit Werbetreibenden",
            "pl": "Twoje interakcje z reklamodawcami",
            "tr": "Reklamverenlerle etkileşimlerin",
            "ar": "تفاعلاتك مع المعلنين",
            "ru": "Ваши взаимодействия с рекламодателями",
            "it": "Le tue interazioni con gli inserzionisti",
            "ro": "Interacțiunile tale cu agenții de publicitate",
            "es": "Tus interacciones con anunciantes",
            "sq": "Ndërveprimet e tua me reklamuesit"
          },
          "description": {
            "en": "This table shows your interactions with ads of companies and organisations, with the action, the ad, the link and the time.",
            "nl": "Deze tabel toont je interacties met advertenties van bedrijven en organisaties, met de actie, de advertentie, de link en het tijdstip.",
            "de": "Diese Tabelle zeigt Ihre Interaktionen mit Werbung von Unternehmen und Organisationen, mit der Handlung, der Anzeige, dem Link und dem Zeitpunkt.",
            "pl": "Ta tabela pokazuje Twoje interakcje z reklamami firm i organizacji wraz z działaniem, reklamą, linkiem i czasem.",
            "tr": "Bu tablo, şirketlerin ve kuruluşların reklamlarıyla etkileşimlerini; işlem, reklam, bağlantı ve zaman ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول تفاعلاتك مع إعلانات الشركات والمؤسسات، مع الإجراء والإعلان والرابط والوقت.",
            "ru": "В этой таблице показаны ваши взаимодействия с рекламой компаний и организаций: действие, объявление, ссылка и время.",
            "it": "Questa tabella mostra le tue interazioni con le inserzioni di aziende e organizzazioni, con l'azione, l'inserzione, il link e l'ora.",
            "ro": "Acest tabel arată interacțiunile tale cu reclamele companiilor și organizațiilor, cu acțiunea, reclama, linkul și ora.",
            "es": "Esta tabla muestra tus interacciones con anuncios de empresas y organizaciones, con la acción, el anuncio, el enlace y la hora.",
            "sq": "Kjo tabelë tregon ndërveprimet e tua me reklamat e kompanive dhe organizatave, me veprimin, reklamën, lidhjen dhe kohën."
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
    data = _st_read(reader, errors, "advertisers_you've_interacted_with.json")
    if data is None:
        return pd.DataFrame()
    actions = {_st_norm(a) for a in _ST_ACTION_LABELS}
    rows = []
    try:
        for record in _st_records(data):
            if not isinstance(record, dict):
                continue
            action, title, url = "", "", ""
            for label, entry in _st_top(record):
                value, href = entry.get("value"), entry.get("href")
                if _st_is_ad_library_url(value) or _st_is_ad_library_url(href):
                    continue
                if _st_is_url(href) or _st_is_url(value):
                    url = url or (href if _st_is_url(href) else value)
                elif value not in (None, ""):
                    if _st_norm(label) in actions and not action:
                        action = _st_fix(value)
                    elif not title:
                        title = _st_fix(value)
            if not title and record.get("title"):
                title = _st_fix(record["title"])
            rows.append((action, title, url, _st_iso(_st_record_ts(record), errors)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return _st_df(rows, ["Event", "Advertiser", "URL", "Date"])


def facebook_other_categories_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
          "source_file": "ads_information/other_categories_used_to_reach_you.json",
          "columns": {
            "Category": "Category used to reach the participant with ads."
          }
        }

    Table config::

        {
          "id": "facebook_other_categories",
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
    data = _st_read(reader, errors, "other_categories_used_to_reach_you.json")
    if data is None:
        return pd.DataFrame()
    try:
        rows = [(value,) for _, value in _st_vec_values(data)]
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        rows = []
    return _st_df(rows, ["Category"])


def facebook_ad_topics_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Ad topics.

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
        Columns: ``Topic``, ``List``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "This table shows topics and interests that Meta assigned to you for advertising, and the list in your export in which they appear.",
          "source_file": "ads_information/ad_preferences.json",
          "columns": {
            "Topic": "Name of the topic or interest.",
            "List": "List of Meta's export the topic appears in (e.g. activated or deactivated ad topics)."
          }
        }

    Table config::

        {
          "id": "facebook_ad_topics",
          "title": {
            "en": "Ad topics",
            "nl": "Advertentieonderwerpen",
            "de": "Werbethemen",
            "pl": "Tematy reklam",
            "tr": "Reklam konuları",
            "ar": "مواضيع الإعلانات",
            "ru": "Темы рекламы",
            "it": "Argomenti delle inserzioni",
            "ro": "Subiecte ale reclamelor",
            "es": "Temas de anuncios",
            "sq": "Temat e reklamave"
          },
          "description": {
            "en": "This table shows topics and interests that Meta assigned to you for advertising, and the list in your export in which they appear.",
            "nl": "Deze tabel toont onderwerpen en interesses die Meta voor advertenties aan je heeft toegewezen, en de lijst in je export waarin ze staan.",
            "de": "Diese Tabelle zeigt Themen und Interessen, die Meta Ihnen für Werbung zugeordnet hat, und die Liste in Ihrem Export, in der sie stehen.",
            "pl": "Ta tabela pokazuje tematy i zainteresowania przypisane Ci przez Meta na potrzeby reklam oraz listę w Twoim eksporcie, w której się znajdują.",
            "tr": "Bu tablo, Meta'nın reklam için sana atadığı konuları ve ilgi alanlarını ve bunların dışa aktarımında yer aldığı listeyi gösterir.",
            "ar": "يعرض هذا الجدول المواضيع والاهتمامات التي خصصتها لك Meta للإعلانات، والقائمة التي تظهر فيها في ملف التصدير.",
            "ru": "В этой таблице показаны темы и интересы, присвоенные вам Meta для рекламы, и список в вашем экспорте, в котором они указаны.",
            "it": "Questa tabella mostra gli argomenti e gli interessi che Meta ti ha assegnato per la pubblicità e l'elenco del tuo export in cui compaiono.",
            "ro": "Acest tabel arată subiectele și interesele atribuite ție de Meta pentru publicitate și lista din exportul tău în care apar.",
            "es": "Esta tabla muestra los temas e intereses que Meta te asignó para la publicidad y la lista de tu exportación en la que aparecen.",
            "sq": "Kjo tabelë tregon temat dhe interesat që Meta të ka caktuar për reklama dhe listën në eksportin tënd ku shfaqen."
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
            "List": {
              "en": "List",
              "nl": "Lijst",
              "de": "Liste",
              "pl": "Lista",
              "tr": "Liste",
              "ar": "القائمة",
              "ru": "Список",
              "it": "Elenco",
              "ro": "Listă",
              "es": "Lista",
              "sq": "Lista"
            }
          }
        }
    """
    data = _st_read(reader, errors, "ads_information/ad_preferences.json")
    if data is None:
        return pd.DataFrame()
    rows = []
    try:
        for record in _st_records(data):
            if not isinstance(record, dict):
                continue
            for label, entry in _st_top(record):
                listname = _st_fix(label)
                for item in entry.get("vec") or []:
                    if isinstance(item, dict) and item.get("value") not in (None, ""):
                        rows.append((_st_fix(item["value"]), listname))
                for item in entry.get("dict") or []:
                    fields = [f for f in (item or {}).get("dict") or [] if isinstance(f, dict)]
                    values = [f.get("value") for f in fields if f.get("value") not in (None, "")]
                    has_time = any(_st_is_epoch(f.get("timestamp_value")) for f in fields)
                    # Topic lists hold exactly one name per entry; hidden ads etc. have more fields.
                    if len(fields) == 1 and len(values) == 1 and not has_time:
                        rows.append((_st_fix(values[0]), listname))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return _st_df(rows, ["Topic", "List"])


def _st_find_key(node, key):
    """First value stored under *key* anywhere in *node*."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            found = _st_find_key(child, key)
            if found is not None:
                return found
    return None


def facebook_login_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
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
          "source_file": "security_and_login_information/logins_and_logouts.json, profile_information.json (registration time only)",
          "columns": {
            "Event": "Type of activity, translated to German where known (e.g. PAGE_VIEW -> Seitenaufruf).",
            "Date": "ISO 8601 timestamp."
          }
        }

    Table config::

        {
          "id": "facebook_login",
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
    data = _st_read(reader, errors, "security_and_login_information/logins_and_logouts.json")
    try:
        for record in _st_records(data) if data is not None else []:
            if isinstance(record, dict):
                # Only the kind of event and its time; site, IP address etc. are not read.
                rows.append((_st_fix(record.get("action") or ""), _st_iso(record.get("timestamp"), errors)))
        profile = _st_read(reader, errors, "profile_information/profile_information.json")
        found = _st_find_key(profile, "registration_timestamp")
        if _st_is_epoch(found):
            rows.append(("Registrierung", _st_iso(found, errors)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    rows.sort(key=lambda r: r[1], reverse=True)
    return _st_df(rows, ["Event", "Date"])


# ---------------------------------------------------------------------------
# Extractor registry & platform info
# ---------------------------------------------------------------------------

#: Mapping from the string names used in port_config.json to actual extractor functions.
#: Only the data described in the participant information; the order is the table order.
EXTRACTOR_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "facebook_following_to_df": facebook_following_to_df,
    "facebook_feed_to_df": facebook_feed_to_df,
    "facebook_ads_viewed_to_df": facebook_ads_viewed_to_df,
    "story_views_in_past_7_days_to_df": story_views_in_past_7_days_to_df,
    "facebook_likes_to_df": facebook_likes_to_df,
    "facebook_story_interactions_to_df": facebook_story_interactions_to_df,
    "facebook_saved_to_df": facebook_saved_to_df,
    "facebook_comments_to_df": facebook_comments_to_df,
    "facebook_searches_to_df": facebook_searches_to_df,
    "facebook_friends_to_df": facebook_friends_to_df,
    "facebook_interests_to_df": facebook_interests_to_df,
    "facebook_link_history_to_df": facebook_link_history_to_df,
    "facebook_off_meta_to_df": facebook_off_meta_to_df,
    "facebook_ad_topics_to_df": facebook_ad_topics_to_df,
    "facebook_advertisers_using_to_df": facebook_advertisers_using_to_df,
    "facebook_advertisers_interacted_to_df": facebook_advertisers_interacted_to_df,
    "facebook_other_categories_to_df": facebook_other_categories_to_df,
    "facebook_login_to_df": facebook_login_to_df,
}
# Empty tables (no rows, or only empty cells) are never shown to participants.
EXTRACTOR_REGISTRY = {name: _st_without_empty_rows(fn) for name, fn in EXTRACTOR_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Main extraction & flow
# ---------------------------------------------------------------------------

def extraction(facebook_zip: SeekableBinaryReader, validation) -> ExtractionResult:
    """Extract data from a Facebook DDP zip and return consent-form tables.

    Parameters
    ----------
    facebook_zip:
        Seekable binary reader over the Facebook DDP zip — the upload
        adapter itself, never a path (ADR-0026).
    validation:
        Validation result object whose ``archive_members`` attribute is passed
        to ``ZipArchiveReader``.
    """
    config = load_port_config(EXTRACTOR_REGISTRY, "facebook")
    errors: Counter = Counter()
    reader = ZipArchiveReader(facebook_zip, validation.archive_members, errors)
    return run_extraction(reader, errors, config)


class FacebookFlow(FlowBuilder):
    """Flow implementation for the Facebook data donation study."""

    def __init__(self, session_id: str):
        super().__init__(session_id, "Facebook")

    def validate_file(self, file):
        return validate.validate_zip(DDP_CATEGORIES, file)

    def extract_data(self, file_value, validation):
        return extraction(file_value, validation)


def process(session_id):
    flow = FacebookFlow(session_id)
    return flow.start_flow()