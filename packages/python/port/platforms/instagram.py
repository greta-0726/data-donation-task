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
# Helper functions
# ---------------------------------------------------------------------------

def _sort_by_date(out: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Sort *out* by *date_column* using ISO-timestamp ordering.

    Parameters
    ----------
    out:
        DataFrame to sort.
    date_column:
        Name of the column that contains ISO-formatted timestamp strings.
        Rows with empty timestamps are placed last.
    """
    return out.sort_values(by=date_column, key=eh.sort_isotimestamp_empty_timestamp_last)


# ---------------------------------------------------------------------------
# Language-aware label candidates
# ---------------------------------------------------------------------------
# Field labels are localized to the account's UI language, not fixed
# English keys -- match against known language variants.
_URL_LABELS = ["URL", "الرابط", "URL-адрес"]
_NAME_LABELS = ["Naam", "Name", "Nazwa", "Ad", "الاسم", "Имя", "Nome", "Nume", "Nombre", "Emri"]
_USERNAME_LABELS = ["Gebruikersnaam", "Username", "Author", "Nutzername", "Benutzername","Nazwa użytkownika", "Kullanıcı adı", "اسم المستخدم", "Имя пользователя", "Nome utente", "Nume de utilizator", "Nombre de usuario", "Emri i përdoruesit"]
_AUTHOR_LABELS = ["Author", "Auteur", "Autor", "Yazar", "الكاتب", "Автор", "Autore"]
_TIME_LABELS = ["Time", "Tijd", "Zeit", "Godzina", "Saat", "الوقت", "Время", "Ora", "Hora"]
_COMMENT_LABELS = ["Comment", "Opmerking", "Kommentar", "Komentarz", "Yorum", "تعليق", "Комментарий", "Commento", "Comentariu", "Comentario", "Koment"]
_MEDIA_OWNER_LABELS = ["Media Owner", "Media-eigenaar", "Medieninhaber", "Właściciel mediów", "Medya sahibi", "مالك الوسائط", "Владелец медиафайла", "Proprietario del contenuto multimediale", "Proprietarul conținutului media", "Propietario del contenido multimedia", "Pronari i medias"]
_SAVED_ON_LABELS = ["Saved on", "Opgeslagen op", "Gespeichert am", "Zapisano", "Kaydedilme tarihi", "تم الحفظ في", "Сохранено", "Salvato il", "Salvat la", "Guardado el", "Ruajtur më"]


def _first_present(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Return the first dict value found for the given keys, or empty dict.

    Parameters
    ----------
    data:
        Dictionary to search.
    keys:
        Ordered list of keys to try; the value of the first key whose
        corresponding value is a ``dict`` is returned.
    """
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _extract_owner_details(label_values: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Extract ``(owner_name, owner_username, url)`` from a nested label_values structure.

    This structure is used in newer Instagram export formats.

    Parameters
    ----------
    label_values:
        Nested list/dict structure from the Instagram DDP containing labelled
        metadata fields such as ``"Name"``, ``"Username"``, and ``"URL"``.

    Returns
    -------
    tuple[str, str, str]
        A three-tuple of ``(owner_name, owner_username, url)``.  Any field
        not found in *label_values* is returned as an empty string.
    """
    owner_name = ""
    owner_username = ""
    url = ""

    def visit(node: Any) -> None:
        nonlocal owner_name, owner_username, url

        if isinstance(node, list):
            for item in node:
                visit(item)
            return

        if not isinstance(node, dict):
            return

        label = str(node.get("label", ""))
        value = str(node.get("value", ""))
        href = str(node.get("href", ""))

        if label in _URL_LABELS and not url:
            url = href or value
        elif label in _NAME_LABELS and not owner_name:
            owner_name = eh.fix_latin1_string(value)
        elif label in _USERNAME_LABELS and not owner_username:
            owner_username = eh.fix_latin1_string(value)

        for child in node.values():
            visit(child)

    visit(label_values)
    return owner_name, owner_username, url


# ---------------------------------------------------------------------------
# Per-table extraction functions
# ---------------------------------------------------------------------------

def followers_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "followers_1.json",
) -> pd.DataFrame:
    """Extract the list of followers into a DataFrame.

    Handles both the newer bare top-level list format and the older format
    where entries are wrapped under a ``"relationships_followers"`` key.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.
    filename:
        Path inside the zip archive to read.  Defaults to
        ``"followers_1.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one account that follows the participant on Instagram, including when they started following.",
          "source_file": "followers_1.json",
          "columns": {
            "Account": "Username or display name of the follower account.",
            "URL": "Direct URL to the follower's Instagram profile.",
            "Date": "ISO 8601 timestamp of when the account started following the participant."
          }
        }

    Table config::

                {
          "id": "instagram_followers",
          "title": {
            "en": "Your Instagram followers",
            "nl": "Je Instagram-volgers",
            "de": "Ihre Instagram-Follower",
            "pl": "Twoi obserwujący na Instagramie",
            "tr": "Instagram takipçilerin",
            "ar": "متابعوك على إنستغرام",
            "ru": "Ваши подписчики в Instagram",
            "it": "I tuoi follower su Instagram",
            "ro": "Urmăritorii tăi de pe Instagram",
            "es": "Tus seguidores de Instagram",
            "sq": "Ndjekësit e tu në Instagram"
          },
          "description": {
            "en": "List of accounts that follow you on Instagram.",
            "nl": "Lijst van accounts die jou op Instagram volgen.",
            "de": "Liste der Konten, die Ihnen auf Instagram folgen.",
            "pl": "Lista kont, które obserwują cię na Instagramie.",
            "tr": "Instagram'da seni takip eden hesapların listesi.",
            "ar": "قائمة الحسابات التي تتابعك على إنستغرام.",
            "ru": "Список аккаунтов, которые подписаны на вас в Instagram.",
            "it": "Elenco degli account che ti seguono su Instagram.",
            "ro": "Lista conturilor care te urmăresc pe Instagram.",
            "es": "Lista de cuentas que te siguen en Instagram.",
            "sq": "Lista e llogarive që të ndjekin në Instagram."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Konto",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogari"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Data",
              "es": "Fecha",
              "sq": "Data"
            }
          }
        }
    """
    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()
    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        if isinstance(data, dict):
            items = data.get("relationships_followers", [])
        else:
            items = data  # pyright: ignore

        for item in items:
            d = eh.dict_denester(item)
            datapoints.append((
                eh.fix_latin1_string(eh.find_item(d, "value") or eh.find_item(d, "title")),
                eh.find_item(d, "href"),
                eh.epoch_to_iso(eh.find_item(d, "timestamp"), errors=errors),
            ))
        out = pd.DataFrame(datapoints, columns=["Account", "URL", "Date"])  # pyright: ignore
        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def following_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "following.json",
) -> pd.DataFrame:
    """Extract the list of followed accounts into a DataFrame.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.
    filename:
        Path inside the zip archive to read.  Defaults to
        ``"following.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one account that the participant follows on Instagram, including when they started following.",
          "source_file": "following.json",
          "columns": {
            "Account": "Username or display name of the followed account.",
            "URL": "Direct URL to the followed account's Instagram profile.",
            "Date": "ISO 8601 timestamp of when the participant started following this account."
          }
        }

    Table config::

                {
          "id": "instagram_following",
          "title": {
            "en": "Accounts that you follow on Instagram",
            "nl": "Accounts die je volgt op Instagram",
            "de": "Konten, denen Sie auf Instagram folgen",
            "pl": "Konta, które obserwujesz na Instagramie",
            "tr": "Instagram'da takip ettiğin hesaplar",
            "ar": "الحسابات التي تتابعها على إنستغرام",
            "ru": "Аккаунты, на которые вы подписаны в Instagram",
            "it": "Account che segui su Instagram",
            "ro": "Conturile pe care le urmărești pe Instagram",
            "es": "Cuentas que sigues en Instagram",
            "sq": "Llogaritë që ndjek në Instagram"
          },
          "description": {
            "en": "In this table, you find the accounts that you follow on Instagram.",
            "nl": "In deze tabel zie je de accounts die je volgt op Instagram.",
            "de": "In dieser Tabelle finden Sie die Konten, denen Sie auf Instagram folgen.",
            "pl": "W tej tabeli znajdziesz konta, które obserwujesz na Instagramie.",
            "tr": "Bu tabloda Instagram'da takip ettiğin hesapları bulabilirsin.",
            "ar": "في هذا الجدول، تجد الحسابات التي تتابعها على إنستغرام.",
            "ru": "В этой таблице вы найдёте аккаунты, на которые вы подписаны в Instagram.",
            "it": "In questa tabella trovi gli account che segui su Instagram.",
            "ro": "În acest tabel găsești conturile pe care le urmărești pe Instagram.",
            "es": "En esta tabla encontrarás las cuentas que sigues en Instagram.",
            "sq": "Në këtë tabelë gjen llogaritë që ndjek në Instagram."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Konto",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogari"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Data",
              "es": "Fecha",
              "sq": "Data"
            }
          }
        }
    """
    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()
    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = data["relationships_following"]  # pyright: ignore
        for item in items:
            d = eh.dict_denester(item)
            datapoints.append((
                eh.fix_latin1_string(eh.find_item(d, "title") or eh.find_item(d, "value")),
                eh.find_item(d, "href"),
                eh.epoch_to_iso(eh.find_item(d, "timestamp"), errors=errors),
            ))
        out = pd.DataFrame(datapoints, columns=["Account", "URL", "Date"])  # pyright: ignore
        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def ads_viewed_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "ads_viewed.json",
) -> pd.DataFrame:
    """Extract the list of viewed ads into a DataFrame.

    Supports both the list-at-root format and the dict format keyed by
    ``"impressions_history_ads_seen"``.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.
    filename:
        Path inside the zip archive to read.  Defaults to
        ``"ads_viewed.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account name``, ``Name``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one advertisement impression shown to the participant on Instagram. Includes the advertiser identity and when the ad was displayed.",
          "source_file": "ads_viewed.json",
          "columns": {
            "Account name": "Username of the advertiser's Instagram account.",
            "Name": "Display name of the advertiser.",
            "URL": "URL associated with the advertisement.",
            "Date": "ISO 8601 timestamp of when the ad was shown to the participant."
          }
        }

    Table config::

                {
          "id": "instagram_ads_viewed",
          "title": {
            "en": "Ads viewed on Instagram",
            "nl": "Advertenties bekeken op Instagram",
            "de": "Auf Instagram angesehene Werbeanzeigen",
            "pl": "Wyświetlone reklamy na Instagramie",
            "tr": "Instagram'da görüntülenen reklamlar",
            "ar": "الإعلانات التي شاهدتها على إنستغرام",
            "ru": "Реклама, просмотренная в Instagram",
            "it": "Inserzioni visualizzate su Instagram",
            "ro": "Reclame vizualizate pe Instagram",
            "es": "Anuncios vistos en Instagram",
            "sq": "Reklamat e shikuara në Instagram"
          },
          "description": {
            "en": "List of ads that you viewed on Instagram.",
            "nl": "Lijst van advertenties die je op Instagram hebt bekeken.",
            "de": "Liste der Werbeanzeigen, die Sie sich auf Instagram angesehen haben.",
            "pl": "Lista reklam, które wyświetliłeś/aś na Instagramie.",
            "tr": "Instagram'da görüntülediğin reklamların listesi.",
            "ar": "قائمة الإعلانات التي شاهدتها على إنستغرام.",
            "ru": "Список рекламы, которую вы просмотрели в Instagram.",
            "it": "Elenco delle inserzioni che hai visualizzato su Instagram.",
            "ro": "Lista reclamelor pe care le-ai vizualizat pe Instagram.",
            "es": "Lista de anuncios que viste en Instagram.",
            "sq": "Lista e reklamave që ke shikuar në Instagram."
          },
          "headers": {
            "Account name": {
              "en": "Account name",
              "nl": "Accountnaam",
              "de": "Instagram-Nutzername",
              "pl": "Nazwa konta",
              "tr": "Hesap adı",
              "ar": "اسم الحساب",
              "ru": "Имя аккаунта",
              "it": "Nome account",
              "ro": "Numele contului",
              "es": "Nombre de la cuenta",
              "sq": "Emri i llogarisë"
            },
            "Name": {
              "en": "Name",
              "nl": "Naam",
              "de": "Name des Accounts",
              "pl": "Nazwa",
              "tr": "Ad",
              "ar": "الاسم",
              "ru": "Имя",
              "it": "Nome",
              "ro": "Nume",
              "es": "Nombre",
              "sq": "Emri"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "Link zur Anzeige",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum en tijd",
              "de": "Zeitpunkt",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Data",
              "es": "Fecha",
              "sq": "Data"
            }
          }
        }
    """
    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()
    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("impressions_history_ads_seen", [])  # pyright: ignore
        else:
            items = []

        for item in items:  # pyright: ignore
            owner_name, owner_username, url = _extract_owner_details(item.get("label_values", []))
            datapoints.append((
                owner_username or owner_name,
                owner_name,
                url,
                eh.epoch_to_iso(item.get("timestamp", ""), errors=errors),
            ))

        out = pd.DataFrame(datapoints, columns=["Account name", "Name", "URL", "Date"])  # pyright: ignore
        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out

def other_categories_used_to_reach_you_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "other_categories_used_to_reach_you.json",
) -> pd.DataFrame:
    """Extract advertising targeting categories associated with the participant.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"other_categories_used_to_reach_you.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Category``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one category that may be used to reach the participant with advertising on Instagram.",
          "source_file": "other_categories_used_to_reach_you.json",
          "columns": {
            "Category": "A category associated with the participant that may be used for advertising targeting."
          }
        }

    Table config::

        {
          "id": "instagram_other_categories_used_to_reach_you",
          "title": {
            "en": "Categories used to reach you with ads",
            "nl": "Categorieën die worden gebruikt om je met advertenties te bereiken",
            "de": "Kategorien, die verwendet werden, um Sie mit Werbung zu erreichen",
            "pl": "Kategorie używane do docierania do Ciebie z reklamami",
            "tr": "Sana reklamlarla ulaşmak için kullanılan kategoriler",
            "ar": "الفئات المستخدمة للوصول إليك بالإعلانات",
            "ru": "Категории, используемые для показа вам рекламы",
            "it": "Categorie utilizzate per raggiungerti con le inserzioni",
            "ro": "Categorii utilizate pentru a ajunge la tine prin reclame",
            "es": "Categorías utilizadas para mostrarte anuncios",
            "sq": "Kategoritë e përdorura për të të arritur me reklama"
          },
          "description": {
            "en": "This table shows categories that Meta may use to determine which ads could be shown to you.",
            "nl": "Deze tabel toont categorieën die Meta kan gebruiken om te bepalen welke advertenties aan je kunnen worden getoond.",
            "de": "Diese Tabelle zeigt Kategorien, die Meta verwendet, um zu bestimmen, welche Werbung Ihnen angezeigt werden könnte.",
            "pl": "Ta tabela pokazuje kategorie, których Meta może używać do określania, jakie reklamy mogą być Ci wyświetlane.",
            "tr": "Bu tablo, Meta'nın sana hangi reklamların gösterilebileceğini belirlemek için kullanabileceği kategorileri gösterir.",
            "ar": "يعرض هذا الجدول الفئات التي قد تستخدمها Meta لتحديد الإعلانات التي يمكن عرضها لك.",
            "ru": "В этой таблице показаны категории, которые Meta может использовать для определения рекламы, которая может быть вам показана.",
            "it": "Questa tabella mostra le categorie che Meta può utilizzare per determinare quali inserzioni potrebbero esserti mostrate.",
            "ro": "Acest tabel arată categoriile pe care Meta le poate utiliza pentru a determina ce reclame ți-ar putea fi afișate.",
            "es": "Esta tabla muestra las categorías que Meta puede utilizar para determinar qué anuncios podrían mostrarse.",
            "sq": "Kjo tabelë tregon kategoritë që Meta mund të përdorë për të përcaktuar se cilat reklama mund të të shfaqen."
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

    result = reader.json(filename)

    if not result.found:
        return pd.DataFrame()

    data = result.data

    datapoints = []

    try:
        label_values = cast(dict, data).get("label_values", [])

        for item in label_values:
            if item.get("label") != "Name":
                continue

            for entry in item.get("vec", []):
                value = entry.get("value", "")

                if value:
                    datapoints.append((
                        eh.fix_latin1_string(value),
                    ))

        out = pd.DataFrame(
            datapoints,
            columns=["Category"],
        )

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return pd.DataFrame()

    return out

def posts_viewed_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "posts_viewed.json",
) -> pd.DataFrame:
    """Extract the list of viewed posts into a DataFrame.

    Handles both the older ``string_map_data`` format (dict root keyed by
    ``"impressions_history_posts_seen"``) and the newer ``label_values``
    list-at-root format.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.
    filename:
        Path inside the zip archive to read.  Defaults to
        ``"posts_viewed.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Author``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one post that appeared in the participant's Instagram feed and was registered as viewed. Captures the author and timing of each impression.",
          "source_file": "posts_viewed.json",
          "columns": {
            "Author": "Username or display name of the account that published the viewed post.",
            "URL": "Direct URL to the viewed post.",
            "Date": "ISO 8601 timestamp of when the post was viewed."
          }
        }

    Table config::

                {
          "id": "instagram_posts_viewed",
          "title": {
            "en": "Posts viewed on Instagram",
            "nl": "Berichten bekeken op Instagram",
            "de": "Auf Instagram angesehene Beiträge",
            "pl": "Wyświetlone posty na Instagramie",
            "tr": "Instagram'da görüntülenen gönderiler",
            "ar": "المنشورات التي شاهدتها على إنستغرام",
            "ru": "Публикации, просмотренные в Instagram",
            "it": "Post visualizzati su Instagram",
            "ro": "Postări vizualizate pe Instagram",
            "es": "Publicaciones vistas en Instagram",
            "sq": "Postimet e shikuara në Instagram"
          },
          "description": {
            "en": "In this table you find the accounts of posts you viewed on Instagram sorted over time. Below, you find visualizations of different parts of this table. First, you find a timeline showing you the number of posts you viewed over time. Next, you find a histogram indicating how many posts you have viewed per hour of the day.",
            "nl": "In deze tabel zie je de accounts van berichten die je op Instagram hebt bekeken, gesorteerd op tijd. Hieronder vind je visualisaties van verschillende onderdelen van deze tabel. Eerst zie je een tijdlijn met het aantal berichten dat je in de loop van de tijd hebt bekeken. Daarna zie je een histogram dat aangeeft hoeveel berichten je per uur van de dag hebt bekeken.",
            "de": "In dieser Tabelle finden Sie die Konten der Beiträge, die Sie auf Instagram angesehen haben, sortiert nach Zeit. Unten finden Sie Visualisierungen verschiedener Teile dieser Tabelle. Zuerst sehen Sie eine Zeitachse mit der Anzahl der Beiträge, die Sie im Laufe der Zeit angesehen haben. Danach sehen Sie ein Histogramm, das zeigt, wie viele Beiträge Sie pro Stunde des Tages angesehen haben.",
            "pl": "W tej tabeli znajdziesz konta postów, które wyświetliłeś/aś na Instagramie, posortowane chronologicznie. Poniżej znajdziesz wizualizacje różnych części tej tabeli. Najpierw zobaczysz oś czasu pokazującą liczbę postów, które wyświetlałeś/aś w czasie. Następnie zobaczysz histogram pokazujący, ile postów wyświetlałeś/aś w poszczególnych godzinach doby.",
            "tr": "Bu tabloda, zamana göre sıralanmış olarak Instagram'da görüntülediğin gönderilerin hesaplarını bulabilirsin. Aşağıda, bu tablonun farklı bölümlerine ait görselleştirmeler bulunur. İlk olarak, zaman içinde görüntülediğin gönderi sayısını gösteren bir zaman çizelgesi görürsün. Ardından, günün saatine göre kaç gönderi görüntülediğini gösteren bir histogram görürsün.",
            "ar": "في هذا الجدول، تجد حسابات المنشورات التي شاهدتها على إنستغرام مرتبة حسب الوقت. أدناه، تجد تصورات لأجزاء مختلفة من هذا الجدول. أولاً، تجد خطاً زمنياً يوضح عدد المنشورات التي شاهدتها عبر الوقت. بعد ذلك، تجد رسماً بيانياً يوضح عدد المنشورات التي شاهدتها في كل ساعة من اليوم.",
            "ru": "В этой таблице вы найдёте аккаунты публикаций, которые вы просматривали в Instagram, отсортированные по времени. Ниже вы найдёте визуализации различных частей этой таблицы. Сначала вы увидите временную шкалу с количеством просмотренных публикаций с течением времени. Затем вы увидите гистограмму, показывающую, сколько публикаций вы просматривали по часам суток.",
            "it": "In questa tabella trovi gli account dei post che hai visualizzato su Instagram, ordinati nel tempo. Di seguito trovi le visualizzazioni di diverse parti di questa tabella. Prima trovi una sequenza temporale che mostra il numero di post che hai visualizzato nel tempo. Poi trovi un istogramma che indica quanti post hai visualizzato per ogni ora del giorno.",
            "ro": "În acest tabel găsești conturile postărilor pe care le-ai vizualizat pe Instagram, sortate în timp. Mai jos găsești vizualizări ale diferitelor părți ale acestui tabel. Mai întâi găsești o cronologie care arată numărul de postări vizualizate în timp. Apoi găsești o histogramă care indică câte postări ai vizualizat pe oră din zi.",
            "es": "En esta tabla encontrarás las cuentas de las publicaciones que viste en Instagram, ordenadas cronológicamente. A continuación, encontrarás visualizaciones de diferentes partes de esta tabla. Primero, verás una línea de tiempo que muestra el número de publicaciones que viste a lo largo del tiempo. Luego, verás un histograma que indica cuántas publicaciones viste por hora del día.",
            "sq": "Në këtë tabelë gjen llogaritë e postimeve që ke shikuar në Instagram, të renditura sipas kohës. Më poshtë gjen vizualizime të pjesëve të ndryshme të kësaj tabele. Së pari, gjen një vijë kohore që tregon numrin e postimeve që ke parë me kalimin e kohës. Më pas, gjen një histogram që tregon sa postime ke parë për çdo orë të ditës."
          },
          "headers": {
            "Author": {
              "en": "Author",
              "nl": "Auteur",
              "de": "Autor*in",
              "pl": "Autor",
              "tr": "Yazar",
              "ar": "الكاتب",
              "ru": "Автор",
              "it": "Autore",
              "ro": "Autor",
              "es": "Autor",
              "sq": "Autor"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Data",
              "es": "Fecha",
              "sq": "Data"
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
              "type": "area",
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
            }
          ]
        }
    """
    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()
    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        if isinstance(data, dict):
            items = data["impressions_history_posts_seen"]  # pyright: ignore
            for item in items:
                string_map_data = item.get("string_map_data", {})
                author = _first_present(string_map_data, _AUTHOR_LABELS)
                time = _first_present(string_map_data, _TIME_LABELS)
                url = _first_present(string_map_data, _URL_LABELS)
                datapoints.append((
                    eh.fix_latin1_string(str(author.get("value", ""))),
                    url.get("href", ""),
                    eh.epoch_to_iso(time.get("timestamp", ""), errors=errors),
                ))
        else:
            for item in data:  # pyright: ignore
                owner_name, owner_username, url = _extract_owner_details(item.get("label_values", []))
                datapoints.append((
                    owner_username or owner_name,
                    url,
                    eh.epoch_to_iso(item.get("timestamp", ""), errors=errors),
                ))

        out = pd.DataFrame(datapoints, columns=["Author", "URL", "Date"])  # pyright: ignore
        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def videos_watched_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "videos_watched.json",
) -> pd.DataFrame:
    """Extract the list of watched videos into a DataFrame.

    Handles both the older ``string_map_data`` format (dict root keyed by
    ``"impressions_history_videos_watched"``) and the newer ``label_values``
    list-at-root format.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.
    filename:
        Path inside the zip archive to read.  Defaults to
        ``"videos_watched.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Author``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one video (including Reels) that the participant watched on Instagram. Captures the creator and timing of each view event.",
          "source_file": "videos_watched.json",
          "columns": {
            "Author": "Username or display name of the account that published the watched video.",
            "URL": "Direct URL to the watched video.",
            "Date": "ISO 8601 timestamp of when the video was watched."
          }
        }

    Table config::

                {
          "id": "instagram_videos_watched",
          "title": {
            "en": "Videos watched on Instagram",
            "nl": "Video's bekeken op Instagram",
            "de": "Auf Instagram angesehene Videos",
            "pl": "Obejrzane filmy na Instagramie",
            "tr": "Instagram'da izlenen videolar",
            "ar": "مقاطع الفيديو التي شاهدتها على إنستغرام",
            "ru": "Видео, просмотренные в Instagram",
            "it": "Video visualizzati su Instagram",
            "ro": "Videoclipuri vizionate pe Instagram",
            "es": "Vídeos vistos en Instagram",
            "sq": "Videot e shikuara në Instagram"
          },
          "description": {
            "en": "In this table you find the accounts of videos you watched on Instagram sorted over time. Below, you find a timeline showing you the number of videos you watched over time.",
            "nl": "In deze tabel zie je de accounts van video's die je op Instagram hebt bekeken, gesorteerd op tijd. Hieronder zie je een tijdlijn met het aantal video's dat je in de loop van de tijd hebt bekeken.",
            "de": "In dieser Tabelle finden Sie die Konten der Videos, die Sie auf Instagram angesehen haben, sortiert nach Zeit. Unten sehen Sie eine Zeitachse mit der Anzahl der Videos, die Sie angesehen haben.",
            "pl": "W tej tabeli znajdziesz konta filmów, które obejrzałeś/aś na Instagramie, posortowane chronologicznie. Poniżej zobaczysz oś czasu pokazującą liczbę filmów, które oglądałeś/aś w czasie.",
            "tr": "Bu tabloda, zamana göre sıralanmış olarak Instagram'da izlediğin videoların hesaplarını bulabilirsin. Aşağıda, zaman içinde izlediğin video sayısını gösteren bir zaman çizelgesi görürsün.",
            "ar": "في هذا الجدول، تجد حسابات مقاطع الفيديو التي شاهدتها على إنستغرام مرتبة حسب الوقت. أدناه، ترى خطاً زمنياً يوضح عدد مقاطع الفيديو التي شاهدتها عبر الوقت.",
            "ru": "В этой таблице вы найдёте аккаунты видео, которые вы смотрели в Instagram, отсортированные по времени. Ниже вы увидите временную шкалу с количеством просмотренных видео с течением времени.",
            "it": "In questa tabella trovi gli account dei video che hai guardato su Instagram, ordinati nel tempo. Di seguito vedi una sequenza temporale con il numero di video che hai guardato nel tempo.",
            "ro": "În acest tabel găsești conturile videoclipurilor pe care le-ai vizionat pe Instagram, sortate în timp. Mai jos vezi o cronologie cu numărul de videoclipuri vizionate în timp.",
            "es": "En esta tabla encontrarás las cuentas de los vídeos que viste en Instagram, ordenados cronológicamente. A continuación, verás una línea de tiempo con el número de vídeos que viste a lo largo del tiempo.",
            "sq": "Në këtë tabelë gjen llogaritë e videove që ke parë në Instagram, të renditura sipas kohës. Më poshtë sheh një vijë kohore me numrin e videove që ke parë me kalimin e kohës."
          },
          "headers": {
            "Author": {
              "en": "Author",
              "nl": "Auteur",
              "de": "Autor*in",
              "pl": "Autor",
              "tr": "Yazar",
              "ar": "الكاتب",
              "ru": "Автор",
              "it": "Autore",
              "ro": "Autor",
              "es": "Autor",
              "sq": "Autor"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Data",
              "es": "Fecha",
              "sq": "Data"
            }
          },
          "visualizations": [
            {
              "title": {
                "en": "The total number of videos watched on Instagram over time",
                "nl": "Het totale aantal video's dat je op Instagram hebt bekeken in de loop van de tijd",
                "de": "Die Gesamtzahl der auf Instagram angesehenen Videos im Laufe der Zeit",
                "pl": "Łączna liczba filmów obejrzanych na Instagramie w czasie",
                "tr": "Zaman içinde Instagram'da izlenen videoların toplam sayısı",
                "ar": "إجمالي عدد مقاطع الفيديو التي شوهدت على إنستغرام عبر الوقت",
                "ru": "Общее количество видео, просмотренных в Instagram с течением времени",
                "it": "Il numero totale di video visualizzati su Instagram nel tempo",
                "ro": "Numărul total de videoclipuri vizionate pe Instagram în timp",
                "es": "El número total de vídeos vistos en Instagram a lo largo del tiempo",
                "sq": "Numri total i videove të shikuara në Instagram me kalimin e kohës"
              },
              "type": "area",
              "group": {
                "column": "Date",
                "dateFormat": "auto",
                "label": "Datum"
              },
              "values": [
                {
                  "aggregate": "count",
                  "label": "Anzahl"
                }
              ]
            }
          ]
        }
    """
    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()
    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        if isinstance(data, dict):
            items = data["impressions_history_videos_watched"]  # pyright: ignore
            for item in items:
                string_map_data = item.get("string_map_data", {})
                author = _first_present(string_map_data, _AUTHOR_LABELS)
                time = _first_present(string_map_data, _TIME_LABELS)
                url = _first_present(string_map_data, _URL_LABELS)
                datapoints.append((
                    eh.fix_latin1_string(str(author.get("value", ""))),
                    url.get("href", ""),
                    eh.epoch_to_iso(time.get("timestamp", ""), errors=errors),
                ))
        else:
            for item in data:  # pyright: ignore
                owner_name, owner_username, url = _extract_owner_details(item.get("label_values", []))
                datapoints.append((
                    owner_username or owner_name,
                    url,
                    eh.epoch_to_iso(item.get("timestamp", ""), errors=errors),
                ))

        out = pd.DataFrame(datapoints, columns=["Author", "URL", "Date"])  # pyright: ignore
        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def post_comments_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename_pattern: str = r"(^|/)post_comments(?:_\d+)?\.json$",
) -> pd.DataFrame:
    """Extract all post comments across multiple matching files into a DataFrame.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.
    filename_pattern:
        Regular expression matched against archive member paths.  All matching
        files are read and combined.  Defaults to a pattern that matches
        ``post_comments.json``, ``post_comments_1.json``, etc.

    Returns
    -------
    pd.DataFrame
        Columns: ``Comment``, ``Media owner``, ``Date``.
        Empty DataFrame when no matching files are found or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one comment the participant posted on an Instagram post. Covers all matching comment files in the archive (e.g. post_comments.json, post_comments_1.json).",
          "source_file": "post_comments*.json",
          "columns": {
            "Comment": "The full text of the comment posted by the participant.",
            "Media owner": "Username of the account that owns the post the comment was placed on.",
            "Date": "ISO 8601 timestamp of when the comment was posted."
          }
        }

    Table config::

                {
          "id": "instagram_post_comments",
          "title": {
            "en": "Comments posted on Instagram",
            "nl": "Reacties geplaatst op Instagram",
            "de": "Auf Instagram veröffentlichte Kommentare",
            "pl": "Komentarze opublikowane na Instagramie",
            "tr": "Instagram'da paylaşılan yorumlar",
            "ar": "التعليقات التي نشرتها على إنستغرام",
            "ru": "Комментарии, опубликованные в Instagram",
            "it": "Commenti pubblicati su Instagram",
            "ro": "Comentarii publicate pe Instagram",
            "es": "Comentarios publicados en Instagram",
            "sq": "Komentet e publikuara në Instagram"
          },
          "description": {
            "en": "List of comments you posted on Instagram.",
            "nl": "Lijst van reacties die je op Instagram hebt geplaatst.",
            "de": "Liste der Kommentare, die Sie auf Instagram veröffentlicht haben.",
            "pl": "Lista komentarzy, które opublikowałeś/aś na Instagramie.",
            "tr": "Instagram'da paylaştığın yorumların listesi.",
            "ar": "قائمة التعليقات التي نشرتها على إنستغرام.",
            "ru": "Список комментариев, которые вы опубликовали в Instagram.",
            "it": "Elenco dei commenti che hai pubblicato su Instagram.",
            "ro": "Lista comentariilor pe care le-ai publicat pe Instagram.",
            "es": "Lista de comentarios que publicaste en Instagram.",
            "sq": "Lista e komenteve që ke publikuar në Instagram."
          },
          "headers": {
            "Comment": {
              "en": "Comment",
              "nl": "Reactie",
              "de": "Kommentar",
              "pl": "Komentarz",
              "tr": "Yorum",
              "ar": "تعليق",
              "ru": "Комментарий",
              "it": "Commento",
              "ro": "Comentariu",
              "es": "Comentario",
              "sq": "Koment"
            },
            "Media owner": {
              "en": "Media owner",
              "nl": "Media-eigenaar",
              "de": "Autor*in des kommentierten Beitrags",
              "pl": "Właściciel mediów",
              "tr": "Medya sahibi",
              "ar": "مالك الوسائط",
              "ru": "Владелец медиафайла",
              "it": "Proprietario del contenuto multimediale",
              "ro": "Proprietarul conținutului media",
              "es": "Propietario del contenido multimedia",
              "sq": "Pronari i medias"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Data",
              "es": "Fecha",
              "sq": "Data"
            }
          }
        }
    """
    out = pd.DataFrame()
    datapoints = []

    try:
        results = reader.json_all(filename_pattern)
        if not results:
            return pd.DataFrame()

        for result in results:
            data = result.data
            items = data if isinstance(data, list) else data.get("comments_media_comments", [])
            for item in items:  # pyright: ignore[assignment]
                string_map_data = item.get("string_map_data", {})
                comment = _first_present(string_map_data, _COMMENT_LABELS)
                owner = _first_present(string_map_data, _MEDIA_OWNER_LABELS)
                time = _first_present(string_map_data, _TIME_LABELS)
                datapoints.append((
                    eh.fix_latin1_string(str(comment.get("value", ""))),
                    eh.fix_latin1_string(str(owner.get("value", ""))),
                    eh.epoch_to_iso(time.get("timestamp", ""), errors=errors),
                ))

        out = pd.DataFrame(datapoints, columns=["Comment", "Media owner", "Date"])  # pyright: ignore
        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def liked_comments_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "liked_comments.json",
) -> pd.DataFrame:
    """Extract the list of liked comments into a DataFrame.

    Handles both the older ``string_list_data`` format (dict root keyed by
    ``"likes_comment_likes"``) and the newer ``label_values`` list-at-root
    format.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"liked_comments.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Author``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one comment the participant liked on Instagram, including the author of the comment, the associated URL, and when the comment was liked.",
          "source_file": "liked_comments.json",
          "columns": {
            "Author": "Username or display name of the account whose comment was liked.",
            "URL": "URL associated with the liked comment or the Instagram content on which it appeared.",
            "Date": "ISO 8601 timestamp of when the comment was liked."
          }
        }

    Table config::

        {
          "id": "instagram_liked_comments",
          "title": {
            "en": "Instagram liked comments",
            "nl": "Instagram-reacties die je leuk vond",
            "de": "Auf Instagram gelikte Kommentare",
            "pl": "Polubione komentarze na Instagramie",
            "tr": "Instagram'da beğenilen yorumlar",
            "ar": "التعليقات التي أعجبت بها على إنستغرام",
            "ru": "Комментарии, которые вам понравились в Instagram",
            "it": "Commenti apprezzati su Instagram",
            "ro": "Comentarii apreciate pe Instagram",
            "es": "Comentarios que te gustaron en Instagram",
            "sq": "Komentet që i ke pëlqyer në Instagram"
          },
          "description": {
            "en": "List of comments that you liked on Instagram.",
            "nl": "Lijst van reacties die je leuk vond op Instagram.",
            "de": "Liste der Kommentare, die Ihnen auf Instagram gefallen haben.",
            "pl": "Lista komentarzy, które polubiłeś/aś na Instagramie.",
            "tr": "Instagram'da beğendiğin yorumların listesi.",
            "ar": "قائمة التعليقات التي أعجبت بها على إنستغرام.",
            "ru": "Список комментариев, которые вам понравились в Instagram.",
            "it": "Elenco dei commenti che ti sono piaciuti su Instagram.",
            "ro": "Lista comentariilor care ți-au plăcut pe Instagram.",
            "es": "Lista de comentarios que te gustaron en Instagram.",
            "sq": "Lista e komenteve që i ke pëlqyer në Instagram."
          },
          "headers": {
            "Author": {
              "en": "Comment author",
              "nl": "Auteur van de reactie",
              "de": "Autor*in des Kommentars",
              "pl": "Autor komentarza",
              "tr": "Yorumun yazarı",
              "ar": "كاتب التعليق",
              "ru": "Автор комментария",
              "it": "Autore del commento",
              "ro": "Autorul comentariului",
              "es": "Autor del comentario",
              "sq": "Autori i komentit"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
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

    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()

    data = result.data
    out = pd.DataFrame()
    datapoints = []

    try:
        if isinstance(data, dict):
            items = data.get("likes_comment_likes", [])

            for item in items:
                string_list_data = item.get("string_list_data", [])

                if not string_list_data:
                    continue

                entry = string_list_data[0]

                datapoints.append((
                    eh.fix_latin1_string(item.get("title", "")),
                    entry.get("href", ""),
                    eh.epoch_to_iso(
                        entry.get("timestamp", ""),
                        errors=errors,
                    ),
                ))

        else:
            for item in data:  # pyright: ignore
                owner_name, owner_username, url = _extract_owner_details(
                    item.get("label_values", [])
                )

                datapoints.append((
                    owner_username or owner_name,
                    url,
                    eh.epoch_to_iso(
                        item.get("timestamp", ""),
                        errors=errors,
                    ),
                ))

        out = pd.DataFrame(
            datapoints,
            columns=["Author", "URL", "Date"],
        )

        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out
def liked_posts_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "liked_posts.json",
) -> pd.DataFrame:
    """Extract the list of liked posts into a DataFrame.

    Handles both the older ``dict_denester`` format (dict root keyed by
    ``"likes_media_likes"``) and the newer ``label_values`` list-at-root
    format.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"liked_posts.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one post the participant liked on Instagram, including the account whose post was liked, the URL of the post, and when the like was given.",
          "source_file": "liked_posts.json",
          "columns": {
            "Account": "Username or display name of the account whose post was liked.",
            "URL": "Direct URL to the liked Instagram post.",
            "Date": "ISO 8601 timestamp of when the post was liked."
          }
        }

    Table config::

        {
          "id": "instagram_liked_posts",
          "title": {
            "en": "Instagram liked posts",
            "nl": "Instagram-berichten die je leuk vond",
            "de": "Auf Instagram gelikte Beiträge",
            "pl": "Polubione posty na Instagramie",
            "tr": "Instagram'da beğenilen gönderiler",
            "ar": "المنشورات التي أعجبت بها على إنستغرام",
            "ru": "Публикации, которые вам понравились в Instagram",
            "it": "Post apprezzati su Instagram",
            "ro": "Postări apreciate pe Instagram",
            "es": "Publicaciones que te gustaron en Instagram",
            "sq": "Postimet që i ke pëlqyer në Instagram"
          },
          "description": {
            "en": "List of posts that you liked on Instagram.",
            "nl": "Lijst van berichten die je leuk vond op Instagram.",
            "de": "Liste der Beiträge, die Ihnen auf Instagram gefallen haben.",
            "pl": "Lista postów, które polubiłeś/aś na Instagramie.",
            "tr": "Instagram'da beğendiğin gönderilerin listesi.",
            "ar": "قائمة المنشورات التي أعجبت بها على إنستغرام.",
            "ru": "Список публикаций, которые вам понравились в Instagram.",
            "it": "Elenco dei post che ti sono piaciuti su Instagram.",
            "ro": "Lista postărilor care ți-au plăcut pe Instagram.",
            "es": "Lista de publicaciones que te gustaron en Instagram.",
            "sq": "Lista e postimeve që i ke pëlqyer në Instagram."
          },
          "headers": {
            "Account": {
              "en": "Account",
              "nl": "Account",
              "de": "Kontoname",
              "pl": "Konto",
              "tr": "Hesap",
              "ar": "الحساب",
              "ru": "Аккаунт",
              "it": "Account",
              "ro": "Cont",
              "es": "Cuenta",
              "sq": "Llogari"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
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
                "en": "Most liked accounts",
                "nl": "Meest gelikete accounts",
                "de": "Am häufigsten gelikte Konten",
                "pl": "Najczęściej polubione konta",
                "tr": "En çok beğenilen hesaplar",
                "ar": "الحسابات الأكثر إعجاباً",
                "ru": "Самые популярные аккаунты по лайкам",
                "it": "Account più apprezzati",
                "ro": "Cele mai apreciate conturi",
                "es": "Cuentas más gustadas",
                "sq": "Llogaritë më të pëlqyera"
              },
              "type": "wordcloud",
              "textColumn": "Account",
              "tokenize": false
            }
          ]
        }
    """

    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()

    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        if isinstance(data, dict):
            items = data.get("likes_media_likes", [])

            for item in items:
                d = eh.dict_denester(item)

                datapoints.append((
                    eh.fix_latin1_string(
                        eh.find_item(d, "title")
                        or eh.find_item(d, "value")
                        or ""
                    ),
                    eh.find_item(d, "href") or "",
                    eh.epoch_to_iso(
                        eh.find_item(d, "timestamp"),
                        errors=errors,
                    ),
                ))

        else:
            for item in data:  # pyright: ignore
                owner_name, owner_username, url = _extract_owner_details(
                    item.get("label_values", [])
                )

                datapoints.append((
                    owner_username or owner_name,
                    url,
                    eh.epoch_to_iso(
                        item.get("timestamp", ""),
                        errors=errors,
                    ),
                ))

        out = pd.DataFrame(
            datapoints,
            columns=["Account", "URL", "Date"],
        )

        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out

def searches_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    word_or_phrase_filename: str = "word_or_phrase_searches.json",
    recent_filename: str = "recent_searches.json",
) -> pd.DataFrame:
    """Extract Instagram word and phrase searches into one DataFrame.

    Instagram may record the same search in both
    ``word_or_phrase_searches.json`` and ``recent_searches.json`` a few
    seconds apart. Records from the word-or-phrase file take precedence;
    a recent-search record is omitted only when its normalized text matches
    and its timestamp differs by no more than three seconds.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    word_or_phrase_filename:
        Path inside the zip archive to the word-or-phrase search file.
    recent_filename:
        Path inside the zip archive to the recent-search file.

    Returns
    -------
    pd.DataFrame
        Columns: ``Search``, ``Date``.
        Empty DataFrame when neither source file is present or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a word or phrase searched for by the participant on Instagram. Duplicate records emitted by Instagram's two search-history files are combined when their text matches and their timestamps differ by no more than three seconds.",
          "source_files": [
            "word_or_phrase_searches.json",
            "recent_searches.json"
          ],
          "columns": {
            "Search": "Word or phrase searched for on Instagram.",
            "Date": "ISO 8601 timestamp of when the search was performed."
          }
        }

    Table config::

        {
          "id": "instagram_searches",
          "title": {
            "en": "Your Instagram searches",
            "nl": "Je Instagram-zoekopdrachten",
            "de": "Ihre Instagram-Suchen",
            "pl": "Twoje wyszukiwania na Instagramie",
            "tr": "Instagram aramaların",
            "ar": "عمليات بحثك على إنستغرام",
            "ru": "Ваши поисковые запросы в Instagram",
            "it": "Le tue ricerche su Instagram",
            "ro": "Căutările tale pe Instagram",
            "es": "Tus búsquedas en Instagram",
            "sq": "Kërkimet e tua në Instagram"
          },
          "description": {
            "en": "List of words or phrases you have searched for on Instagram.",
            "nl": "Lijst van woorden of woordgroepen waarnaar je op Instagram hebt gezocht.",
            "de": "Liste der Wörter oder Wortgruppen, nach denen Sie auf Instagram gesucht haben.",
            "pl": "Lista słów lub wyrażeń wyszukiwanych przez Ciebie na Instagramie.",
            "tr": "Instagram'da aradığın kelime veya ifadelerin listesi.",
            "ar": "قائمة بالكلمات أو العبارات التي بحثت عنها على إنستغرام.",
            "ru": "Список слов или фраз, которые вы искали в Instagram.",
            "it": "Elenco delle parole o frasi che hai cercato su Instagram.",
            "ro": "Lista cuvintelor sau expresiilor pe care le-ai căutat pe Instagram.",
            "es": "Lista de palabras o frases que buscaste en Instagram.",
            "sq": "Lista e fjalëve ose frazave që ke kërkuar në Instagram."
          },
          "headers": {
            "Search": {
              "en": "Search",
              "nl": "Zoekopdracht",
              "de": "Suche",
              "pl": "Wyszukiwanie",
              "tr": "Arama",
              "ar": "البحث",
              "ru": "Поисковый запрос",
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

    word_result = reader.json(word_or_phrase_filename)
    recent_result = reader.json(recent_filename)

    if not word_result.found and not recent_result.found:
        return pd.DataFrame()

    word_searches: list[tuple[str, Any]] = []
    recent_searches: list[tuple[str, Any]] = []

    try:
        if word_result.found and isinstance(word_result.data, dict):
            items = cast(dict, word_result.data).get("searches_keyword", [])
            for item in items:
                string_map_data = item.get("string_map_data", {})
                entries = (
                    string_map_data.values()
                    if isinstance(string_map_data, dict)
                    else []
                )
                entries = list(entries)
                search = next((
                    eh.fix_latin1_string(str(entry.get("value", ""))).strip()
                    for entry in entries
                    if str(entry.get("value", "")).strip()
                ), "")
                timestamp = next((
                    entry.get("timestamp")
                    for entry in entries
                    if entry.get("timestamp") not in (None, "", 0)
                ), "")
                if search:
                    word_searches.append((search, timestamp))

        if recent_result.found and isinstance(recent_result.data, list):
            for item in cast(list, recent_result.data):
                if not isinstance(item, dict):
                    continue
                label_values = item.get("label_values", [])
                search = next((
                    eh.fix_latin1_string(str(entry.get("value", ""))).strip()
                    for entry in label_values
                    if isinstance(entry, dict)
                    and str(entry.get("value", "")).strip()
                ), "")
                timestamp = next((
                    entry.get("timestamp_value")
                    for entry in label_values
                    if isinstance(entry, dict)
                    and entry.get("timestamp_value") not in (None, "", 0)
                ), item.get("timestamp", ""))
                if search:
                    recent_searches.append((search, timestamp))

        datapoints = list(word_searches)
        for search, timestamp in recent_searches:
            normalized_search = search.casefold().strip()
            is_duplicate = any(
                normalized_search == word_search.casefold().strip()
                and isinstance(timestamp, (int, float))
                and isinstance(word_timestamp, (int, float))
                and abs(timestamp - word_timestamp) <= 3
                for word_search, word_timestamp in word_searches
            )
            if not is_duplicate:
                datapoints.append((search, timestamp))

        out = pd.DataFrame(
            [
                (search, eh.epoch_to_iso(timestamp, errors=errors))
                for search, timestamp in datapoints
            ],
            columns=["Search", "Date"],
        )
        return _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return pd.DataFrame()


def profile_searches_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "profile_searches.json",
) -> pd.DataFrame:
    """Extract the list of Instagram profile searches into a DataFrame.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"profile_searches.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Name``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one profile search performed by the participant on Instagram, including the searched profile, its URL, and when the search was performed.",
          "source_file": "profile_searches.json",
          "columns": {
            "Name": "Username or display name of the Instagram profile that was searched for.",
            "URL": "URL associated with the searched Instagram profile.",
            "Date": "ISO 8601 timestamp of when the profile search was performed."
          }
        }

    Table config::

        {
          "id": "instagram_profile_searches",
          "title": {
            "en": "Your Instagram profile searches",
            "nl": "Je Instagram-profielzoekopdrachten",
            "de": "Ihre Instagram-Profilsuchen",
            "pl": "Twoje wyszukiwania profili na Instagramie",
            "tr": "Instagram profil aramaların",
            "ar": "عمليات بحثك عن الملفات الشخصية على إنستغرام",
            "ru": "Ваши поиски профилей в Instagram",
            "it": "Le tue ricerche di profili su Instagram",
            "ro": "Căutările tale de profiluri pe Instagram",
            "es": "Tus búsquedas de perfiles en Instagram",
            "sq": "Kërkimet e tua për profile në Instagram"
          },
          "description": {
            "en": "List of profiles you have searched for on Instagram.",
            "nl": "Lijst van profielen die je op Instagram hebt gezocht.",
            "de": "Liste der Profile, nach denen Sie auf Instagram gesucht haben.",
            "pl": "Lista profili, których szukałeś/aś na Instagramie.",
            "tr": "Instagram'da aradığın profillerin listesi.",
            "ar": "قائمة الملفات الشخصية التي بحثت عنها على إنستغرام.",
            "ru": "Список профилей, которые вы искали в Instagram.",
            "it": "Elenco dei profili che hai cercato su Instagram.",
            "ro": "Lista profilurilor pe care le-ai căutat pe Instagram.",
            "es": "Lista de perfiles que buscaste en Instagram.",
            "sq": "Lista e profileve që ke kërkuar në Instagram."
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
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
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

    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()

    data = result.data
    out = pd.DataFrame()
    datapoints = []

    try:
        items = cast(dict, data).get("searches_user", [])

        for item in items:
            d = eh.dict_denester(item)

            datapoints.append((
                eh.fix_latin1_string(
                    eh.find_item(d, "title")
                    or eh.find_item(d, "value")
                    or ""
                ),
                eh.find_item(d, "href") or "",
                eh.epoch_to_iso(
                    eh.find_item(d, "timestamp"),
                    errors=errors,
                ),
            ))

        out = pd.DataFrame(
            datapoints,
            columns=["Name", "URL", "Date"],
        )

        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def story_likes_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "story_likes.json",
) -> pd.DataFrame:
    """Extract the list of liked stories into a DataFrame.

    Handles both the older ``string_list_data`` format (dict root keyed by
    ``"story_activities_story_likes"``) and the newer ``label_values``
    list-at-root format.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.
    filename:
        Path inside the zip archive to read.  Defaults to
        ``"story_likes.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account name``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one Instagram Story the participant liked, recording the account whose story was liked and when.",
          "source_file": "story_likes.json",
          "columns": {
            "Account name": "Username of the account whose story was liked.",
            "Date": "ISO 8601 timestamp of when the story was liked."
          }
        }

    Table config::

                {
          "id": "instagram_story_likes",
          "title": {
            "en": "Story likes on Instagram",
            "nl": "Story-likes op Instagram",
            "de": "Story-Likes auf Instagram",
            "pl": "Polubienia relacji na Instagramie",
            "tr": "Instagram hikaye beğenilerin",
            "ar": "إعجاباتك بالقصص على إنستغرام",
            "ru": "Понравившиеся истории в Instagram",
            "it": "Storie apprezzate su Instagram",
            "ro": "Aprecieri la povești pe Instagram",
            "es": "Historias que te gustaron en Instagram",
            "sq": "Pëlqimet e stories në Instagram"
          },
          "description": {
            "en": "List of Instagram stories you liked.",
            "nl": "Lijst van Instagram-stories die je leuk vond.",
            "de": "Liste der Instagram-Storys, die Ihnen gefallen haben.",
            "pl": "Lista relacji na Instagramie, które polubiłeś/aś.",
            "tr": "Beğendiğin Instagram hikayelerinin listesi.",
            "ar": "قائمة قصص إنستغرام التي أعجبت بها.",
            "ru": "Список историй в Instagram, которые вам понравились.",
            "it": "Elenco delle storie di Instagram che ti sono piaciute.",
            "ro": "Lista poveștilor de pe Instagram care ți-au plăcut.",
            "es": "Lista de historias de Instagram que te gustaron.",
            "sq": "Lista e stories në Instagram që i ke pëlqyer."
          },
          "headers": {
            "Account name": {
              "en": "Account name",
              "nl": "Accountnaam",
              "de": "Autor*in der Story",
              "pl": "Nazwa konta",
              "tr": "Hesap adı",
              "ar": "اسم الحساب",
              "ru": "Имя аккаунта",
              "it": "Nome account",
              "ro": "Numele contului",
              "es": "Nombre de la cuenta",
              "sq": "Emri i llogarisë"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Data",
              "es": "Fecha",
              "sq": "Data"
            }
          }
        }
    """
    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()
    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        if isinstance(data, dict):
            items = data["story_activities_story_likes"]  # pyright: ignore
            for item in items:
                entry = item.get("string_list_data", [{}])[0]
                datapoints.append((
                    eh.fix_latin1_string(item.get("title", "")),
                    eh.epoch_to_iso(entry.get("timestamp", ""), errors=errors),
                ))
        else:
            for item in data:  # pyright: ignore
                owner_name, owner_username, _ = _extract_owner_details(item.get("label_values", []))
                datapoints.append((
                    owner_username or owner_name,
                    eh.epoch_to_iso(item.get("timestamp", ""), errors=errors),
                ))

        out = pd.DataFrame(datapoints, columns=["Account name", "Date"])  # pyright: ignore
        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out
def stories_viewed_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "stories_viewed.json",
) -> pd.DataFrame:
    """Extract the list of Instagram Stories viewed by the participant.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"stories_viewed.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Author``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one Instagram Story viewed by the participant, including the author and time of the view.",
          "source_file": "stories_viewed.json",
          "columns": {
            "Author": "Username or display name of the account that published the viewed Story.",
            "Date": "ISO 8601 timestamp of when the Story was viewed."
          }
        }

    Table config::

        {
          "id": "instagram_stories_viewed",
          "title": {
            "en": "Stories viewed on Instagram",
            "nl": "Stories bekeken op Instagram",
            "de": "Auf Instagram angesehene Storys",
            "pl": "Wyświetlone relacje na Instagramie",
            "tr": "Instagram'da görüntülenen hikayeler",
            "ar": "القصص التي شاهدتها على إنستغرام",
            "ru": "Просмотренные истории в Instagram",
            "it": "Storie visualizzate su Instagram",
            "ro": "Povești vizualizate pe Instagram",
            "es": "Historias vistas en Instagram",
            "sq": "Stories të shikuara në Instagram"
          },
          "description": {
            "en": "This table shows the Instagram Stories you viewed.",
            "nl": "Deze tabel toont de Instagram Stories die je hebt bekeken.",
            "de": "Diese Tabelle zeigt die Instagram-Storys, die Sie angesehen haben.",
            "pl": "Ta tabela pokazuje relacje na Instagramie, które wyświetliłeś/aś.",
            "tr": "Bu tablo Instagram'da görüntülediğin hikayeleri gösterir.",
            "ar": "يعرض هذا الجدول قصص إنستغرام التي شاهدتها.",
            "ru": "В этой таблице показаны истории Instagram, которые вы просмотрели.",
            "it": "Questa tabella mostra le Storie di Instagram che hai visualizzato.",
            "ro": "Acest tabel arată poveștile de pe Instagram pe care le-ai vizualizat.",
            "es": "Esta tabla muestra las historias de Instagram que viste.",
            "sq": "Kjo tabelë tregon Stories në Instagram që ke parë."
          },
          "headers": {
            "Author": {
              "en": "Author",
              "nl": "Auteur",
              "de": "Autor*in der Story",
              "pl": "Autor",
              "tr": "Yazar",
              "ar": "الكاتب",
              "ru": "Автор",
              "it": "Autore",
              "ro": "Autor",
              "es": "Autor",
              "sq": "Autor"
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

    result = reader.json(filename)

    if not result.found:
        return pd.DataFrame()

    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        for item in data:
            owner_name, owner_username, _ = _extract_owner_details(
                item.get("label_values", [])
            )

            author = owner_username or owner_name

            datapoints.append((
                author,
                eh.epoch_to_iso(
                    item.get("timestamp", ""),
                    errors=errors,
                ),
            ))

        out = pd.DataFrame(
            datapoints,
            columns=["Author", "Date"],
        )

        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out

def threads_viewed_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "threads_viewed.json",
) -> pd.DataFrame:
    """Extract the list of viewed Threads posts into a DataFrame.

    Handles both the older ``string_map_data`` format (dict root keyed by
    ``"text_post_app_text_post_app_posts_seen"``) and the newer
    ``label_values`` list-at-root format.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.
    filename:
        Path inside the zip archive to read.  Defaults to
        ``"threads_viewed.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Author``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one post on Threads (Meta's text-based social network linked to Instagram) that the participant viewed, including the author and timing.",
          "source_file": "threads_viewed.json",
          "columns": {
            "Author": "Username or display name of the account that published the viewed Threads post.",
            "URL": "Direct URL to the viewed Threads post.",
            "Date": "ISO 8601 timestamp of when the post was viewed."
          }
        }

    Table config::

                {
          "id": "instagram_threads_viewed",
          "title": {
            "en": "Threads viewed",
            "nl": "Threads bekeken",
            "de": "Angesehene Threads-Beiträge",
            "pl": "Wyświetlone posty na Threads",
            "tr": "Görüntülenen Threads gönderileri",
            "ar": "منشورات Threads التي شاهدتها",
            "ru": "Просмотренные публикации в Threads",
            "it": "Post di Threads visualizzati",
            "ro": "Postări Threads vizualizate",
            "es": "Publicaciones de Threads vistas",
            "sq": "Postimet e Threads të shikuara"
          },
          "description": {
            "en": "List of Threads posts you viewed.",
            "nl": "Lijst van Threads-berichten die je hebt bekeken.",
            "de": "Liste der Threads-Beiträge, die Sie sich angesehen haben.",
            "pl": "Lista postów z Threads, które wyświetliłeś/aś.",
            "tr": "Görüntülediğin Threads gönderilerinin listesi.",
            "ar": "قائمة منشورات Threads التي شاهدتها.",
            "ru": "Список публикаций Threads, которые вы просмотрели.",
            "it": "Elenco dei post di Threads che hai visualizzato.",
            "ro": "Lista postărilor Threads pe care le-ai vizualizat.",
            "es": "Lista de publicaciones de Threads que viste.",
            "sq": "Lista e postimeve të Threads që ke parë."
          },
          "headers": {
            "Author": {
              "en": "Author",
              "nl": "Auteur",
              "de": "Autor*in",
              "pl": "Autor",
              "tr": "Yazar",
              "ar": "الكاتب",
              "ru": "Автор",
              "it": "Autore",
              "ro": "Autor",
              "es": "Autor",
              "sq": "Autor"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Data",
              "es": "Fecha",
              "sq": "Data"
            }
          }
        }
    """
    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()
    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        if isinstance(data, dict):
            items = data["text_post_app_text_post_app_posts_seen"]  # pyright: ignore
            for item in items:
                string_map_data = item.get("string_map_data", {})
                author = _first_present(string_map_data, _AUTHOR_LABELS)
                time = _first_present(string_map_data, _TIME_LABELS)
                url = _first_present(string_map_data, _URL_LABELS)
                datapoints.append((
                    eh.fix_latin1_string(str(author.get("value", ""))),
                    url.get("href", ""),
                    eh.epoch_to_iso(time.get("timestamp", ""), errors=errors),
                ))
        else:
            for item in data:  # pyright: ignore
                owner_name, owner_username, url = _extract_owner_details(item.get("label_values", []))
                datapoints.append((
                    owner_username or owner_name,
                    url,
                    eh.epoch_to_iso(item.get("timestamp", ""), errors=errors),
                ))

        out = pd.DataFrame(datapoints, columns=["Author", "URL", "Date"])  # pyright: ignore
        out = _sort_by_date(out, "Date")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def saved_posts_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "saved_posts.json",
) -> pd.DataFrame:
    """Extract the list of saved posts into a DataFrame.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction.  Updated in-place.
    filename:
        Path inside the zip archive to read.  Defaults to
        ``"saved_posts.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Title``, ``URL``, ``Timestamp``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one post the participant bookmarked (saved) on Instagram for later viewing.",
          "source_file": "saved_posts.json",
          "columns": {
            "Title": "Title or label of the saved post as stored in the export.",
            "URL": "Direct URL to the saved post.",
            "Timestamp": "ISO 8601 timestamp of when the post was saved."
          }
        }

    Table config::

                {
          "id": "instagram_saved_posts",
          "title": {
            "en": "Your saved posts on Instagram",
            "nl": "Je opgeslagen berichten op Instagram",
            "de": "Ihre gespeicherten Beiträge auf Instagram",
            "pl": "Twoje zapisane posty na Instagramie",
            "tr": "Instagram'da kaydettiğin gönderiler",
            "ar": "منشوراتك المحفوظة على إنستغرام",
            "ru": "Ваши сохранённые публикации в Instagram",
            "it": "I tuoi post salvati su Instagram",
            "ro": "Postările tale salvate pe Instagram",
            "es": "Tus publicaciones guardadas en Instagram",
            "sq": "Postimet e tua të ruajtura në Instagram"
          },
          "description": {
            "en": "List of posts you have saved on Instagram.",
            "nl": "Lijst van berichten die je hebt opgeslagen op Instagram.",
            "de": "Liste der Beiträge, die Sie auf Instagram gespeichert haben.",
            "pl": "Lista postów, które zapisałeś/aś na Instagramie.",
            "tr": "Instagram'da kaydettiğin gönderilerin listesi.",
            "ar": "قائمة المنشورات التي حفظتها على إنستغرام.",
            "ru": "Список публикаций, которые вы сохранили в Instagram.",
            "it": "Elenco dei post che hai salvato su Instagram.",
            "ro": "Lista postărilor pe care le-ai salvat pe Instagram.",
            "es": "Lista de publicaciones que guardaste en Instagram.",
            "sq": "Lista e postimeve që ke ruajtur në Instagram."
          },
          "headers": {
            "Title": {
              "en": "Title",
              "nl": "Titel",
              "de": "Titel",
              "pl": "Tytuł",
              "tr": "Başlık",
              "ar": "العنوان",
              "ru": "Заголовок",
              "it": "Titolo",
              "ro": "Titlu",
              "es": "Título",
              "sq": "Titulli"
            },
            "Timestamp": {
              "en": "Timestamp",
              "nl": "Datum en tijd",
              "de": "Zeitstempel",
              "pl": "Znacznik czasu",
              "tr": "Zaman Damgası",
              "ar": "الطابع الزمني",
              "ru": "Отметка времени",
              "it": "Timestamp",
              "ro": "Marcaj temporal",
              "es": "Marca de tiempo",
              "sq": "Vula kohore"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط",
              "ru": "URL-адрес",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            }
          }
        }
    """
    result = reader.json(filename)
    if not result.found:
        return pd.DataFrame()
    data = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = data["saved_saved_media"]  # pyright: ignore
        for item in items:
            title = eh.fix_latin1_string(item.get("title", ""))
            if "string_list_data" in item:
                string_list = item.get("string_list_data", [{}])
                entry = string_list[0] if string_list else {}
            else:
                entry = _first_present(item.get("string_map_data", {}), _SAVED_ON_LABELS)
            datapoints.append((
                title,
                entry.get("href", ""),
                eh.epoch_to_iso(entry.get("timestamp", ""), errors=errors),
            ))
        out = pd.DataFrame(datapoints, columns=["Title", "URL", "Timestamp"])  # pyright: ignore
        out = _sort_by_date(out, "Timestamp")

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Category extractors (settings, security)
#
# These tables bundle many small Facebook files into a handful of tables so
# participants are not shown dozens of near-empty tables.  The files are
# heterogeneous: most use the localized ``label_values`` structure (labels are
# in the participant's Facebook language), some use ``string_map_data`` dicts.
# The helpers below therefore walk the structure generically instead of
# relying on language-specific labels, and drop anything that looks like an
# identifier (IP address, e-mail, phone number, cookie, user agent).
# ---------------------------------------------------------------------------

_LV_KEYS = {"label", "value", "timestamp_value", "vec", "dict", "title", "label_values", "href"}
_PATH_SEP = " › "

_RE_IPV4 = re.compile(r"\d{1,3}(\.\d{1,3}){3}")
_RE_IPV6 = re.compile(r"[0-9a-fA-F:.]+")
_RE_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_RE_PHONE = re.compile(r"\+\d[\d\s().-]{6,}")
_RE_HEX_TOKEN = re.compile(r"[0-9a-fA-F]{32,}")
_RE_LONG_ID = re.compile(r"\d{15,}")


def _is_sensitive_value(value: str) -> bool:
    """Return True if *value* looks like an identifier that must not be donated."""
    v = value.strip()
    if not v:
        return False
    if v.startswith("Mozilla/") or "****" in v:
        return True
    if _RE_EMAIL.fullmatch(v) or _RE_PHONE.fullmatch(v) or _RE_IPV4.fullmatch(v):
        return True
    if _RE_HEX_TOKEN.fullmatch(v) or _RE_LONG_ID.fullmatch(v):
        return True
    if v.count(":") >= 3 and _RE_IPV6.fullmatch(v):
        return True
    return False


#: Display labels for the few English dict keys used by older export files.
_PLAIN_KEY_LABELS = {
    "location_services_setting_v2": "Einstellung für Standortdienste",
}


#: Meta leaves some labels untranslated (English) in otherwise German exports.
#: Known ones are mapped to German here; unknown strings are kept as exported.
_META_ENGLISH_TO_GERMAN = {
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


def _read_json_data(reader: ZipArchiveReader, errors: Counter, path: str):
    """Return parsed JSON for *path*, or None if the file is absent or unreadable."""
    try:
        result = reader.json(path)
    except Exception as e:  # found-but-broken file (ADR-0024)
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return None
    if not result.found:
        return None
    return result.data


def _label_value_rows(
    data,
    category: str,
    errors: Counter,
    *,
    scalars_only: bool = False,
    nested_only: bool = False,
    keep_empty_labels: bool = False,
) -> list[tuple[str, str, str, str]]:
    """Flatten a ``label_values`` (or plain dict) structure into settings rows.

    Returns ``(category, setting, value, date)`` tuples.

    Parameters
    ----------
    scalars_only:
        Do not descend into nested ``dict`` / ``vec`` values (used for files
        whose nested values are lists of other people).
    nested_only:
        Only keep values found inside a nested ``dict`` / ``vec``.
    keep_empty_labels:
        Also emit labels that carry no value at all (as a row with an empty
        Value).  Used for consent records, where the label *is* the information
        and the record date is the item's timestamp.
    """
    rows: list[tuple[str, str, str, str]] = []

    def emit(path: list, value, date: str, depth: int, *, is_timestamp: bool = False) -> None:
        if nested_only and depth == 0:
            return
        text = _clean_text(value)
        if not text or _is_sensitive_value(text):
            return
        setting = _PATH_SEP.join(p for p in path if p) or category
        if is_timestamp:
            # A timestamp *is* the date of the record: show it in the Date column.
            rows.append((category, setting, "", text))
        else:
            rows.append((category, setting, text, date))

    def leaf_text(leaf: dict) -> str:
        if leaf.get("value") not in (None, ""):
            return _clean_text(leaf["value"])
        if leaf.get("timestamp_value"):
            return eh.epoch_to_iso(leaf["timestamp_value"], errors=errors)
        return ""

    def try_group(child, path: list, date: str, depth: int) -> bool:
        """Emit one 'name: value; ...' row for an unlabeled group of scalar settings."""
        if not isinstance(child, dict) or child.get("label") or child.get("title"):
            return False
        leaves = child.get("dict")
        if not isinstance(leaves, list) or not leaves:
            return False
        parts = []
        for leaf in leaves:
            if not isinstance(leaf, dict) or "label" not in leaf or "dict" in leaf or "vec" in leaf:
                return False
            text = leaf_text(leaf)
            if text and not _is_sensitive_value(text):
                parts.append(f"{_clean_text(leaf['label'])}: {text}")
        if len(parts) < 2:
            return False
        emit(path, "; ".join(parts), date, depth)
        return True

    def walk_children(children, path: list, date: str, depth: int) -> None:
        if not isinstance(children, list):
            walk(children, path, date, depth)
            return
        for child in children:
            if not try_group(child, path, date, depth):
                walk(child, path, date, depth)

    def walk(node, path: list, date: str, depth: int) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, path, date, depth)
            return
        if not isinstance(node, dict):
            emit(path, node, date, depth)
            return
        if not (node.keys() & _LV_KEYS):
            # Plain (old-style) mapping with English keys.
            for key, val in node.items():
                if key in ("media", "fbid", "timestamp"):
                    continue
                walk(val, path + [_PLAIN_KEY_LABELS.get(key, _clean_text(key))], date, depth)
            return
        if "label_values" in node:
            ts = node.get("timestamp")
            if isinstance(ts, (int, float)) and ts > 0:
                date = eh.epoch_to_iso(ts, errors=errors)
            walk_children(node["label_values"], path, date, depth)
            return
        label = node.get("label") or node.get("title")
        new_path = path + [_clean_text(label)] if label else path
        if keep_empty_labels and label and not any(k in node for k in ("value", "timestamp_value", "vec", "dict")):
            rows.append((category, _PATH_SEP.join(p for p in new_path if p), "", date))
        if node.get("timestamp_value"):
            emit(new_path, eh.epoch_to_iso(node["timestamp_value"], errors=errors), date, depth, is_timestamp=True)
        if "value" in node:
            emit(new_path, node["value"], date, depth)
        if scalars_only:
            return
        for key in ("vec", "dict"):
            if key in node:
                walk_children(node[key], new_path, date, depth + 1)

    walk(data, [], "", 0)
    return rows


def _settings_df(
    reader: ZipArchiveReader,
    errors: Counter,
    sources: list[tuple[str, str, dict]],
) -> pd.DataFrame:
    """Build a ``Category / Setting / Value / Date`` table from several files."""
    rows: list[tuple[str, str, str, str]] = []
    for path, category, options in sources:
        data = _read_json_data(reader, errors, path)
        if data is None:
            continue
        try:
            rows.extend(_label_value_rows(data, category, errors, **options))
        except Exception as e:
            logger.error("Exception caught: %s", e)
            errors[type(e).__name__] += 1
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["Category", "Setting", "Value", "Date"])


def _string_map_events(data, category: str, event: str, errors: Counter) -> list[tuple[str, str, str]]:
    """Return ``(category, event, date)`` for ``string_map_data`` style records.

    Only the first non-zero timestamp of each record is used; values (IP address,
    user agent, cookie name, username, e-mail, phone, ...) are never read.  The
    keys of ``string_map_data`` are localized, so they are not relied upon.
    """
    rows: list[tuple[str, str, str]] = []
    if not isinstance(data, dict):
        return rows
    for records in data.values():
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            ts = next(
                (v["timestamp"] for v in (record.get("string_map_data") or {}).values()
                 if isinstance(v, dict) and isinstance(v.get("timestamp"), (int, float)) and v["timestamp"] > 0),
                None,
            )
            if ts:
                rows.append((category, event, eh.epoch_to_iso(ts, errors=errors)))
    return rows



def ad_settings_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract ad-related and off-Instagram activity settings from Instagram.

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
        Columns: ``Category``, ``Setting``, ``Value``, ``Date``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is one setting related to advertising on Instagram or to activity of other websites and apps shared with Meta (ad preferences, ad-free subscription status, off-Meta activity settings, in-app ad messages).",
          "source_file": "ads_information/instagram_ads_and_businesses/ad_preferences.json, ads_information/instagram_ads_and_businesses/subscription_for_no_ads.json, ads_information/ads_and_topics/in-app_message.json, apps_and_websites_off_of_instagram/apps_and_websites/your_activity_off_meta_technologies_settings.json",
          "columns": {
            "Category": "Which Instagram file the row comes from.",
            "Setting": "Name of the setting, as displayed in the participant's Instagram language (nested settings are joined with ' › ').",
            "Value": "Value of the setting. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed.",
            "Date": "ISO 8601 timestamp of when the record was last updated, or the timestamp itself when the setting is a point in time (empty when not available)."
          }
        }

    Table config::

        {
          "id": "instagram_ad_settings",
          "title": {
            "en": "Ad and off-Instagram activity settings",
            "nl": "Advertentie-instellingen en activiteit buiten Instagram",
            "de": "Werbeeinstellungen und Aktivitäten außerhalb von Instagram",
            "pl": "Ustawienia reklam i aktywności poza Instagramem",
            "tr": "Reklam ve Instagram dışı etkinlik ayarları",
            "ar": "إعدادات الإعلانات والنشاط خارج إنستغرام",
            "ru": "Настройки рекламы и активности вне Instagram",
            "it": "Impostazioni degli annunci e dell'attività fuori da Instagram",
            "ro": "Setări pentru reclame și activitatea din afara Instagram",
            "es": "Ajustes de anuncios y de la actividad fuera de Instagram",
            "sq": "Cilësimet e reklamave dhe aktivitetit jashtë Instagram"
          },
          "description": {
            "en": "This table shows the settings and permissions related to the ads you see on Instagram and to the activity of other websites and apps that is shared with Instagram.",
            "nl": "Deze tabel toont de instellingen en machtigingen met betrekking tot de advertenties die je op Instagram ziet en de activiteit van andere websites en apps die met Instagram wordt gedeeld.",
            "de": "Diese Tabelle zeigt die Einstellungen und Berechtigungen zu den Werbeanzeigen, die Sie auf Instagram sehen, sowie zu Aktivitäten anderer Websites und Apps, die mit Instagram geteilt werden.",
            "pl": "Ta tabela pokazuje ustawienia i uprawnienia dotyczące reklam wyświetlanych na Instagramie oraz aktywności innych witryn i aplikacji udostępnianej Instagramowi.",
            "tr": "Bu tablo, Instagram'da gördüğün reklamlarla ve diğer web sitelerinin ve uygulamaların Instagram ile paylaşılan etkinliğiyle ilgili ayarları ve izinleri gösterir.",
            "ar": "يعرض هذا الجدول الإعدادات والأذونات المتعلقة بالإعلانات التي تراها على إنستغرام ونشاط المواقع والتطبيقات الأخرى الذي تتم مشاركته مع إنستغرام.",
            "ru": "В этой таблице показаны настройки и разрешения, связанные с рекламой, которую вы видите на Instagram, и с активностью других сайтов и приложений, передаваемой Instagram.",
            "it": "Questa tabella mostra le impostazioni e le autorizzazioni relative agli annunci che vedi su Instagram e all'attività di altri siti web e app condivisa con Instagram.",
            "ro": "Acest tabel arată setările și permisiunile legate de reclamele pe care le vezi pe Instagram și de activitatea altor site-uri și aplicații partajată cu Instagram.",
            "es": "Esta tabla muestra los ajustes y permisos relacionados con los anuncios que ves en Instagram y con la actividad de otros sitios web y aplicaciones que se comparte con Instagram.",
            "sq": "Kjo tabelë tregon cilësimet dhe lejet që lidhen me reklamat që sheh në Instagram dhe me aktivitetin e faqeve të tjera dhe aplikacioneve që ndahet me Instagram."
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
            },
            "Setting": {
              "en": "Setting",
              "nl": "Instelling",
              "de": "Einstellung",
              "pl": "Ustawienie",
              "tr": "Ayar",
              "ar": "الإعداد",
              "ru": "Настройка",
              "it": "Impostazione",
              "ro": "Setare",
              "es": "Ajuste",
              "sq": "Cilësimi"
            },
            "Value": {
              "en": "Value",
              "nl": "Waarde",
              "de": "Wert",
              "pl": "Wartość",
              "tr": "Değer",
              "ar": "القيمة",
              "ru": "Значение",
              "it": "Valore",
              "ro": "Valoare",
              "es": "Valor",
              "sq": "Vlera"
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
    return _settings_df(reader, errors, [
        ("ads_information/instagram_ads_and_businesses/ad_preferences.json", "Werbepräferenzen", {}),
        ("ads_information/instagram_ads_and_businesses/subscription_for_no_ads.json", "Werbefreies Abo", {}),
        ("ads_information/ads_and_topics/in-app_message.json", "In-App-Nachrichten", {}),
        ("apps_and_websites_off_of_instagram/apps_and_websites/your_activity_off_meta_technologies_settings.json", "Einstellungen zu Aktivitäten außerhalb von Meta", {}),
    ])


def consents_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the consents the participant gave to Instagram.

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
        Columns: ``Category``, ``Setting``, ``Value``, ``Date``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is one consent record the participant gave to Instagram (e.g. processing of data by advertising partners) with its date.",
          "source_file": "preferences/settings/consents.json",
          "columns": {
            "Category": "Which Instagram file the row comes from.",
            "Setting": "Name of the consent, as displayed in the participant's Instagram language.",
            "Value": "Status of the consent where available (often empty).",
            "Date": "ISO 8601 timestamp of the consent or its last update."
          }
        }

    Table config::

        {
          "id": "instagram_consents",
          "title": {
            "en": "Consents you gave",
            "nl": "Toestemmingen die je hebt gegeven",
            "de": "Von Ihnen erteilte Einwilligungen",
            "pl": "Udzielone zgody",
            "tr": "Verdiğin onaylar",
            "ar": "الموافقات التي منحتها",
            "ru": "Данные вами согласия",
            "it": "Consensi che hai dato",
            "ro": "Consimțămintele date",
            "es": "Consentimientos que diste",
            "sq": "Pëlqimet që ke dhënë"
          },
          "description": {
            "en": "This table shows which consents (for example to the processing of your data by advertising partners) you gave to Instagram and when.",
            "nl": "Deze tabel toont welke toestemmingen (bijvoorbeeld voor de verwerking van je gegevens door advertentiepartners) je aan Instagram hebt gegeven en wanneer.",
            "de": "Diese Tabelle zeigt, welche Einwilligungen (zum Beispiel zur Verarbeitung Ihrer Daten durch Werbepartner) Sie Instagram erteilt haben und wann.",
            "pl": "Ta tabela pokazuje, jakich zgód (na przykład na przetwarzanie Twoich danych przez partnerów reklamowych) udzieliłeś/aś Instagramowi i kiedy.",
            "tr": "Bu tablo, Instagram'a hangi onayları (örneğin verilerinin reklam ortakları tarafından işlenmesi için) ne zaman verdiğini gösterir.",
            "ar": "يعرض هذا الجدول الموافقات التي منحتها لإنستغرام (مثل معالجة بياناتك من قبل الشركاء الإعلانيين) ومتى منحتها.",
            "ru": "В этой таблице показано, какие согласия (например, на обработку ваших данных рекламными партнёрами) вы дали Instagram и когда.",
            "it": "Questa tabella mostra quali consensi (ad esempio al trattamento dei tuoi dati da parte di partner pubblicitari) hai dato a Instagram e quando.",
            "ro": "Acest tabel arată ce consimțăminte (de exemplu pentru prelucrarea datelor tale de către parteneri publicitari) ai dat Instagram și când.",
            "es": "Esta tabla muestra qué consentimientos (por ejemplo, al tratamiento de tus datos por parte de socios publicitarios) diste a Instagram y cuándo.",
            "sq": "Kjo tabelë tregon cilat pëlqime (për shembull për përpunimin e të dhënave të tua nga partnerët reklamues) i ke dhënë Instagram dhe kur."
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
            },
            "Setting": {
              "en": "Setting",
              "nl": "Instelling",
              "de": "Einstellung",
              "pl": "Ustawienie",
              "tr": "Ayar",
              "ar": "الإعداد",
              "ru": "Настройка",
              "it": "Impostazione",
              "ro": "Setare",
              "es": "Ajuste",
              "sq": "Cilësimi"
            },
            "Value": {
              "en": "Value",
              "nl": "Waarde",
              "de": "Wert",
              "pl": "Wartość",
              "tr": "Değer",
              "ar": "القيمة",
              "ru": "Значение",
              "it": "Valore",
              "ro": "Valoare",
              "es": "Valor",
              "sq": "Vlera"
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
    return _settings_df(reader, errors, [
        ("preferences/settings/consents.json", "Einwilligungen", {"keep_empty_labels": True}),
    ])


def link_history_settings_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the Instagram link-history setting.

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
        Columns: ``Category``, ``Setting``, ``Value``, ``Date``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "The participant's Instagram link-history setting (whether links opened in the app are kept) and when it was last updated.",
          "source_file": "logged_information/link_history/your_link_history_settings.json",
          "columns": {
            "Category": "Which Instagram file the row comes from.",
            "Setting": "Name of the setting, as displayed in the participant's Instagram language.",
            "Value": "Status of the setting where available (often empty).",
            "Date": "ISO 8601 timestamp of the last update."
          }
        }

    Table config::

        {
          "id": "instagram_link_history_settings",
          "title": {
            "en": "Link history setting",
            "nl": "Instelling voor linkgeschiedenis",
            "de": "Einstellung für den Link-Verlauf",
            "pl": "Ustawienie historii linków",
            "tr": "Bağlantı geçmişi ayarı",
            "ar": "إعداد سجل الروابط",
            "ru": "Настройка истории ссылок",
            "it": "Impostazione della cronologia dei link",
            "ro": "Setarea istoricului linkurilor",
            "es": "Ajuste del historial de enlaces",
            "sq": "Cilësimi i historikut të lidhjeve"
          },
          "description": {
            "en": "This table shows the setting for Instagram's link history (the links you opened from within the app) and when it was last updated.",
            "nl": "Deze tabel toont de instelling voor de linkgeschiedenis van Instagram (de links die je vanuit de app hebt geopend) en wanneer die voor het laatst is bijgewerkt.",
            "de": "Diese Tabelle zeigt die Einstellung für den Link-Verlauf von Instagram (die Links, die Sie in der App geöffnet haben) und wann sie zuletzt aktualisiert wurde.",
            "pl": "Ta tabela pokazuje ustawienie historii linków na Instagramie (linków otwartych w aplikacji) i kiedy zostało ostatnio zaktualizowane.",
            "tr": "Bu tablo, Instagram'ın bağlantı geçmişi ayarını (uygulama içinden açtığın bağlantılar) ve en son ne zaman güncellendiğini gösterir.",
            "ar": "يعرض هذا الجدول إعداد سجل الروابط في إنستغرام (الروابط التي فتحتها من داخل التطبيق) ومتى تم تحديثه آخر مرة.",
            "ru": "В этой таблице показана настройка истории ссылок Instagram (ссылки, открытые в приложении) и время её последнего обновления.",
            "it": "Questa tabella mostra l'impostazione della cronologia dei link di Instagram (i link aperti dall'app) e quando è stata aggiornata l'ultima volta.",
            "ro": "Acest tabel arată setarea istoricului linkurilor din Instagram (linkurile deschise din aplicație) și când a fost actualizată ultima dată.",
            "es": "Esta tabla muestra el ajuste del historial de enlaces de Instagram (los enlaces que abriste desde la aplicación) y cuándo se actualizó por última vez.",
            "sq": "Kjo tabelë tregon cilësimin e historikut të lidhjeve në Instagram (lidhjet që ke hapur nga aplikacioni) dhe kur u përditësua së fundmi."
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
            },
            "Setting": {
              "en": "Setting",
              "nl": "Instelling",
              "de": "Einstellung",
              "pl": "Ustawienie",
              "tr": "Ayar",
              "ar": "الإعداد",
              "ru": "Настройка",
              "it": "Impostazione",
              "ro": "Setare",
              "es": "Ajuste",
              "sq": "Cilësimi"
            },
            "Value": {
              "en": "Value",
              "nl": "Waarde",
              "de": "Wert",
              "pl": "Wartość",
              "tr": "Değer",
              "ar": "القيمة",
              "ru": "Значение",
              "it": "Valore",
              "ro": "Valoare",
              "es": "Valor",
              "sq": "Vlera"
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
    return _settings_df(reader, errors, [
        ("logged_information/link_history/your_link_history_settings.json", "Einstellung für den Link-Verlauf", {"keep_empty_labels": True}),
    ])


def security_and_login_events_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract security and login events (event type and time only).

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
        Columns: ``Category``, ``Event``, ``Date``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is one security- or login-related event (log-ins, log-outs, profile changes, password changes, sign-up, checkpoints). Only the type of event and the time are kept; IP addresses, user agents, devices, cookies, locations, contact details and usernames are never extracted.",
          "source_file": "security_and_login_information/login_and_profile_creation/ (login_activity, logout_activity, profile_activity, password_change_activity, signup_details, last_known_location)",
          "columns": {
            "Category": "Which security file the event comes from.",
            "Event": "Type of event (e.g. login, logout, profile changed), as displayed in the export.",
            "Date": "ISO 8601 timestamp of the event."
          }
        }

    Table config::

        {
          "id": "instagram_security_and_login_events",
          "title": {
            "en": "Security and login events",
            "nl": "Beveiligings- en inloggebeurtenissen",
            "de": "Sicherheits- und Anmeldeereignisse",
            "pl": "Zdarzenia związane z bezpieczeństwem i logowaniem",
            "tr": "Güvenlik ve oturum açma olayları",
            "ar": "أحداث الأمان وتسجيل الدخول",
            "ru": "События безопасности и входа",
            "it": "Eventi di sicurezza e di accesso",
            "ro": "Evenimente de securitate și autentificare",
            "es": "Eventos de seguridad e inicio de sesión",
            "sq": "Ngjarjet e sigurisë dhe hyrjes"
          },
          "description": {
            "en": "This table shows when you logged in or out of Instagram and other security-related events. Only the type of event and the time are included. IP addresses, devices, cookies and contact details are not included.",
            "nl": "Deze tabel toont wanneer je bent in- of uitgelogd bij Instagram en andere beveiligingsgebeurtenissen. Alleen het type gebeurtenis en het tijdstip zijn opgenomen. IP-adressen, apparaten, cookies en contactgegevens zijn niet opgenomen.",
            "de": "Diese Tabelle zeigt, wann Sie sich bei Instagram an- oder abgemeldet haben, sowie weitere sicherheitsrelevante Ereignisse. Es sind nur die Art des Ereignisses und der Zeitpunkt enthalten. IP-Adressen, Geräte, Cookies und Kontaktdaten sind nicht enthalten.",
            "pl": "Ta tabela pokazuje, kiedy logowałeś/aś się i wylogowywałeś/aś z Instagrama, oraz inne zdarzenia związane z bezpieczeństwem. Uwzględniono tylko rodzaj zdarzenia i czas. Adresy IP, urządzenia, pliki cookie i dane kontaktowe nie są uwzględnione.",
            "tr": "Bu tablo, Instagram'a ne zaman giriş yaptığını veya çıkış yaptığını ve diğer güvenlikle ilgili olayları gösterir. Yalnızca olay türü ve zaman dahildir. IP adresleri, cihazlar, çerezler ve iletişim bilgileri dahil değildir.",
            "ar": "يعرض هذا الجدول متى سجّلت الدخول إلى إنستغرام أو خرجت منه، وأحداث الأمان الأخرى. يتضمن نوع الحدث ووقته فقط. لا تتضمن البيانات عناوين IP والأجهزة وملفات تعريف الارتباط وبيانات الاتصال.",
            "ru": "В этой таблице показано, когда вы входили в Instagram и выходили из него, а также другие события безопасности. Включены только тип события и время. IP-адреса, устройства, файлы cookie и контактные данные не включены.",
            "it": "Questa tabella mostra quando hai effettuato l'accesso o sei uscito da Instagram e altri eventi legati alla sicurezza. Sono inclusi solo il tipo di evento e l'orario. Indirizzi IP, dispositivi, cookie e dati di contatto non sono inclusi.",
            "ro": "Acest tabel arată când te-ai conectat sau te-ai deconectat de la Instagram și alte evenimente legate de securitate. Sunt incluse doar tipul evenimentului și ora. Adresele IP, dispozitivele, cookie-urile și datele de contact nu sunt incluse.",
            "es": "Esta tabla muestra cuándo iniciaste o cerraste sesión en Instagram y otros eventos relacionados con la seguridad. Solo se incluyen el tipo de evento y la hora. No se incluyen direcciones IP, dispositivos, cookies ni datos de contacto.",
            "sq": "Kjo tabelë tregon kur ke hyrë ose dalë nga Instagram dhe ngjarje të tjera që lidhen me sigurinë. Përfshihen vetëm lloji i ngjarjes dhe koha. Adresat IP, pajisjet, cookies dhe të dhënat e kontaktit nuk përfshihen."
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
            },
            "Event": {
              "en": "Event",
              "nl": "Gebeurtenis",
              "de": "Ereignis",
              "pl": "Zdarzenie",
              "tr": "Olay",
              "ar": "الحدث",
              "ru": "Событие",
              "it": "Evento",
              "ro": "Eveniment",
              "es": "Evento",
              "sq": "Ngjarja"
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
    base = "security_and_login_information/login_and_profile_creation/"
    rows: list[tuple[str, str, str]] = []

    try:
        # string_map_data files: only the event time is used.
        for filename, category, event in [
            ("login_activity.json", "Login-Aktivität", "Login"),
            ("logout_activity.json", "Logout-Aktivität", "Logout"),
            ("password_change_activity.json", "Passwortänderungen", "Passwortänderung"),
            ("signup_details.json", "Registrierung", "Registrierung"),
            ("last_known_location.json", "Letzter bekannter Standort", "Standort hochgeladen"),
        ]:
            data = _read_json_data(reader, errors, base + filename)
            if data is not None:
                rows.extend(_string_map_events(data, category, event, errors))

        # profile_activity repeats the logins/logouts above and adds profile
        # changes and checkpoints: keep only events not already listed.
        seen = {date for _, _, date in rows}
        data = _read_json_data(reader, errors, base + "profile_activity.json")
        for item in data if isinstance(data, list) else []:
            label_values = [lv for lv in item.get("label_values", []) if isinstance(lv, dict)]
            ts = next((lv["timestamp_value"] for lv in label_values if lv.get("timestamp_value")), None)
            if not ts:
                continue
            date = eh.epoch_to_iso(ts, errors=errors)
            if date in seen:
                continue
            # The first plain value is the event type (e.g. profile changed).
            event = next((_clean_text(lv["value"]) for lv in label_values
                          if lv.get("value") and not _is_sensitive_value(_clean_text(lv["value"]))), "")
            rows.append(("Profilaktivität", event or "Profilaktivität", date))
            seen.add(date)

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["Category", "Event", "Date"]).sort_values("Date", ascending=False).reset_index(drop=True)


def removed_suggestions_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract when the participant removed suggested accounts (dates only, no names).

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
        Columns: ``Date``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is a time at which the participant removed an account from the accounts Instagram suggested to follow. The names and usernames of the removed accounts are not extracted.",
          "source_file": "connections/followers_and_following/removed_suggestions.json",
          "columns": {
            "Date": "ISO 8601 timestamp of when the suggestion was removed."
          }
        }

    Table config::

        {
          "id": "instagram_removed_suggestions",
          "title": {
            "en": "Removed follow suggestions",
            "nl": "Verwijderde volgsuggesties",
            "de": "Entfernte Vorschläge zum Folgen",
            "pl": "Usunięte propozycje obserwowania",
            "tr": "Kaldırılan takip önerileri",
            "ar": "اقتراحات المتابعة التي أزلتها",
            "ru": "Удалённые рекомендации подписок",
            "it": "Suggerimenti da seguire rimossi",
            "ro": "Sugestii de urmărire eliminate",
            "es": "Sugerencias de seguimiento eliminadas",
            "sq": "Sugjerimet e ndjekjes të hequra"
          },
          "description": {
            "en": "This table shows when you removed accounts from the accounts Instagram suggested you follow. The names of those accounts are not included.",
            "nl": "Deze tabel toont wanneer je accounts hebt verwijderd uit de accounts die Instagram je voorstelde om te volgen. De namen van die accounts zijn niet opgenomen.",
            "de": "Diese Tabelle zeigt, wann Sie Accounts aus den von Instagram vorgeschlagenen Accounts zum Folgen entfernt haben. Die Namen dieser Accounts sind nicht enthalten.",
            "pl": "Ta tabela pokazuje, kiedy usunąłeś/usunęłaś konta z kont, które Instagram proponował Ci obserwować. Nazwy tych kont nie są uwzględnione.",
            "tr": "Bu tablo, Instagram'ın takip etmeni önerdiği hesaplar arasından bazı hesapları ne zaman kaldırdığını gösterir. Bu hesapların adları dahil değildir.",
            "ar": "يعرض هذا الجدول متى أزلت حسابات من الحسابات التي اقترح عليك إنستغرام متابعتها. لا تتضمن البيانات أسماء هذه الحسابات.",
            "ru": "В этой таблице показано, когда вы удаляли аккаунты из рекомендаций Instagram для подписки. Названия этих аккаунтов не включены.",
            "it": "Questa tabella mostra quando hai rimosso account dai suggerimenti che Instagram ti proponeva di seguire. I nomi di questi account non sono inclusi.",
            "ro": "Acest tabel arată când ai eliminat conturi din sugestiile de urmărire pe care ți le-a oferit Instagram. Numele acestor conturi nu sunt incluse.",
            "es": "Esta tabla muestra cuándo eliminaste cuentas de las sugerencias de seguimiento que Instagram te proponía. No se incluyen los nombres de esas cuentas.",
            "sq": "Kjo tabelë tregon kur ke hequr llogari nga sugjerimet që Instagram të propozonte t'i ndiqje. Emrat e këtyre llogarive nuk përfshihen."
          },
          "headers": {
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
    data = _read_json_data(reader, errors, "connections/followers_and_following/removed_suggestions.json")
    dates: list[str] = []

    try:
        for item in data if isinstance(data, list) else []:
            ts = item.get("timestamp") if isinstance(item, dict) else None
            if ts:
                dates.append(eh.epoch_to_iso(ts, errors=errors))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    if not dates:
        return pd.DataFrame()
    return pd.DataFrame({"Date": sorted(dates, reverse=True)})


# Extractor registry & platform info
# ---------------------------------------------------------------------------

#: Mapping from the string names used in port_config.json to actual extractor functions.
EXTRACTOR_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "followers_to_df": followers_to_df,
    "following_to_df": following_to_df,
    "ads_viewed_to_df": ads_viewed_to_df,
    "other_categories_used_to_reach_you_to_df":other_categories_used_to_reach_you_to_df,
    "posts_viewed_to_df": posts_viewed_to_df,
    "videos_watched_to_df": videos_watched_to_df,
    "post_comments_to_df": post_comments_to_df,
    "liked_comments_to_df": liked_comments_to_df,
    "liked_posts_to_df": liked_posts_to_df,
    "searches_to_df": searches_to_df,
    "profile_searches_to_df": profile_searches_to_df,
    "story_likes_to_df": story_likes_to_df,
    "stories_viewed_to_df": stories_viewed_to_df,
    "threads_viewed_to_df": threads_viewed_to_df,
    "saved_posts_to_df": saved_posts_to_df,
    "ad_settings_to_df": ad_settings_to_df,
    "consents_to_df": consents_to_df,
    "link_history_settings_to_df": link_history_settings_to_df,
    "security_and_login_events_to_df": security_and_login_events_to_df,
    "removed_suggestions_to_df": removed_suggestions_to_df,
}


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