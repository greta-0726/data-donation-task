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


def who_youve_followed_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the list of profiles and pages you follow on Facebook.

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
        Columns: ``Name``, ``Timestamp``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a Facebook profile or page that the participant follows, including the name and the time they started following.",
          "source_file": "who_you've_followed.json (or pages_and_profiles_you_follow.json if absent)",
          "columns": {
            "Name": "Name of the followed profile or page.",
            "Timestamp": "ISO 8601 timestamp of when the participant started following."
          }
        }

    Table config::

        {
          "id": "facebook_who_youve_followed",
          "title": {
            "en": "Who you follow",
            "nl": "Wie je volgt",
            "de": "Wem Sie folgen",
            "pl": "Kogo obserwujesz",
            "tr": "Takip ettiklerin",
            "ar": "من تتابعهم",
            "ru": "На кого вы подписаны",
            "it": "Chi segui",
            "ro": "Pe cine urmărești",
            "es": "A quién sigues",
            "sq": "Kë ndjek"
          },
          "description": {
            "en": "This table shows the Facebook profiles and pages you currently follow.",
            "nl": "Deze tabel toont de Facebook-profielen en -pagina's die je momenteel volgt.",
            "de": "Diese Tabelle zeigt die Facebook-Profile und -Seiten, denen Sie aktuell folgen.",
            "pl": "Ta tabela pokazuje profile i strony na Facebooku, które aktualnie obserwujesz.",
            "tr": "Bu tablo şu anda Facebook'ta takip ettiğin profilleri ve sayfaları gösterir.",
            "ar": "يعرض هذا الجدول ملفات الأشخاص وصفحات فيسبوك التي تتابعها حاليًا.",
            "ru": "В этой таблице показаны профили и страницы Facebook, на которые вы сейчас подписаны.",
            "it": "Questa tabella mostra i profili e le pagine di Facebook che segui attualmente.",
            "ro": "Acest tabel arată profilurile și paginile de Facebook pe care le urmărești în prezent.",
            "es": "Esta tabla muestra los perfiles y páginas de Facebook que sigues actualmente.",
            "sq": "Kjo tabelë tregon profilet dhe faqet e Facebook-ut që ndjek aktualisht."
          },
          "headers": {
            "Name": {
              "en": "Name",
              "nl": "Naam",
              "de": "Name",
              "pl": "Nazwa",
              "tr": "Ad",
              "ar": "الاسم",
              "ru": "Название",
              "it": "Nome",
              "ro": "Nume",
              "es": "Nombre",
              "sq": "Emri"
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
            }
          }
        }
    """
    result = reader.json("who_you've_followed.json")
    if not result.found:
        # Some exports only contain pages_and_profiles_you_follow.json, which
        # holds the same information (it used to be a separate, duplicate table).
        result = reader.json("pages_and_profiles_you_follow.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        if "following_v3" in d:  # pyright: ignore
            for item in d["following_v3"]:  # pyright: ignore
                datapoints.append((
                    eh.fix_latin1_string(item.get("name", "")),
                    eh.epoch_to_iso(item.get("timestamp", {}), errors=errors)
                ))
        else:
            for item in d["pages_followed_v2"]:  # pyright: ignore
                datapoints.append((
                    eh.fix_latin1_string(item.get("title", "")),
                    eh.epoch_to_iso(item.get("timestamp", ""), errors=errors)
                ))

        out = pd.DataFrame(datapoints, columns=["Name", "Timestamp"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def facebook_reels_usage_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract Facebook Reels usage information.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Reel interaction``, ``Number of Reels``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a type of interaction the participant had with Facebook Reels and the corresponding number of Reels.",
          "source_file": "facebook_reels_usage_information.json",
          "columns": {
            "Reel interaction": "Type of interaction with Facebook Reels.",
            "Number of Reels": "Number of Reels associated with the interaction."
          }
        }

    Table config::

        {
          "id": "facebook_reels_usage",
          "title": {
            "en": "Interactions with Facebook Reels",
            "nl": "Interacties met Facebook Reels",
            "de": "Interaktionen mit Facebook Reels",
            "pl": "Interakcje z Reels na Facebooku",
            "tr": "Facebook Reels ile etkileşimlerin",
            "ar": "تفاعلاتك مع Reels على فيسبوك",
            "ru": "Взаимодействия с Reels на Facebook",
            "it": "Interazioni con i Reels di Facebook",
            "ro": "Interacțiunile tale cu Reels pe Facebook",
            "es": "Interacciones con Reels de Facebook",
            "sq": "Ndërveprimet e tua me Reels në Facebook"
          },
          "description": {
            "en": "This table shows your interactions with Facebook Reels, such as videos you've watched or engaged with.",
            "nl": "Deze tabel toont je interacties met Facebook Reels, zoals video's die je hebt bekeken of waarmee je hebt gecommuniceerd.",
            "de": "Diese Tabelle zeigt Ihre Interaktionen mit Facebook Reels, zum Beispiel Videos, die Sie sich angesehen oder mit denen Sie interagiert haben.",
            "pl": "Ta tabela pokazuje Twoje interakcje z Facebook Reels, na przykład filmy, które obejrzałeś/aś lub z którymi wchodziłeś/aś w interakcję.",
            "tr": "Bu tablo, izlediğin veya etkileşimde bulunduğun videolar gibi Facebook Reels ile olan etkileşimlerini gösterir.",
            "ar": "يعرض هذا الجدول تفاعلاتك مع Reels على فيسبوك، مثل مقاطع الفيديو التي شاهدتها أو تفاعلت معها.",
            "ru": "В этой таблице показаны ваши взаимодействия с Reels на Facebook, например просмотренные видео или видео, с которыми вы взаимодействовали.",
            "it": "Questa tabella mostra le tue interazioni con i Reels di Facebook, ad esempio i video che hai guardato o con cui hai interagito.",
            "ro": "Acest tabel arată interacțiunile tale cu Reels pe Facebook, cum ar fi videoclipurile pe care le-ai vizionat sau cu care ai interacționat.",
            "es": "Esta tabla muestra tus interacciones con los Reels de Facebook, como los videos que has visto o con los que has interactuado.",
            "sq": "Kjo tabelë tregon ndërveprimet e tua me Reels në Facebook, si videot që ke parë ose me të cilat ke ndërvepruar."
          },
          "headers": {
            "Reel interaction": {
              "en": "Reel interaction",
              "nl": "Interactie met reels",
              "de": "Reel-Interaktion",
              "pl": "Interakcja z Reels",
              "tr": "Reels Etkileşimi",
              "ar": "التفاعل مع Reels",
              "ru": "Взаимодействие с Reels",
              "it": "Interazione con i Reels",
              "ro": "Interacțiune cu Reels",
              "es": "Interacción con Reels",
              "sq": "Ndërveprim me Reels"
            },
            "Number of Reels": {
              "en": "Number of Reels",
              "nl": "Aantal Reels",
              "de": "Anzahl der Reels",
              "pl": "Liczba Reels",
              "tr": "Reels Sayısı",
              "ar": "عدد Reels",
              "ru": "Количество Reels",
              "it": "Numero di Reels",
              "ro": "Număr de Reels",
              "es": "Número de Reels",
              "sq": "Numri i Reels"
            }
          }
        }
    """
    result = reader.json("facebook_reels_usage_information.json")
    if not result.found:
        return pd.DataFrame()

    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d.get("label_values", [])  # pyright: ignore
        d = items[0]

        for item in d["dict"]:
            denested_dict = eh.dict_denester(item)

            label = eh.fix_latin1_string(
                eh.find_item(denested_dict, "label")
            )
            value = eh.find_item(denested_dict, "value")

            # Shorter, participant-friendly German labels
            reel_label_translations = {
                "The number of Reels you have seen in the last 7 days":
                    "In den letzten 7 Tagen angesehen",

                "The number of Reels you have seen in the last 30 days":
                    "In den letzten 30 Tagen angesehen",

                "The number of Reels you have liked in the last 30 days":
                    "In den letzten 30 Tagen mit „Gefällt mir“ markiert",

                "The number of Reels you have seen in the horizontal Reels tray in the last 7 days":
                    "In den letzten 7 Tagen im horizontalen Reels-Bereich angesehen",

                "The number of Reels you have clicked from the horizontal Reels tray in the last 7 days":
                    "In den letzten 7 Tagen im horizontalen Reels-Bereich angeklickt",
            }

            label = reel_label_translations.get(label, label)

            label_lower = label.lower()

            # Fallback for slightly different German wording in Meta exports
            if (
                "horizontalen reels-bereich" in label_lower
                and "gesehen" in label_lower
                and "7" in label_lower
            ):
                label = (
                    "In den letzten 7 Tagen im horizontalen "
                    "Reels-Bereich angesehen"
                )

            datapoints.append((
                label,
                value,
            ))

        out = pd.DataFrame(
            datapoints,
            columns=["Reel interaction", "Number of Reels"]
        )  # pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out

def video_consumption_summary_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = (
        "your_facebook_activity/other_activity/"
        "your_video_consumption_summary.json"
    ),
) -> pd.DataFrame:
    """Extract the participant's Facebook video-consumption summary.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"your_facebook_activity/other_activity/"
        "your_video_consumption_summary.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Metric``, ``Duration (seconds)``.
        Entries without a numeric duration are excluded.
        Empty DataFrame when the file is absent, contains no numeric values,
        or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one aggregate measure of the participant's video-viewing activity on Facebook.",
          "source_file": "your_facebook_activity/other_activity/your_video_consumption_summary.json",
          "columns": {
            "Metric": "Description of the video-viewing measure and its reporting period.",
            "Duration (seconds)": "Total viewing time reported by Meta, expressed in seconds."
          }
        }

    Table config::

        {
          "id": "facebook_video_consumption_summary",
          "title": {
            "en": "Video viewing summary",
            "nl": "Overzicht van bekeken video's",
            "de": "Zusammenfassung Ihrer Videowiedergabe",
            "pl": "Podsumowanie oglądania filmów",
            "tr": "Video izleme özeti",
            "ar": "ملخص مشاهدة مقاطع الفيديو",
            "ru": "Сводка просмотра видео",
            "it": "Riepilogo della visualizzazione dei video",
            "ro": "Rezumatul vizionării videoclipurilor",
            "es": "Resumen de visualización de vídeos",
            "sq": "Përmbledhje e shikimit të videove"
          },
          "description": {
            "en": "This table shows how much time you spent watching different types of video content on Facebook during several recent periods.",
            "nl": "Deze tabel toont hoeveel tijd je in verschillende recente perioden hebt besteed aan het bekijken van verschillende soorten video-inhoud op Facebook.",
            "de": "Diese Tabelle zeigt, wie viel Zeit Sie in verschiedenen Zeiträumen mit unterschiedlichen Arten von Videoinhalten auf Facebook verbracht haben.",
            "pl": "Ta tabela pokazuje czas spędzony na oglądaniu różnych rodzajów treści wideo na Facebooku w kilku ostatnich okresach.",
            "tr": "Bu tablo, son dönemlerde Facebook'ta farklı video içerik türlerini izleyerek ne kadar zaman geçirdiğini gösterir.",
            "ar": "يعرض هذا الجدول الوقت الذي قضيته في مشاهدة أنواع مختلفة من محتوى الفيديو على فيسبوك خلال فترات حديثة متعددة.",
            "ru": "В этой таблице показано время, проведённое за просмотром различных видов видеоконтента на Facebook за несколько последних периодов.",
            "it": "Questa tabella mostra il tempo trascorso a guardare diversi tipi di contenuti video su Facebook in vari periodi recenti.",
            "ro": "Acest tabel arată timpul petrecut vizionând diferite tipuri de conținut video pe Facebook în mai multe perioade recente.",
            "es": "Esta tabla muestra el tiempo que pasaste viendo distintos tipos de contenido de vídeo en Facebook durante varios periodos recientes.",
            "sq": "Kjo tabelë tregon kohën e kaluar duke parë lloje të ndryshme videosh në Facebook gjatë disa periudhave të fundit."
          },
          "headers": {
            "Metric": {
              "en": "Viewing measure",
              "nl": "Kijkmaatstaf",
              "de": "Wiedergabemaß",
              "pl": "Miara oglądania",
              "tr": "İzleme ölçütü",
              "ar": "مقياس المشاهدة",
              "ru": "Показатель просмотра",
              "it": "Misura di visualizzazione",
              "ro": "Indicator de vizionare",
              "es": "Medida de visualización",
              "sq": "Matësi i shikimit"
            },
            "Duration (seconds)": {
              "en": "Viewing time (seconds)",
              "nl": "Kijktijd (seconden)",
              "de": "Wiedergabezeit (Sekunden)",
              "pl": "Czas oglądania (sekundy)",
              "tr": "İzleme süresi (saniye)",
              "ar": "وقت المشاهدة (بالثواني)",
              "ru": "Время просмотра (секунды)",
              "it": "Tempo di visualizzazione (secondi)",
              "ro": "Timp de vizionare (secunde)",
              "es": "Tiempo de visualización (segundos)",
              "sq": "Koha e shikimit (sekonda)"
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
            metric = eh.fix_latin1_string(
                item.get("label", "")
            )

            raw_duration = eh.fix_latin1_string(
                item.get("value", "")
            )

            # Extract digits independently of the language and unit.
            #
            # Examples:
            # "120 seconds"       -> 120
            # "120 Sekunden"      -> 120
            # "1.200 Sekunden"    -> 1200
            # "\u00c2\u00a0Sekunden" -> excluded because no number exists
            duration_parts = re.findall(r"\d+", raw_duration)

            if not metric or not duration_parts:
                continue

            duration_seconds = int("".join(duration_parts))

            datapoints.append((
                metric,
                duration_seconds,
            ))

        return pd.DataFrame(
            datapoints,
            columns=[
                "Metric",
                "Duration (seconds)",
            ],
        )

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return pd.DataFrame()

def last_28_days_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract how many videos you watched in the last 28 days on Facebook Watch.

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
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Contains the number of videos the participant watched on Facebook in the past 28 days.",
          "source_file": "your_facebook_watch_activity_in_the_last_28_days.json",
          "columns": {
            "Count": "Number of videos watched in the last 28 days."
          }
        }

    Table config::

        {
          "id": "facebook_last_28",
          "title": {
            "en": "How many videos you watched in the last 28 days",
            "nl": "Hoeveel video's je de afgelopen 28 dagen hebt bekeken",
            "de": "Wie viele Videos Sie in den letzten 28 Tagen angesehen haben",
            "pl": "Ile filmów obejrzałeś/aś w ciągu ostatnich 28 dni",
            "tr": "Son 28 günde izlediğin video sayısı",
            "ar": "عدد الفيديوهات التي شاهدتها خلال آخر 28 يومًا",
            "ru": "Сколько видео вы посмотрели за последние 28 дней",
            "it": "Quanti video hai guardato negli ultimi 28 giorni",
            "ro": "Câte videoclipuri ai vizionat în ultimele 28 de zile",
            "es": "Cuántos videos viste en los últimos 28 días",
            "sq": "Sa video ke parë në 28 ditët e fundit"
          },
          "description": {
            "en": "This table indicates the number of videos you have watched on Facebook in the past 28 days.",
            "nl": "Deze tabel geeft het aantal video's aan dat je de afgelopen 28 dagen op Facebook hebt bekeken.",
            "de": "Diese Tabelle zeigt, wie viele Videos Sie in den letzten 28 Tagen auf Facebook angesehen haben.",
            "pl": "Ta tabela pokazuje liczbę filmów, które obejrzałeś/aś na Facebooku w ciągu ostatnich 28 dni.",
            "tr": "Bu tablo, son 28 günde Facebook'ta izlediğin video sayısını gösterir.",
            "ar": "يوضح هذا الجدول عدد مقاطع الفيديو التي شاهدتها على فيسبوك خلال آخر 28 يومًا.",
            "ru": "В этой таблице указано количество видео, которые вы посмотрели на Facebook за последние 28 дней.",
            "it": "Questa tabella indica il numero di video che hai guardato su Facebook negli ultimi 28 giorni.",
            "ro": "Acest tabel indică numărul de videoclipuri pe care le-ai vizionat pe Facebook în ultimele 28 de zile.",
            "es": "Esta tabla indica el número de videos que has visto en Facebook en los últimos 28 días.",
            "sq": "Kjo tabelë tregon numrin e videove që ke parë në Facebook gjatë 28 ditëve të fundit."
          },
          "headers": {
            "Count": {
              "en": "Count",
              "nl": "Aantal",
              "de": "Anzahl",
              "pl": "Liczba",
              "tr": "Sayı",
              "ar": "العدد",
              "ru": "Количество",
              "it": "Numero",
              "ro": "Număr",
              "es": "Número",
              "sq": "Numri"
            }
          }
        }
    """
    result = reader.json("your_facebook_watch_activity_in_the_last_28_days.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        denested_dict = eh.dict_denester(d)
        datapoints.append((
            eh.find_item(denested_dict, "-value"),
        ))

        out = pd.DataFrame(datapoints, columns=["Count"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out

def time_spent_on_facebook_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = (
        "your_facebook_activity/other_activity/"
        "time_spent_on_facebook.json"
    ),
) -> pd.DataFrame:
    """Extract recorded Facebook usage intervals.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"your_facebook_activity/other_activity/"
        "time_spent_on_facebook.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Start time``, ``End time``, ``Duration (seconds)``,
        ``Timezone``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one recorded interval during which the participant used Facebook.",
          "source_file": "your_facebook_activity/other_activity/time_spent_on_facebook.json",
          "columns": {
            "Start time": "ISO 8601 timestamp marking the beginning of the recorded usage interval.",
            "End time": "ISO 8601 timestamp marking the end of the recorded usage interval.",
            "Duration (seconds)": "Length of the recorded usage interval in seconds.",
            "Timezone": "Timezone reported by Facebook for the relevant period."
          }
        }

    Table config::

        {
          "id": "facebook_time_spent",
          "title": {
            "en": "Time spent on Facebook",
            "nl": "Tijd besteed op Facebook",
            "de": "Auf Facebook verbrachte Zeit",
            "pl": "Czas spędzony na Facebooku",
            "tr": "Facebook'ta geçirilen süre",
            "ar": "الوقت الذي قضيته على فيسبوك",
            "ru": "Время, проведённое на Facebook",
            "it": "Tempo trascorso su Facebook",
            "ro": "Timp petrecut pe Facebook",
            "es": "Tiempo pasado en Facebook",
            "sq": "Koha e kaluar në Facebook"
          },
          "description": {
            "en": "This table shows recorded intervals during which you used Facebook, including their start, end, and duration.",
            "nl": "Deze tabel toont geregistreerde perioden waarin je Facebook gebruikte, inclusief het begin, einde en de duur.",
            "de": "Diese Tabelle zeigt erfasste Zeiträume, in denen Sie Facebook genutzt haben, einschließlich Beginn, Ende und Dauer.",
            "pl": "Ta tabela pokazuje zarejestrowane okresy korzystania z Facebooka, w tym ich początek, koniec i czas trwania.",
            "tr": "Bu tablo, Facebook'u kullandığın kaydedilmiş zaman aralıklarını başlangıç, bitiş ve süreleriyle birlikte gösterir.",
            "ar": "يعرض هذا الجدول الفترات المسجلة التي استخدمت فيها فيسبوك، بما في ذلك وقت البداية والنهاية والمدة.",
            "ru": "В этой таблице показаны зарегистрированные периоды использования Facebook, включая время начала, окончания и продолжительность.",
            "it": "Questa tabella mostra gli intervalli registrati durante i quali hai utilizzato Facebook, inclusi inizio, fine e durata.",
            "ro": "Acest tabel arată intervalele înregistrate în care ai utilizat Facebook, inclusiv începutul, sfârșitul și durata acestora.",
            "es": "Esta tabla muestra los intervalos registrados durante los que utilizaste Facebook, incluidos su inicio, fin y duración.",
            "sq": "Kjo tabelë tregon intervalet e regjistruara gjatë të cilave ke përdorur Facebook, duke përfshirë fillimin, përfundimin dhe kohëzgjatjen."
          },
          "headers": {
            "Start time": {
              "en": "Start time",
              "nl": "Begintijd",
              "de": "Startzeit",
              "pl": "Czas rozpoczęcia",
              "tr": "Başlangıç zamanı",
              "ar": "وقت البدء",
              "ru": "Время начала",
              "it": "Ora di inizio",
              "ro": "Ora de început",
              "es": "Hora de inicio",
              "sq": "Koha e fillimit"
            },
            "End time": {
              "en": "End time",
              "nl": "Eindtijd",
              "de": "Endzeit",
              "pl": "Czas zakończenia",
              "tr": "Bitiş zamanı",
              "ar": "وقت الانتهاء",
              "ru": "Время окончания",
              "it": "Ora di fine",
              "ro": "Ora de încheiere",
              "es": "Hora de finalización",
              "sq": "Koha e përfundimit"
            },
            "Duration (seconds)": {
              "en": "Duration (seconds)",
              "nl": "Duur (seconden)",
              "de": "Dauer (Sekunden)",
              "pl": "Czas trwania (sekundy)",
              "tr": "Süre (saniye)",
              "ar": "المدة (بالثواني)",
              "ru": "Продолжительность (секунды)",
              "it": "Durata (secondi)",
              "ro": "Durată (secunde)",
              "es": "Duración (segundos)",
              "sq": "Kohëzgjatja (sekonda)"
            },
            "Timezone": {
              "en": "Timezone",
              "nl": "Tijdzone",
              "de": "Zeitzone",
              "pl": "Strefa czasowa",
              "tr": "Saat dilimi",
              "ar": "المنطقة الزمنية",
              "ru": "Часовой пояс",
              "it": "Fuso orario",
              "ro": "Fus orar",
              "es": "Zona horaria",
              "sq": "Zona kohore"
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

        timezone_periods = []
        fallback_timezones = []

        # Find the timezone history by structure rather than translated labels.
        for group in label_values:
            for entry in group.get("dict", []):
                timezone = eh.fix_latin1_string(
                    entry.get("value", "")
                )
                effective_date = eh.fix_latin1_string(
                    entry.get("label", "")
                )

                if not timezone:
                    continue

                fallback_timezones.append(timezone)

                try:
                    parsed_date = pd.to_datetime(
                        effective_date
                    ).date()

                    timezone_periods.append((
                        parsed_date,
                        timezone,
                    ))
                except (TypeError, ValueError):
                    pass

        timezone_periods.sort(key=lambda item: item[0])

        def timezone_for_epoch(epoch: int) -> str:
            """Return the timezone applying at the given epoch."""

            if timezone_periods:
                interval_date = pd.to_datetime(
                    epoch,
                    unit="s",
                    utc=True,
                ).date()

                applicable = [
                    timezone
                    for effective_date, timezone in timezone_periods
                    if effective_date <= interval_date
                ]

                if applicable:
                    return applicable[-1]

                return timezone_periods[0][1]

            if fallback_timezones:
                return fallback_timezones[-1]

            return ""

        # The intervals group is identified by its nested vec/dict structure.
        for group in label_values:
            intervals = group.get("vec", [])

            for interval in intervals:
                timestamps = []

                for entry in interval.get("dict", []):
                    timestamp = entry.get("timestamp_value")

                    try:
                        timestamps.append(int(timestamp))
                    except (TypeError, ValueError):
                        continue

                if len(timestamps) < 2:
                    continue

                start_epoch = min(timestamps)
                end_epoch = max(timestamps)

                datapoints.append((
                    eh.epoch_to_iso(
                        start_epoch,
                        errors=errors,
                    ),
                    eh.epoch_to_iso(
                        end_epoch,
                        errors=errors,
                    ),
                    end_epoch - start_epoch,
                    timezone_for_epoch(start_epoch),
                ))

        return pd.DataFrame(
            datapoints,
            columns=[
                "Start time",
                "End time",
                "Duration (seconds)",
                "Timezone",
            ],
        )

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return pd.DataFrame()

def your_search_history_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract Facebook search history.

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
        Columns: ``Search term``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a search query the participant made on Facebook, including the search term and date.",
          "source_file": "logged_information/search/your_search_history.json",
          "columns": {
            "Search term": "The search query entered by the participant.",
            "Date": "ISO 8601 timestamp of when the search was made."
          }
        }

    Table config::

        {
          "id": "facebook_search_history",
          "title": {
            "en": "Your search history",
            "nl": "Je zoekgeschiedenis",
            "de": "Ihr Suchverlauf",
            "pl": "Historia wyszukiwania",
            "tr": "Arama geçmişin",
            "ar": "سجل بحثك",
            "ru": "История ваших поисковых запросов",
            "it": "La tua cronologia di ricerca",
            "ro": "Istoricul căutărilor tale",
            "es": "Tu historial de búsqueda",
            "sq": "Historiku i kërkimeve të tua"
          },
          "description": {
            "en": "This table contains a record of your search queries on Facebook.",
            "nl": "Deze tabel bevat een overzicht van je zoekopdrachten op Facebook.",
            "de": "Diese Tabelle enthält eine Übersicht Ihrer Suchanfragen auf Facebook.",
            "pl": "Ta tabela zawiera zapis Twoich zapytań wyszukiwania na Facebooku.",
            "tr": "Bu tablo, Facebook'ta yaptığın arama sorgularının bir kaydını içerir.",
            "ar": "يحتوي هذا الجدول على سجل لعمليات البحث التي أجريتها على فيسبوك.",
            "ru": "В этой таблице содержится история ваших поисковых запросов на Facebook.",
            "it": "Questa tabella contiene un registro delle tue ricerche su Facebook.",
            "ro": "Acest tabel conține o evidență a căutărilor pe care le-ai făcut pe Facebook.",
            "es": "Esta tabla contiene un registro de tus búsquedas en Facebook.",
            "sq": "Kjo tabelë përmban një regjistër të kërkimeve që ke bërë në Facebook."
          },
          "headers": {
            "Search term": {
              "en": "Search term",
              "nl": "Zoekterm",
              "de": "Suchbegriff",
              "pl": "Wyszukiwane hasło",
              "tr": "Arama Terimi",
              "ar": "مصطلح البحث",
              "ru": "Поисковый запрос",
              "it": "Termine di ricerca",
              "ro": "Termen de căutare",
              "es": "Término de búsqueda",
              "sq": "Termi i kërkimit"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum",
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
                "en": "Terms you searched for",
                "nl": "Zoektermen waar je naar zocht",
                "de": "Begriffe, nach denen Sie gesucht haben",
                "pl": "Hasła, których szukałeś/aś",
                "tr": "Aradığın terimler",
                "ar": "الكلمات التي بحثت عنها",
                "ru": "Термины, которые вы искали",
                "it": "Termini che hai cercato",
                "ro": "Termeni pe care i-ai căutat",
                "es": "Términos que buscaste",
                "sq": "Termat që ke kërkuar"
              },
              "type": "wordcloud",
              "textColumn": "Search term",
              "tokenize": false
            }
          ]
        }
    """
    result = reader.json("logged_information/search/your_search_history.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["searches_v2"]  # pyright: ignore
        for item in items:
            denested_dict = eh.dict_denester(item)

            datapoints.append((
                eh.fix_latin1_string(eh.find_item(denested_dict, "text")),
                eh.epoch_to_iso(eh.find_item(denested_dict, "timestamp"), errors=errors),
            ))

        out = pd.DataFrame(datapoints, columns=["Search term", "Date"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def your_friends_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the number of Facebook friends.

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
        Columns: ``Number of friends``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Contains the total number of friends the participant has on Facebook.",
          "source_file": "your_friends.json",
          "columns": {
            "Number of friends": "Total count of Facebook friends."
          }
        }

    Table config::

        {
          "id": "facebook_your_friends",
          "title": {
            "en": "Your friends on Facebook",
            "nl": "Je vrienden op Facebook",
            "de": "Ihre Freunde auf Facebook",
            "pl": "Twoi znajomi na Facebooku",
            "tr": "Facebook'taki arkadaşların",
            "ar": "أصدقاؤك على فيسبوك",
            "ru": "Ваши друзья на Facebook",
            "it": "I tuoi amici su Facebook",
            "ro": "Prietenii tăi de pe Facebook",
            "es": "Tus amigos en Facebook",
            "sq": "Miqtë e tu në Facebook"
          },
          "description": {
            "en": "This table lists your current friends on Facebook.",
            "nl": "Deze tabel toont je huidige vrienden op Facebook.",
            "de": "Diese Tabelle zeigt Ihre aktuellen Freunde auf Facebook.",
            "pl": "Ta tabela zawiera listę Twoich obecnych znajomych na Facebooku.",
            "tr": "Bu tablo, Facebook'taki mevcut arkadaşlarını listeler.",
            "ar": "يعرض هذا الجدول أصدقاءك الحاليين على فيسبوك.",
            "ru": "В этой таблице перечислены ваши текущие друзья на Facebook.",
            "it": "Questa tabella elenca i tuoi amici attuali su Facebook.",
            "ro": "Acest tabel listează prietenii tăi actuali de pe Facebook.",
            "es": "Esta tabla enumera tus amigos actuales en Facebook.",
            "sq": "Kjo tabelë liston miqtë e tu aktualë në Facebook."
          },
          "headers": {
            "Number of friends": {
              "en": "Number of friends",
              "nl": "Aantal vrienden op facebook",
              "de": "Anzahl der Freunde",
              "pl": "Liczba znajomych",
              "tr": "Arkadaş Sayısı",
              "ar": "عدد الأصدقاء",
              "ru": "Количество друзей",
              "it": "Numero di amici",
              "ro": "Numărul de prieteni",
              "es": "Número de amigos",
              "sq": "Numri i miqve"
            }
          }
        }
    """
    result = reader.json("your_friends.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["friends_v2"]  # pyright: ignore
        datapoints.append((len(items)))

        out = pd.DataFrame(datapoints, columns=["Number of friends"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def ads_interests_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract Facebook ad interests.

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
        Columns: ``Ad``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents an interest topic Facebook has associated with the participant for ad targeting purposes.",
          "source_file": "ads_interests.json",
          "columns": {
            "Ad": "Interest topic used for ad targeting."
          }
        }

    Table config::

        {
          "id": "facebook_ads_interests",
          "title": {
            "en": "Your ad interests",
            "nl": "Je advertentie-interesses",
            "de": "Ihre Werbeinteressen",
            "pl": "Twoje zainteresowania reklamowe",
            "tr": "Reklam ilgi alanların",
            "ar": "اهتماماتك الإعلانية",
            "ru": "Ваши рекламные интересы",
            "it": "I tuoi interessi pubblicitari",
            "ro": "Interesele tale de publicitate",
            "es": "Tus intereses publicitarios",
            "sq": "Interesat e tua reklamuese"
          },
          "description": {
            "en": "This table shows the interests Facebook has identified for showing you personalized ads.",
            "nl": "Deze tabel toont de interesses die Facebook heeft geïdentificeerd om je gepersonaliseerde advertenties te tonen.",
            "de": "Diese Tabelle zeigt die Interessen, die Facebook für Sie ermittelt hat, um Ihnen personalisierte Werbung zu zeigen.",
            "pl": "Ta tabela pokazuje zainteresowania, które Facebook zidentyfikował, aby wyświetlać Ci spersonalizowane reklamy.",
            "tr": "Bu tablo, sana kişiselleştirilmiş reklamlar göstermek için Facebook'un belirlediği ilgi alanlarını gösterir.",
            "ar": "يعرض هذا الجدول الاهتمامات التي حددها فيسبوك لعرض إعلانات مخصصة لك.",
            "ru": "В этой таблице показаны интересы, которые Facebook определил для показа вам персонализированной рекламы.",
            "it": "Questa tabella mostra gli interessi che Facebook ha identificato per mostrarti annunci personalizzati.",
            "ro": "Acest tabel arată interesele pe care Facebook le-a identificat pentru a-ți afișa reclame personalizate.",
            "es": "Esta tabla muestra los intereses que Facebook ha identificado para mostrarte anuncios personalizados.",
            "sq": "Kjo tabelë tregon interesat që Facebook ka identifikuar për të të shfaqur reklama të personalizuara."
          },
          "headers": {
            "Ad": {
              "en": "Ad",
              "nl": "Advertentie",
              "de": "Interesse",
              "pl": "Zainteresowanie",
              "tr": "İlgi Alanı",
              "ar": "الاهتمام",
              "ru": "Интерес",
              "it": "Interesse",
              "ro": "Interes",
              "es": "Interés",
              "sq": "Interesi"
            }
          }
        }
    """
    result = reader.json("ads_interests.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["topics_v2"]  # pyright: ignore
        for item in items:
            datapoints.append((
                eh.fix_latin1_string(item),
            ))
        out = pd.DataFrame(datapoints, columns=["Ad"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out

def other_categories_used_to_reach_you_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "ads_information/other_categories_used_to_reach_you.json",
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
        ``"ads_information/other_categories_used_to_reach_you.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Category``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one category that may be used to reach the participant with advertising on Facebook.",
          "source_file": "ads_information/other_categories_used_to_reach_you.json",
          "columns": {
            "Category": "A category associated with the participant that may be used for advertising targeting."
          }
        }

    Table config::

        {
          "id": "facebook_other_categories_used_to_reach_you",
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
            "en": "This table shows categories that Meta may use to determine which ads could be shown to you on Facebook.",
            "nl": "Deze tabel toont categorieën die Meta kan gebruiken om te bepalen welke advertenties aan je kunnen worden getoond op Facebook.",
            "de": "Diese Tabelle zeigt Kategorien, die Meta verwenden kann, um zu bestimmen, welche Werbung Ihnen auf Facebook angezeigt werden könnte.",
            "pl": "Ta tabela pokazuje kategorie, których Meta może używać do określania, jakie reklamy mogą być Ci wyświetlane na Facebooku.",
            "tr": "Bu tablo, Meta'nın Facebook'ta sana hangi reklamların gösterilebileceğini belirlemek için kullanabileceği kategorileri gösterir.",
            "ar": "يعرض هذا الجدول الفئات التي قد تستخدمها Meta لتحديد الإعلانات التي يمكن عرضها لك على فيسبوك.",
            "ru": "В этой таблице показаны категории, которые Meta может использовать для определения рекламы, которая может быть показана вам на Facebook.",
            "it": "Questa tabella mostra le categorie che Meta può utilizzare per determinare quali inserzioni potrebbero esserti mostrate su Facebook.",
            "ro": "Acest tabel arată categoriile pe care Meta le poate utiliza pentru a determina ce reclame ți-ar putea fi afișate pe Facebook.",
            "es": "Esta tabla muestra las categorías que Meta puede utilizar para determinar qué anuncios podrían mostrarse en Facebook.",
            "sq": "Kjo tabelë tregon kategoritë që Meta mund të përdorë për të përcaktuar se cilat reklama mund të të shfaqen në Facebook."
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

def advertisers_using_your_information_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = (
        "ads_information/"
        "advertisers_using_your_activity_or_information.json"
    ),
) -> pd.DataFrame:
    """Extract advertisers using the participant's activity or information.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"ads_information/"
        "advertisers_using_your_activity_or_information.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Advertiser``, ``Information use``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents an advertiser that used the participant's activity or information for advertising.",
          "source_file": "ads_information/advertisers_using_your_activity_or_information.json",
          "columns": {
            "Advertiser": "Name of the advertiser.",
            "Information use": "Meta's description of how the advertiser used information associated with the participant."
          }
        }

    Table config::

        {
          "id": "facebook_advertisers_using_your_information",
          "title": {
            "en": "Advertisers using your information",
            "nl": "Adverteerders die je informatie gebruiken",
            "de": "Werbetreibende, die Ihre Informationen verwenden",
            "pl": "Reklamodawcy korzystający z Twoich informacji",
            "tr": "Bilgilerini kullanan reklamverenler",
            "ar": "المعلنون الذين يستخدمون معلوماتك",
            "ru": "Рекламодатели, использующие вашу информацию",
            "it": "Inserzionisti che utilizzano le tue informazioni",
            "ro": "Agenți de publicitate care utilizează informațiile tale",
            "es": "Anunciantes que utilizan tu información",
            "sq": "Reklamuesit që përdorin informacionin tënd"
          },
          "description": {
            "en": "This table shows advertisers that Meta reports as having used your activity or information, for example through an advertiser list.",
            "nl": "Deze tabel toont adverteerders die volgens Meta je activiteit of informatie hebben gebruikt, bijvoorbeeld via een lijst van een adverteerder.",
            "de": "Diese Tabelle zeigt Werbetreibende, die laut Meta Ihre Aktivitäten oder Informationen verwendet haben, beispielsweise über eine Liste eines Werbetreibenden.",
            "pl": "Ta tabela pokazuje reklamodawców, którzy według Meta korzystali z Twojej aktywności lub informacji, na przykład za pośrednictwem listy reklamodawcy.",
            "tr": "Bu tablo, Meta'ya göre etkinliklerini veya bilgilerini, örneğin bir reklamveren listesi aracılığıyla, kullanan reklamverenleri gösterir.",
            "ar": "يعرض هذا الجدول المعلنين الذين تفيد Meta بأنهم استخدموا نشاطك أو معلوماتك، على سبيل المثال من خلال قائمة خاصة بالمعلن.",
            "ru": "В этой таблице показаны рекламодатели, которые, по данным Meta, использовали ваши действия или информацию, например с помощью списка рекламодателя.",
            "it": "Questa tabella mostra gli inserzionisti che, secondo Meta, hanno utilizzato la tua attività o le tue informazioni, ad esempio tramite un elenco dell'inserzionista.",
            "ro": "Acest tabel arată agenții de publicitate despre care Meta afirmă că au utilizat activitatea sau informațiile tale, de exemplu prin intermediul unei liste a unui agent de publicitate.",
            "es": "Esta tabla muestra los anunciantes que, según Meta, han utilizado tu actividad o información, por ejemplo mediante una lista de anunciantes.",
            "sq": "Kjo tabelë tregon reklamuesit që, sipas Meta, kanë përdorur aktivitetin ose informacionin tënd, për shembull nëpërmjet një liste reklamuesi."
          },
          "headers": {
            "Advertiser": {
              "en": "Advertiser",
              "nl": "Adverteerder",
              "de": "Werbetreibender",
              "pl": "Reklamodawca",
              "tr": "Reklamveren",
              "ar": "المعلن",
              "ru": "Рекламодатель",
              "it": "Inserzionista",
              "ro": "Agent de publicitate",
              "es": "Anunciante",
              "sq": "Reklamuesi"
            },
            "Information use": {
              "en": "How your information was used",
              "nl": "Hoe je informatie is gebruikt",
              "de": "Wie Ihre Informationen verwendet wurden",
              "pl": "Jak wykorzystano Twoje informacje",
              "tr": "Bilgilerinin nasıl kullanıldığı",
              "ar": "كيفية استخدام معلوماتك",
              "ru": "Как использовалась ваша информация",
              "it": "Come sono state utilizzate le tue informazioni",
              "ro": "Cum au fost utilizate informațiile tale",
              "es": "Cómo se utilizó tu información",
              "sq": "Si u përdor informacioni yt"
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
            information_use = eh.fix_latin1_string(
                item.get("label", "")
            )

            for entry in item.get("vec", []):
                advertiser = entry.get("value", "")

                if not advertiser:
                    continue

                datapoints.append((
                    eh.fix_latin1_string(advertiser),
                    information_use,
                ))

        return pd.DataFrame(
            datapoints,
            columns=["Advertiser", "Information use"],
        )

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return pd.DataFrame()
    
def advertisers_interacted_with_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = (
        "ads_information/"
        "advertisers_you've_interacted_with.json"
    ),
) -> pd.DataFrame:
    """Extract ads with which the participant interacted on Facebook.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"ads_information/advertisers_you've_interacted_with.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Action``, ``Ad title``, ``Content URL``, ``Timestamp``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents an advertisement with which the participant interacted on Facebook.",
          "source_file": "ads_information/advertisers_you've_interacted_with.json",
          "columns": {
            "Action": "Type of interaction with the advertisement, such as a click.",
            "Ad title": "Title or text associated with the advertisement.",
            "Content URL": "URL of the associated Facebook post, page, or other content.",
            "Timestamp": "ISO 8601 timestamp of when the interaction occurred."
          }
        }

    Table config::

        {
          "id": "facebook_advertisers_interacted_with",
          "title": {
            "en": "Ads you interacted with",
            "nl": "Advertenties waarmee je interactie had",
            "de": "Werbeanzeigen, mit denen Sie interagiert haben",
            "pl": "Reklamy, z którymi wchodziłeś w interakcję",
            "tr": "Etkileşimde bulunduğun reklamlar",
            "ar": "الإعلانات التي تفاعلت معها",
            "ru": "Реклама, с которой вы взаимодействовали",
            "it": "Inserzioni con cui hai interagito",
            "ro": "Reclame cu care ai interacționat",
            "es": "Anuncios con los que interactuaste",
            "sq": "Reklamat me të cilat ke ndërvepruar"
          },
          "description": {
            "en": "This table shows ads that you interacted with on Facebook, including the type and time of the interaction.",
            "nl": "Deze tabel toont advertenties waarmee je op Facebook interactie had, inclusief het type en tijdstip van de interactie.",
            "de": "Diese Tabelle zeigt Werbeanzeigen, mit denen Sie auf Facebook interagiert haben, einschließlich der Art und des Zeitpunkts der Interaktion.",
            "pl": "Ta tabela pokazuje reklamy, z którymi wchodziłeś w interakcję na Facebooku, w tym rodzaj i czas interakcji.",
            "tr": "Bu tablo, Facebook'ta etkileşimde bulunduğun reklamları, etkileşim türünü ve zamanını gösterir.",
            "ar": "يعرض هذا الجدول الإعلانات التي تفاعلت معها على فيسبوك، بما في ذلك نوع التفاعل ووقته.",
            "ru": "В этой таблице показана реклама, с которой вы взаимодействовали на Facebook, включая тип и время взаимодействия.",
            "it": "Questa tabella mostra le inserzioni con cui hai interagito su Facebook, inclusi il tipo e il momento dell'interazione.",
            "ro": "Acest tabel arată reclamele cu care ai interacționat pe Facebook, inclusiv tipul și momentul interacțiunii.",
            "es": "Esta tabla muestra los anuncios con los que interactuaste en Facebook, incluidos el tipo y el momento de la interacción.",
            "sq": "Kjo tabelë tregon reklamat me të cilat ke ndërvepruar në Facebook, duke përfshirë llojin dhe kohën e ndërveprimit."
          },
          "headers": {
            "Action": {
              "en": "Action",
              "nl": "Actie",
              "de": "Handlung",
              "pl": "Działanie",
              "tr": "Eylem",
              "ar": "الإجراء",
              "ru": "Действие",
              "it": "Azione",
              "ro": "Acțiune",
              "es": "Acción",
              "sq": "Veprimi"
            },
            "Ad title": {
              "en": "Ad title",
              "nl": "Advertentietitel",
              "de": "Titel der Werbeanzeige",
              "pl": "Tytuł reklamy",
              "tr": "Reklam başlığı",
              "ar": "عنوان الإعلان",
              "ru": "Название рекламы",
              "it": "Titolo dell'inserzione",
              "ro": "Titlul reclamei",
              "es": "Título del anuncio",
              "sq": "Titulli i reklamës"
            },
            "Content URL": {
              "en": "Content URL",
              "nl": "URL van de inhoud",
              "de": "URL des Inhalts",
              "pl": "Adres URL treści",
              "tr": "İçerik URL'si",
              "ar": "رابط المحتوى",
              "ru": "URL контента",
              "it": "URL del contenuto",
              "ro": "Adresa URL a conținutului",
              "es": "URL del contenido",
              "sq": "URL-ja e përmbajtjes"
            },
            "Timestamp": {
              "en": "Date",
              "nl": "Datum",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Dată",
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
    datapoints = []

    # Labels inside label_values can depend on the participant's
    # Facebook account language.
    action_labels = {
        "Action",
        "Actie",
        "Handlung",
        "Działanie",
        "Eylem",
        "الإجراء",
        "Действие",
        "Azione",
        "Acțiune",
        "Acción",
        "Veprimi",
    }

    title_labels = {
        "Title",
        "Titel",
        "Tytuł",
        "Başlık",
        "العنوان",
        "Название",
        "Titolo",
        "Titlu",
        "Título",
        "Titulli",
    }

    try:
        for item in cast(list, data):
            action = ""
            ad_title = ""
            content_url = ""

            for entry in item.get("label_values", []):
                label = eh.fix_latin1_string(
                    entry.get("label", "")
                )
                value = eh.fix_latin1_string(
                    entry.get("value", "")
                )
                href = eh.fix_latin1_string(
                    entry.get("href", "")
                )

                # Prefer href when Meta supplies both value and href.
                url = href or value

                if label in action_labels:
                    action = value

                elif label in title_labels:
                    ad_title = value

                # Do not include links to Meta's public Ad Library.
                elif "/ads/library/" in url:
                    continue

                elif url.startswith(("http://", "https://")):
                    content_url = url

            datapoints.append((
                action,
                ad_title,
                content_url,
                eh.epoch_to_iso(
                    item.get("timestamp", ""),
                    errors=errors,
                ),
            ))

        return pd.DataFrame(
            datapoints,
            columns=[
                "Action",
                "Ad title",
                "Content URL",
                "Timestamp",
            ],
        )

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return pd.DataFrame()


def ads_viewed_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = "logged_information/interactions/ads.json",
) -> pd.DataFrame:
    """Extract advertisements viewed by the participant on Facebook.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"logged_information/interactions/ads.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Ad``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one advertisement viewed by the participant on Facebook and when it was viewed.",
          "source_file": "logged_information/interactions/ads.json",
          "columns": {
            "Ad": "Name of the advertiser / ad as recorded by Facebook.",
            "URL": "Link to the ad content (post, reel or page).",
            "Date": "ISO 8601 timestamp of when the advertisement was viewed."
          }
        }

    Table config::

        {
          "id": "facebook_ads_viewed",
          "title": {
            "en": "Ads you viewed",
            "nl": "Advertenties die je hebt bekeken",
            "de": "Werbeanzeigen, die Sie angesehen haben",
            "pl": "Reklamy, które wyświetliłeś/aś",
            "tr": "Görüntülediğin reklamlar",
            "ar": "الإعلانات التي شاهدتها",
            "ru": "Реклама, которую вы просмотрели",
            "it": "Inserzioni che hai visualizzato",
            "ro": "Reclamele pe care le-ai vizualizat",
            "es": "Anuncios que viste",
            "sq": "Reklamat që ke parë"
          },
          "description": {
            "en": "This table shows the ads you viewed on Facebook and when you viewed them.",
            "nl": "Deze tabel toont de advertenties die je op Facebook hebt bekeken en wanneer je ze hebt bekeken.",
            "de": "Diese Tabelle zeigt die Werbeanzeigen, die Sie auf Facebook angesehen haben, und wann Sie sie angesehen haben.",
            "pl": "Ta tabela pokazuje reklamy wyświetlone przez Ciebie na Facebooku oraz czas ich wyświetlenia.",
            "tr": "Bu tablo, Facebook'ta görüntülediğin reklamları ve onları ne zaman görüntülediğini gösterir.",
            "ar": "يعرض هذا الجدول الإعلانات التي شاهدتها على فيسبوك ووقت مشاهدتها.",
            "ru": "В этой таблице показана реклама, которую вы просмотрели на Facebook, и время просмотра.",
            "it": "Questa tabella mostra le inserzioni che hai visualizzato su Facebook e quando le hai visualizzate.",
            "ro": "Acest tabel arată reclamele pe care le-ai vizualizat pe Facebook și când le-ai vizualizat.",
            "es": "Esta tabla muestra los anuncios que viste en Facebook y cuándo los viste.",
            "sq": "Kjo tabelë tregon reklamat që ke parë në Facebook dhe kohën kur i ke parë."
          },
          "headers": {
            "Ad": {
              "en": "Ad",
              "nl": "Advertentie",
              "de": "Werbeanzeige",
              "pl": "Reklama",
              "tr": "Reklam",
              "ar": "الإعلان",
              "ru": "Реклама",
              "it": "Inserzione",
              "ro": "Reclamă",
              "es": "Anuncio",
              "sq": "Reklama"
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
              "nl": "Datum",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Dată",
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
    datapoints = []

    try:
        for item in cast(list, data):
            ad = ""
            url = ""
            timestamp = ""

            for entry in item.get("label_values", []):
                if not isinstance(entry, dict):
                    continue
                if entry.get("timestamp_value") not in (None, ""):
                    if timestamp == "":
                        timestamp = entry.get("timestamp_value", "")
                elif not ad and (entry.get("value") or entry.get("href")):
                    # The value is the advertiser / ad name, not the label
                    # ("Werbeanzeige" / "Ad"), which is the same for every row.
                    ad = eh.fix_latin1_string(entry.get("value", ""))
                    url = entry.get("href", "")

            if not timestamp and item.get("timestamp"):
                timestamp = item.get("timestamp")

            datapoints.append((
                ad,
                url,
                eh.epoch_to_iso(timestamp, errors=errors),
            ))

        return pd.DataFrame(
            datapoints,
            columns=["Ad", "URL", "Date"],
        )

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return pd.DataFrame()


def content_shown_in_feed_to_df(
    reader: ZipArchiveReader,
    errors: Counter,
    *,
    filename: str = (
        "logged_information/interactions/"
        "content_that_has_been_shown_to_you_in_your_feed.json"
    ),
) -> pd.DataFrame:
    """Extract content shown in the participant's Facebook Feed.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.
    filename:
        Path inside the zip archive to read. Defaults to
        ``"logged_information/interactions/"
        "content_that_has_been_shown_to_you_in_your_feed.json"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Description``, ``URL``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents one item that Facebook recorded as having shown in the participant's Feed.",
          "source_file": "logged_information/interactions/content_that_has_been_shown_to_you_in_your_feed.json",
          "columns": {
            "Description": "Description of the item shown in the Feed, as recorded by Facebook.",
            "URL": "URL of the post, video, or link shown in the Feed.",
            "Date": "ISO 8601 timestamp of when the item was shown."
          }
        }

    Table config::

        {
          "id": "facebook_content_shown_in_feed",
          "title": {
            "en": "Content shown in your Facebook Feed",
            "nl": "Inhoud die in je Facebook-overzicht is weergegeven",
            "de": "Inhalte, die Ihnen in Ihrem Facebook-Feed angezeigt wurden",
            "pl": "Treści wyświetlone w Twoim kanale aktualności na Facebooku",
            "tr": "Facebook Akışında sana gösterilen içerikler",
            "ar": "المحتوى الذي ظهر لك في موجز فيسبوك",
            "ru": "Контент, показанный вам в Ленте Facebook",
            "it": "Contenuti mostrati nel tuo feed di Facebook",
            "ro": "Conținut afișat în fluxul tău Facebook",
            "es": "Contenido mostrado en tu feed de Facebook",
            "sq": "Përmbajtjet e shfaqura në furnizimin tënd në Facebook"
          },
          "description": {
            "en": "This table shows posts, videos and links that Facebook recorded as having shown in your Feed, including a description, the link, and when they were shown.",
            "nl": "Deze tabel toont berichten, video's en links die Facebook volgens je gegevens in je overzicht heeft weergegeven, inclusief een beschrijving, de link en wanneer ze zijn weergegeven.",
            "de": "Diese Tabelle zeigt Beiträge, Videos und Links, die Facebook laut Ihren Daten in Ihrem Feed angezeigt hat, einschließlich einer Beschreibung, des Links sowie des Datums und der Uhrzeit der Anzeige.",
            "pl": "Ta tabela pokazuje posty, filmy i linki, które według danych Facebooka zostały wyświetlone w Twoim kanale aktualności, wraz z opisem, linkiem oraz datą i godziną wyświetlenia.",
            "tr": "Bu tablo, Facebook verilerine göre Akışında gösterilen gönderileri, videoları ve bağlantıları; açıklama, bağlantı ve gösterilme zamanı ile birlikte gösterir.",
            "ar": "يعرض هذا الجدول المنشورات ومقاطع الفيديو والروابط التي سجل فيسبوك ظهورها في موجزك، بما في ذلك الوصف والرابط وتاريخ ووقت ظهورها.",
            "ru": "В этой таблице показаны публикации, видео и ссылки, которые, согласно данным Facebook, отображались в вашей Ленте, включая описание, ссылку, дату и время показа.",
            "it": "Questa tabella mostra i post, i video e i link che, secondo i dati di Facebook, sono stati mostrati nel tuo feed, inclusi una descrizione, il link e la data e l'ora di visualizzazione.",
            "ro": "Acest tabel arată postările, videoclipurile și linkurile care, potrivit datelor Facebook, au fost afișate în fluxul tău, inclusiv descrierea, linkul și data și ora afișării.",
            "es": "Esta tabla muestra las publicaciones, los vídeos y los enlaces que, según los datos de Facebook, aparecieron en tu feed, incluida una descripción, el enlace y la fecha y hora en que se mostraron.",
            "sq": "Kjo tabelë tregon postimet, videot dhe lidhjet që, sipas të dhënave të Facebook-ut, janë shfaqur në furnizimin tënd, duke përfshirë përshkrimin, lidhjen dhe datën e orën e shfaqjes."
          },
          "headers": {
            "Description": {
              "en": "Description",
              "nl": "Beschrijving",
              "de": "Beschreibung",
              "pl": "Opis",
              "tr": "Açıklama",
              "ar": "الوصف",
              "ru": "Описание",
              "it": "Descrizione",
              "ro": "Descriere",
              "es": "Descripción",
              "sq": "Përshkrimi"
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
              "nl": "Datum",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Dată",
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
    datapoints = []

    try:
        records = data if isinstance(data, list) else [data]

        for record in records:
            if not isinstance(record, dict):
                continue

            for section in record.get("label_values", []):
                for item in section.get("vec", []):
                    description = ""
                    url = ""
                    timestamp = ""

                    for entry in item.get("dict", []):
                        value = entry.get("value", "")
                        href = entry.get("href", "")

                        if entry.get("timestamp_value") not in (None, ""):
                            timestamp = entry.get("timestamp_value", "")
                        elif href or (
                            isinstance(value, str)
                            and value.startswith(("http://", "https://"))
                        ):
                            url = eh.fix_latin1_string(href or value)
                        elif value and not description:
                            description = eh.fix_latin1_string(value).strip()

                    if description or url or timestamp != "":
                        datapoints.append((
                            description,
                            url,
                            eh.epoch_to_iso(timestamp, errors=errors),
                        ))

        return pd.DataFrame(
            datapoints,
            columns=["Description", "URL", "Date"],
        )

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return pd.DataFrame()


def recently_viewed_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract Facebook items recently viewed.

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
        Columns: ``Category``, ``Name``, ``Link``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a Facebook post, video, or other item the participant recently viewed, including the category, name, link, and date.",
          "source_file": "recently_viewed.json",
          "columns": {
            "Category": "Content category (e.g. Videos, Marketplace).",
            "Name": "Name or title of the viewed item.",
            "Link": "URL of the viewed item.",
            "Date": "ISO 8601 timestamp of when the item was viewed."
          }
        }

    Table config::

        {
          "id": "facebook_recently_viewed",
          "title": {
            "en": "Facebook items you recently viewed",
            "nl": "Facebook items die je recentelijk hebt bekeken",
            "de": "Facebook-Elemente, die Sie kürzlich angesehen haben",
            "pl": "Elementy na Facebooku, które ostatnio wyświetliłeś/aś",
            "tr": "Yakın zamanda görüntülediğin Facebook öğeleri",
            "ar": "عناصر فيسبوك التي شاهدتها مؤخرًا",
            "ru": "Элементы Facebook, которые вы недавно просматривали",
            "it": "Elementi di Facebook visualizzati di recente",
            "ro": "Elemente Facebook pe care le-ai vizualizat recent",
            "es": "Elementos de Facebook que viste recientemente",
            "sq": "Elementet e Facebook-ut që ke parë kohët e fundit"
          },
          "description": {
            "en": "This table shows the Facebook posts, videos, and other items you have recently viewed.",
            "nl": "Deze tabel toont de Facebook-posts, video's en andere items die je recentelijk hebt bekeken.",
            "de": "Diese Tabelle zeigt die Facebook-Beiträge, Videos und anderen Elemente, die Sie kürzlich angesehen haben.",
            "pl": "Ta tabela pokazuje posty, filmy i inne elementy na Facebooku, które ostatnio wyświetliłeś/aś.",
            "tr": "Bu tablo, yakın zamanda görüntülediğin Facebook gönderilerini, videolarını ve diğer öğeleri gösterir.",
            "ar": "يعرض هذا الجدول منشورات فيسبوك ومقاطع الفيديو والعناصر الأخرى التي شاهدتها مؤخرًا.",
            "ru": "В этой таблице показаны публикации, видео и другие элементы Facebook, которые вы недавно просматривали.",
            "it": "Questa tabella mostra i post, i video e altri elementi di Facebook che hai visualizzato di recente.",
            "ro": "Acest tabel arată postările, videoclipurile și alte elemente de Facebook pe care le-ai vizualizat recent.",
            "es": "Esta tabla muestra las publicaciones, videos y otros elementos de Facebook que has visto recientemente.",
            "sq": "Kjo tabelë tregon postimet, videot dhe elementet e tjera të Facebook-ut që ke parë kohët e fundit."
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
            "Name": {
              "en": "Name",
              "nl": "Naam",
              "de": "Name",
              "pl": "Nazwa",
              "tr": "Ad",
              "ar": "الاسم",
              "ru": "Название",
              "it": "Nome",
              "ro": "Nume",
              "es": "Nombre",
              "sq": "Emri"
            },
            "Link": {
              "en": "Link",
              "nl": "Link",
              "de": "Link",
              "pl": "Link",
              "tr": "Bağlantı",
              "ar": "الرابط",
              "ru": "Ссылка",
              "it": "Link",
              "ro": "Link",
              "es": "Enlace",
              "sq": "Lidhja"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum",
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
    result = reader.json("recently_viewed.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["recently_viewed"] # pyright: ignore
        for item in items:

            if "entries" in item:
                for entry in item["entries"]:
                    datapoints.append((
                        eh.fix_latin1_string(item.get("name", "")),
                        eh.fix_latin1_string(entry.get("data", {}).get("name", "")),
                        entry.get("data", {}).get("uri", ""),
                        eh.epoch_to_iso(entry.get("timestamp", ""), errors=errors)
                    ))

            # The nesting goes deeper
            if "children" in item:
                for child in item["children"]:
                    for entry in child["entries"]:
                        datapoints.append((
                            eh.fix_latin1_string(child.get("name", "")),
                            eh.fix_latin1_string(entry.get("data", {}).get("name", "")),
                            entry.get("data", {}).get("uri", ""),
                            eh.epoch_to_iso(entry.get("timestamp", ""), errors=errors)
                        ))

        out = pd.DataFrame(datapoints, columns=["Category", "Name", "Link", "Date"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def recently_visited_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract Facebook profiles recently visited.

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
        Columns: ``Category``, ``Name``, ``Link``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a Facebook profile or page the participant recently visited, including the category, name, link, and date.",
          "source_file": "recently_visited.json",
          "columns": {
            "Category": "Category of the visited item.",
            "Name": "Name or title of the visited profile or page.",
            "Link": "URL of the visited profile or page.",
            "Date": "ISO 8601 timestamp of when the visit occurred."
          }
        }

    Table config::

        {
          "id": "facebook_recently_visited",
          "title": {
            "en": "Profiles you visited recently",
            "nl": "Profielen die je recentelijk hebt bezocht",
            "de": "Profile, die Sie kürzlich besucht haben",
            "pl": "Profile, które ostatnio odwiedziłeś/aś",
            "tr": "Yakın zamanda ziyaret ettiğin profiller",
            "ar": "الملفات الشخصية التي زرتها مؤخرًا",
            "ru": "Профили, которые вы недавно посещали",
            "it": "Profili visitati di recente",
            "ro": "Profiluri pe care le-ai vizitat recent",
            "es": "Perfiles que visitaste recientemente",
            "sq": "Profilet që ke vizituar kohët e fundit"
          },
          "description": {
            "en": "This table lists the Facebook profiles you have visited most recently.",
            "nl": "Deze tabel toont de Facebook-profielen die je recentelijk hebt bezocht.",
            "de": "Diese Tabelle zeigt die Facebook-Profile, die Sie zuletzt besucht haben.",
            "pl": "Ta tabela zawiera listę profili na Facebooku, które ostatnio odwiedziłeś/aś.",
            "tr": "Bu tablo, en son ziyaret ettiğin Facebook profillerini listeler.",
            "ar": "يسرد هذا الجدول ملفات فيسبوك الشخصية التي زرتها مؤخرًا.",
            "ru": "В этой таблице перечислены профили Facebook, которые вы посещали в последнее время.",
            "it": "Questa tabella elenca i profili di Facebook che hai visitato più di recente.",
            "ro": "Acest tabel listează profilurile de Facebook pe care le-ai vizitat cel mai recent.",
            "es": "Esta tabla enumera los perfiles de Facebook que has visitado más recientemente.",
            "sq": "Kjo tabelë liston profilet e Facebook-ut që ke vizituar më së fundmi."
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
            "Name": {
              "en": "Name",
              "nl": "Naam",
              "de": "Name",
              "pl": "Nazwa",
              "tr": "Ad",
              "ar": "الاسم",
              "ru": "Название",
              "it": "Nome",
              "ro": "Nume",
              "es": "Nombre",
              "sq": "Emri"
            },
            "Link": {
              "en": "Link",
              "nl": "Link",
              "de": "Link",
              "pl": "Link",
              "tr": "Bağlantı",
              "ar": "الرابط",
              "ru": "Ссылка",
              "it": "Link",
              "ro": "Link",
              "es": "Enlace",
              "sq": "Lidhja"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum",
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
    result = reader.json("recently_visited.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["visited_things_v2"]  # pyright: ignore
        for item in items:
            if "entries" in item:
                for entry in item["entries"]:
                    datapoints.append((
                        eh.fix_latin1_string(item.get("name", "")),
                        eh.fix_latin1_string(entry.get("data", {}).get("name", "")),
                        entry.get("data", {}).get("uri", ""),
                        eh.epoch_to_iso(entry.get("timestamp", ""), errors=errors)
                    ))

        out = pd.DataFrame(datapoints, columns=["Category", "Name", "Link", "Date"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def pages_youve_liked_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract Facebook pages you have liked.

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
        Columns: ``Name``, ``URL``, ``Timestamp``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a Facebook Page the participant has liked, including the page name, URL, and timestamp.",
          "source_file": "pages_you've_liked.json",
          "columns": {
            "Name": "Name of the liked Facebook Page.",
            "URL": "URL of the liked Facebook Page.",
            "Timestamp": "ISO 8601 timestamp of when the page was liked."
          }
        }

    Table config::

        {
          "id": "facebook_pages_youve_liked",
          "title": {
            "en": "Pages that you have liked",
            "nl": "Pagina's die je leuk vindt",
            "de": "Seiten, die Ihnen gefallen",
            "pl": "Strony, które polubiłeś/aś",
            "tr": "Beğendiğin sayfalar",
            "ar": "الصفحات التي أعجبت بها",
            "ru": "Страницы, которые вам понравились",
            "it": "Pagine a cui hai messo mi piace",
            "ro": "Pagini pe care le-ai apreciat",
            "es": "Páginas que te han gustado",
            "sq": "Faqet që ke pëlqyer"
          },
          "description": {
            "en": "This table contains a history of the Facebook Pages you have liked.",
            "nl": "Deze tabel bevat een overzicht van de Facebookpagina's die je leuk vindt.",
            "de": "Diese Tabelle enthält eine Übersicht der Facebook-Seiten, die Ihnen gefallen.",
            "pl": "Ta tabela zawiera historię stron na Facebooku, które polubiłeś/aś.",
            "tr": "Bu tablo, beğendiğin Facebook Sayfalarının geçmişini içerir.",
            "ar": "يحتوي هذا الجدول على سجل صفحات فيسبوك التي أعجبت بها.",
            "ru": "В этой таблице содержится история страниц Facebook, которые вам понравились.",
            "it": "Questa tabella contiene la cronologia delle Pagine Facebook a cui hai messo mi piace.",
            "ro": "Acest tabel conține istoricul Paginilor de Facebook pe care le-ai apreciat.",
            "es": "Esta tabla contiene un historial de las Páginas de Facebook que te han gustado.",
            "sq": "Kjo tabelë përmban historikun e Faqeve të Facebook-ut që ke pëlqyer."
          },
          "headers": {
            "Name": {
              "en": "Name",
              "nl": "Naam",
              "de": "Name",
              "pl": "Nazwa",
              "tr": "Ad",
              "ar": "الاسم",
              "ru": "Название",
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
              "ar": "الرابط (URL)",
              "ru": "URL-адрес",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
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
            }
          }
        }
    """
    result = reader.json("pages_you've_liked.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["page_likes_v2"]  # pyright: ignore
        for item in items:
            datapoints.append((
                eh.fix_latin1_string(item.get("name", "")),
                item.get("url", ""),
                eh.epoch_to_iso(item.get("timestamp", ""), errors=errors)
            ))

        out = pd.DataFrame(datapoints, columns=["Name", "URL", "Timestamp"]) # pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def your_saved_items_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract your saved items on Facebook.

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
        Columns: ``Title``, ``Timestamp``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a post, video, or other item the participant saved on Facebook, including the title and timestamp.",
          "source_file": "your_saved_items.json",
          "columns": {
            "Title": "Title of the saved item.",
            "Timestamp": "ISO 8601 timestamp of when the item was saved."
          }
        }

    Table config::

        {
          "id": "facebook_your_saved_items",
          "title": {
            "en": "Your saved items",
            "nl": "Je opgeslagen items",
            "de": "Ihre gespeicherten Objekte",
            "pl": "Twoje zapisane elementy",
            "tr": "Kaydettiğin öğeler",
            "ar": "العناصر المحفوظة لديك",
            "ru": "Ваши сохранённые материалы",
            "it": "I tuoi elementi salvati",
            "ro": "Elementele tale salvate",
            "es": "Tus elementos guardados",
            "sq": "Elementet e tua të ruajtura"
          },
          "description": {
            "en": "This table contains the posts, videos, and other content you have saved on Facebook.",
            "nl": "Deze tabel bevat de berichten, video's en andere content die je op Facebook hebt opgeslagen.",
            "de": "Diese Tabelle enthält die Beiträge, Videos und anderen Inhalte, die Sie auf Facebook gespeichert haben.",
            "pl": "Ta tabela zawiera posty, filmy i inne treści, które zapisałeś/aś na Facebooku.",
            "tr": "Bu tablo, Facebook'ta kaydettiğin gönderileri, videoları ve diğer içerikleri içerir.",
            "ar": "يحتوي هذا الجدول على المنشورات ومقاطع الفيديو والمحتويات الأخرى التي حفظتها على فيسبوك.",
            "ru": "В этой таблице содержатся публикации, видео и другой контент, который вы сохранили на Facebook.",
            "it": "Questa tabella contiene i post, i video e altri contenuti che hai salvato su Facebook.",
            "ro": "Acest tabel conține postările, videoclipurile și alt conținut pe care le-ai salvat pe Facebook.",
            "es": "Esta tabla contiene las publicaciones, videos y otro contenido que has guardado en Facebook.",
            "sq": "Kjo tabelë përmban postimet, videot dhe përmbajtjet e tjera që ke ruajtur në Facebook."
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
            }
          }
        }
    """
    result = reader.json("your_saved_items.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["saves_v2"]  # pyright: ignore
        for item in items:
            datapoints.append((
                eh.fix_latin1_string(item.get("title", "")),
                eh.epoch_to_iso(item.get("timestamp", ""), errors=errors)
            ))

        out = pd.DataFrame(datapoints, columns=["Title", "Timestamp"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def _extract_commented_post_author(title: str) -> str:
    """Extract the content author from an English Facebook comment title."""
    title = title.strip()

    own_match = re.fullmatch(
        r"(.+?) commented on (?:his|her|their) own\s+.+",
        title,
        re.IGNORECASE,
    )
    if own_match:
        return own_match.group(1).strip()

    author_match = re.fullmatch(
        r".+? commented on (.+)['’]s?\s+.+",
        title,
        re.IGNORECASE,
    )
    if author_match:
        return author_match.group(1).strip()

    return ""


def comments_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract all comments you made on Facebook.

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
        Columns: ``Author``, ``Comment``, ``Timestamp``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a comment the participant made on Facebook, including the author of the commented content, comment text, and timestamp.",
          "source_file": "comments_and_reactions/comments.json",
          "columns": {
            "Author": "Author of the post or other content the comment was made on, parsed from Facebook's English activity description.",
            "Comment": "Text content of the comment.",
            "Timestamp": "ISO 8601 timestamp of when the comment was made."
          }
        }

    Table config::

        {
          "id": "facebook_comments",
          "title": {
            "en": "Your comments",
            "nl": "Je commentaren",
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
            "en": "This table shows all the comments you have made on Facebook posts and other content.",
            "nl": "Deze tabel toont alle commentaren die je op Facebook-berichten en andere content hebt geplaatst.",
            "de": "Diese Tabelle zeigt alle Kommentare, die Sie zu Facebook-Beiträgen und anderen Inhalten verfasst haben.",
            "pl": "Ta tabela pokazuje wszystkie komentarze, które dodałeś/aś do postów i innych treści na Facebooku.",
            "tr": "Bu tablo, Facebook gönderilerine ve diğer içeriklere yaptığın tüm yorumları gösterir.",
            "ar": "يعرض هذا الجدول جميع التعليقات التي أضفتها على منشورات فيسبوك والمحتويات الأخرى.",
            "ru": "В этой таблице показаны все комментарии, которые вы оставили к публикациям Facebook и другому контенту.",
            "it": "Questa tabella mostra tutti i commenti che hai fatto a post di Facebook e altri contenuti.",
            "ro": "Acest tabel arată toate comentariile pe care le-ai făcut la postările de Facebook și la alt conținut.",
            "es": "Esta tabla muestra todos los comentarios que has hecho en publicaciones de Facebook y otro contenido.",
            "sq": "Kjo tabelë tregon të gjitha komentet që ke bërë në postimet e Facebook-ut dhe përmbajtje të tjera."
          },
          "headers": {
            "Author": {
              "en": "Author of the commented post",
              "nl": "Auteur van het bericht",
              "de": "Autor*in des kommentierten Beitrags",
              "pl": "Autor/ka komentowanego posta",
              "tr": "Yorum yapılan gönderinin yazarı",
              "ar": "مؤلف المنشور المعلّق عليه",
              "ru": "Автор прокомментированной публикации",
              "it": "Autore del post commentato",
              "ro": "Autorul postării comentate",
              "es": "Autor de la publicación comentada",
              "sq": "Autori i postimit të komentuar"
            },
            "Comment": {
              "en": "Comment",
              "nl": "Reactie",
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
            }
          }
        }
    """
    result = reader.json("comments_and_reactions/comments.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["comments_v2"]  # pyright: ignore
        for item in items:
            denested_dict = eh.dict_denester(item)

            datapoints.append((
                _extract_commented_post_author(
                    eh.fix_latin1_string(eh.find_item(denested_dict, "title"))
                ),
                eh.fix_latin1_string(eh.find_item(denested_dict, "comment-comment")),
                eh.epoch_to_iso(eh.find_item(denested_dict, "timestamp"), errors=errors),
            ))

        out = pd.DataFrame(datapoints, columns=["Author", "Comment", "Timestamp"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def likes_and_reactions_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract likes and reactions with titles from Facebook.

    Reads ``likes_and_reactions_x`` numbered files.

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
        Columns: ``Title``, ``Reaction``, ``Timestamp``.
        Empty DataFrame when no matching files are found or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a post the participant liked or reacted to on Facebook, including the post title, reaction type, and timestamp.",
          "source_file": "likes_and_reactions_1.json (and numbered variants)",
          "columns": {
            "Title": "Title of the post that was liked or reacted to.",
            "Reaction": "Type of reaction (e.g. Like, Love, Haha).",
            "Timestamp": "ISO 8601 timestamp of when the reaction was made."
          }
        }

    Table config::

        {
          "id": "facebook_likes_and_reactions",
          "title": {
            "en": "Posts you liked (with title)",
            "nl": "Posts die je leuk vond (met titel)",
            "de": "Beiträge, die Ihnen gefallen haben (mit Titel)",
            "pl": "Posty, które polubiłeś/aś (z tytułem)",
            "tr": "Beğendiğin gönderiler (başlıklı)",
            "ar": "المنشورات التي أعجبت بها (مع العنوان)",
            "ru": "Публикации, которые вам понравились (с названием)",
            "it": "Post a cui hai messo mi piace (con titolo)",
            "ro": "Postări care ți-au plăcut (cu titlu)",
            "es": "Publicaciones que te gustaron (con título)",
            "sq": "Postimet që ke pëlqyer (me titull)"
          },
          "description": {
            "en": "This table shows the titles of posts you liked on Facebook.",
            "nl": "Deze tabel toont de titels van posts die je leuk vond op Facebook.",
            "de": "Diese Tabelle zeigt die Titel der Beiträge, die Ihnen auf Facebook gefallen haben.",
            "pl": "Ta tabela pokazuje tytuły postów, które polubiłeś/aś na Facebooku.",
            "tr": "Bu tablo, Facebook'ta beğendiğin gönderilerin başlıklarını gösterir.",
            "ar": "يعرض هذا الجدول عناوين المنشورات التي أعجبت بها على فيسبوك.",
            "ru": "В этой таблице показаны названия публикаций, которые вам понравились на Facebook.",
            "it": "Questa tabella mostra i titoli dei post a cui hai messo mi piace su Facebook.",
            "ro": "Acest tabel arată titlurile postărilor care ți-au plăcut pe Facebook.",
            "es": "Esta tabla muestra los títulos de las publicaciones que te gustaron en Facebook.",
            "sq": "Kjo tabelë tregon titujt e postimeve që ke pëlqyer në Facebook."
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
            "Reaction": {
              "en": "Reaction",
              "nl": "Reactie",
              "de": "Reaktion",
              "pl": "Reakcja",
              "tr": "Tepki",
              "ar": "التفاعل",
              "ru": "Реакция",
              "it": "Reazione",
              "ro": "Reacție",
              "es": "Reacción",
              "sq": "Reagimi"
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
            }
          }
        }
    """
    out = pd.DataFrame()
    datapoints = []

    results = reader.json_all(r"(^|/)likes_and_reactions_\d+\.json$")
    if not results:
        return pd.DataFrame()

    try:
        for result in results:
            for item in result.data:
                denested_dict = eh.dict_denester(item)

                datapoints.append((
                    eh.fix_latin1_string(eh.find_item(denested_dict, "title")),
                    eh.fix_latin1_string(eh.find_item(denested_dict, "reaction-reaction")),
                    eh.epoch_to_iso(eh.find_item(denested_dict, "timestamp"), errors=errors),
                ))

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
        return pd.DataFrame()

    out = pd.DataFrame(datapoints, columns=["Title", "Reaction", "Timestamp"]) #pyright: ignore

    return out


def your_comment_active_days_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract days you actively commented on Facebook.

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
        Columns: ``Label``, ``Value``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a label-value pair indicating the days on which the participant actively commented on Facebook.",
          "source_file": "your_comment_active_days.json",
          "columns": {
            "Label": "Label describing the activity metric.",
            "Value": "Value associated with the label."
          }
        }

    Table config::

        {
          "id": "facebook_your_comment_active_days",
          "title": {
            "en": "Days you actively commented",
            "nl": "Dagen waarop je actief commentaren hebt geplaatst",
            "de": "Tage, an denen Sie aktiv kommentiert haben",
            "pl": "Dni, w których aktywnie komentowałeś/aś",
            "tr": "Aktif olarak yorum yaptığın günler",
            "ar": "الأيام التي علّقت فيها بنشاط",
            "ru": "Дни, когда вы активно комментировали",
            "it": "Giorni in cui hai commentato attivamente",
            "ro": "Zilele în care ai comentat activ",
            "es": "Días en los que comentaste activamente",
            "sq": "Ditët kur ke komentuar në mënyrë aktive"
          },
          "description": {
            "en": "This table indicates the days on which you made comments on Facebook.",
            "nl": "Deze tabel toont de dagen waarop je commentaren op Facebook hebt geplaatst.",
            "de": "Diese Tabelle zeigt die Tage, an denen Sie Kommentare auf Facebook verfasst haben.",
            "pl": "Ta tabela pokazuje dni, w których dodawałeś/aś komentarze na Facebooku.",
            "tr": "Bu tablo, Facebook'ta yorum yaptığın günleri gösterir.",
            "ar": "يوضح هذا الجدول الأيام التي أضفت فيها تعليقات على فيسبوك.",
            "ru": "В этой таблице показаны дни, в которые вы оставляли комментарии на Facebook.",
            "it": "Questa tabella mostra i giorni in cui hai pubblicato commenti su Facebook.",
            "ro": "Acest tabel arată zilele în care ai făcut comentarii pe Facebook.",
            "es": "Esta tabla muestra los días en los que hiciste comentarios en Facebook.",
            "sq": "Kjo tabelë tregon ditët kur ke bërë komente në Facebook."
          },
          "headers": {
            "Label": {
              "en": "Label",
              "nl": "Label",
              "de": "Bezeichnung",
              "pl": "Etykieta",
              "tr": "Etiket",
              "ar": "التصنيف",
              "ru": "Метка",
              "it": "Etichetta",
              "ro": "Etichetă",
              "es": "Etiqueta",
              "sq": "Etiketa"
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
            }
          }
        }
    """
    result = reader.json("your_comment_active_days.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["label_values"]  # pyright: ignore
        for item in items:
            datapoints.append((
                eh.fix_latin1_string(item.get("label", "")),
                item.get("value", ""),
            ))

        out = pd.DataFrame(datapoints, columns=["Label", "Value"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


def story_reactions_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract your reactions to Facebook Stories.

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
        Columns: ``Title``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a Facebook Story the participant reacted to, identified by its title.",
          "source_file": "story_reactions.json",
          "columns": {
            "Title": "Title of the story that was reacted to."
          }
        }

    Table config::

        {
          "id": "facebook_story_reactions",
          "title": {
            "en": "Your story reactions",
            "nl": "Je story-reacties",
            "de": "Ihre Story-Reaktionen",
            "pl": "Twoje reakcje na relacje",
            "tr": "Hikaye tepkilerin",
            "ar": "تفاعلاتك مع القصص",
            "ru": "Ваши реакции на истории",
            "it": "Le tue reazioni alle storie",
            "ro": "Reacțiile tale la povești",
            "es": "Tus reacciones a historias",
            "sq": "Reagimet e tua ndaj stories"
          },
          "description": {
            "en": "This table contains your reactions to Facebook Stories.",
            "nl": "Deze tabel bevat je reacties op Facebook Stories.",
            "de": "Diese Tabelle enthält Ihre Reaktionen auf Facebook Storys.",
            "pl": "Ta tabela zawiera Twoje reakcje na Relacje na Facebooku.",
            "tr": "Bu tablo, Facebook Hikayelerine verdiğin tepkileri içerir.",
            "ar": "يحتوي هذا الجدول على تفاعلاتك مع قصص فيسبوك.",
            "ru": "В этой таблице содержатся ваши реакции на истории Facebook.",
            "it": "Questa tabella contiene le tue reazioni alle Storie di Facebook.",
            "ro": "Acest tabel conține reacțiile tale la Poveștile de Facebook.",
            "es": "Esta tabla contiene tus reacciones a las Historias de Facebook.",
            "sq": "Kjo tabelë përmban reagimet e tua ndaj Stories në Facebook."
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
            }
          }
        }
    """
    result = reader.json("story_reactions.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        items = d["stories_feedback_v2"]  # pyright: ignore
        for item in items:
            datapoints.append((
                eh.fix_latin1_string(item.get("title", "")),
            ))

        out = pd.DataFrame(datapoints, columns=["Title"]) #pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


# ``label_values`` entries store their ``label`` text in whichever language
# the participant's Facebook account UI is set to (Facebook's export format,
# not a fixed English schema key). The candidate lists below are used to
# match a field regardless of account language. English is the verified
# baseline; the remaining entries are best-effort guesses sourced from
# Facebook's own terminology (see glossary.md) and are UNVERIFIED against
# real non-English exports -- see CHANGES_FACEBOOK.md, "Known limitations".
_REACTION_LABEL_CANDIDATES = ["Reaction", "Reactie", "Reaktion", "Reakcja", "Tepki", "تفاعل", "Реакция", "Reazione", "Reacție", "Reacción", "Reagimi"]
_NAME_LABEL_CANDIDATES = ["Name", "Naam", "Nazwa", "Ad", "الاسم", "Название", "Nome", "Nume", "Nombre", "Emri"]
_URL_LABEL_CANDIDATES = ["URL", "الرابط", "URL-адрес"]


def _lv_get(lv: dict, candidates: list) -> str:
    """Return the first non-empty value found in *lv* for any of *candidates*.

    ``lv`` is built from Facebook's ``label_values`` structure (a list of
    ``{"label": ..., "value": ...}`` dicts turned into a plain dict). The
    ``label`` text is displayed in whatever language the participant's
    Facebook account UI uses, so a single hardcoded English key (as the
    original implementation used) silently returns "" for any non-English
    export. This checks each language-specific candidate in turn.
    """
    for key in candidates:
        value = lv.get(key)
        if value:
            return value
    return ""


def likes_and_reactions_base_to_df(
    reader: ZipArchiveReader,
    errors: Counter
) -> pd.DataFrame:
    """Extract likes and reactions from Facebook.

    Reads ``likes_and_reactions.json`` (no number suffix) and all numbered
    variants ``likes_and_reactions_1.json``, ``_2.json``, etc.  Items with
    ``label_values`` contain Reaction, Name and URL.  Older-format items
    (``title`` / ``data.reaction``) only contribute reaction type and time, and
    are skipped when the same timestamp already appears in a newer-format row.

    Parameters
    ----------
    reader:
        Archive reader used to load JSON files from the DDP zip.
    errors:
        Mutable counter that accumulates error type counts encountered during
        extraction. Updated in-place.

    Returns
    -------
    pd.DataFrame
        Columns: ``Account``, ``Reaction``, ``URL``, ``Timestamp``.
        Empty DataFrame when no matching files are found or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents a like or reaction the participant gave on Facebook, including the account or author of the content, reaction type, URL, and timestamp.",
          "source_file": "likes_and_reactions.json and likes_and_reactions_<n>.json (numbered variants)",
          "columns": {
            "Account": "Name of the account or author whose content the participant reacted to.",
            "Reaction": "Type of reaction (e.g. Like, Love, Haha).",
            "URL": "URL of the content that was reacted to.",
            "Timestamp": "ISO 8601 timestamp of when the reaction was made."
          }
        }

    Table config::

        {
          "id": "facebook_likes_and_reactions_base",
          "title": {
            "en": "Likes and reactions on Facebook",
            "nl": "Likes en reacties op Facebook",
            "de": "Likes und Reaktionen auf Facebook",
            "pl": "Polubienia i reakcje na Facebooku",
            "tr": "Facebook'taki beğeniler ve tepkiler",
            "ar": "الإعجابات والتفاعلات على فيسبوك",
            "ru": "Отметки «Нравится» и реакции на Facebook",
            "it": "Mi piace e reazioni su Facebook",
            "ro": "Aprecieri și reacții pe Facebook",
            "es": "Me gusta y reacciones en Facebook",
            "sq": "Pëlqime dhe reagime në Facebook"
          },
          "description": {
            "en": "This table shows your likes and reactions to posts and other content on Facebook.",
            "nl": "Deze tabel toont je likes en reacties op berichten en andere content op Facebook.",
            "de": "Diese Tabelle zeigt Ihre Likes und Reaktionen auf Beiträge und andere Inhalte auf Facebook.",
            "pl": "Ta tabela pokazuje Twoje polubienia i reakcje na posty i inne treści na Facebooku.",
            "tr": "Bu tablo, Facebook'taki gönderilere ve diğer içeriklere verdiğin beğenileri ve tepkileri gösterir.",
            "ar": "يعرض هذا الجدول إعجاباتك وتفاعلاتك مع المنشورات والمحتويات الأخرى على فيسبوك.",
            "ru": "В этой таблице показаны ваши отметки «Нравится» и реакции на публикации и другой контент на Facebook.",
            "it": "Questa tabella mostra i tuoi mi piace e le tue reazioni a post e altri contenuti su Facebook.",
            "ro": "Acest tabel arată aprecierile și reacțiile tale la postări și alt conținut de pe Facebook.",
            "es": "Esta tabla muestra tus me gusta y reacciones a publicaciones y otro contenido en Facebook.",
            "sq": "Kjo tabelë tregon pëlqimet dhe reagimet e tua ndaj postimeve dhe përmbajtjeve të tjera në Facebook."
          },
          "headers": {
            "Account": {
              "en": "Account / author",
              "nl": "Account / auteur",
              "de": "Autor*in",
              "pl": "Konto / autor",
              "tr": "Hesap / yazar",
              "ar": "الحساب / الكاتب",
              "ru": "Аккаунт / автор",
              "it": "Account / autore",
              "ro": "Cont / autor",
              "es": "Cuenta / autor",
              "sq": "Llogaria / autori"
            },
            "Reaction": {
              "en": "Reaction",
              "nl": "Reactie",
              "de": "Reaktion",
              "pl": "Reakcja",
              "tr": "Tepki",
              "ar": "التفاعل",
              "ru": "Реакция",
              "it": "Reazione",
              "ro": "Reacție",
              "es": "Reacción",
              "sq": "Reagimi"
            },
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط (URL)",
              "ru": "URL-адрес",
              "it": "URL",
              "ro": "URL",
              "es": "URL",
              "sq": "URL"
            },
            "Timestamp": {
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

    datapoints = []
    seen_new: set = set()
    seen_timestamps: set = set()
    old_format_items: list = []

    def _parse_items(d) -> None:
        if not isinstance(d, list):
            return
        for item in d:
            if not isinstance(item, dict):
                continue
            if "label_values" not in item:
                # Older export format: handled after all new-format rows are known.
                old_format_items.append(item)
                continue
            lv = {
                x.get("label", ""): x.get("value", "")
                for x in item.get("label_values", [])
            }
            row = (
                eh.fix_latin1_string(_lv_get(lv, _NAME_LABEL_CANDIDATES)),
                eh.fix_latin1_string(_lv_get(lv, _REACTION_LABEL_CANDIDATES)),
                _lv_get(lv, _URL_LABEL_CANDIDATES),
                eh.epoch_to_iso(item.get("timestamp", ""), errors=errors),
            )
            key = (row[3], row[2], row[1])
            if key in seen_new:
                continue
            seen_new.add(key)
            seen_timestamps.add(row[3])
            datapoints.append(row)

    try:
        # likes_and_reactions.json and the numbered likes_and_reactions_<n>.json
        # files can both be present (and overlap), so always read all of them.
        result = reader.json("likes_and_reactions.json")
        if result.found:
            _parse_items(result.data)

        for r in reader.json_all(r"(^|/)likes_and_reactions_\d+\.json$"):
            _parse_items(r.data)

        # Older format: {"timestamp", "title", "data": [{"reaction": {"reaction", "actor"}}]}.
        # The title is a localized sentence naming the participant, so only the
        # reaction type and time are used; items already seen above are skipped.
        for item in old_format_items:
            timestamp = eh.epoch_to_iso(item.get("timestamp", ""), errors=errors)
            if timestamp in seen_timestamps:
                continue
            reaction = ""
            for entry in item.get("data", []) or []:
                reaction = (entry.get("reaction") or {}).get("reaction", "") or reaction
            seen_timestamps.add(timestamp)
            datapoints.append(("", eh.fix_latin1_string(reaction), "", timestamp))

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    out = (
        pd.DataFrame(
            datapoints,
            columns=["Account", "Reaction", "URL", "Timestamp"],
        )
        if datapoints
        else pd.DataFrame()
    )

    return out


def controls_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract feed controls (show more / show less) from Facebook.

    Reads ``preferences/feed/controls.json``.  The top-level key ``controls``
    is a list of groups (e.g. "Show more", "Show less"), each with an
    ``entries`` list.

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
        Columns: ``Action``, ``Content``, ``Date``.
        Empty DataFrame when the file is absent or parsing fails.

    Table documentation::

        {
          "summary": "Each row represents an action the participant took to customise their Facebook feed (show more or show less of certain content), including the action type, content affected, and date.",
          "source_file": "preferences/feed/controls.json",
          "columns": {
            "Action": "Feed control action taken (e.g. Show more, Show less).",
            "Content": "Content or topic the action was applied to.",
            "Date": "ISO 8601 timestamp of when the action was taken."
          }
        }

    Table config::

        {
          "id": "facebook_feed_controls",
          "title": {
            "en": "Feed controls (show more / show less)",
            "nl": "Feed-voorkeuren (meer zien / minder zien)",
            "de": "Feed-Einstellungen (mehr anzeigen / weniger anzeigen)",
            "pl": "Ustawienia aktualności (pokaż więcej / pokaż mniej)",
            "tr": "Akış kontrolleri (daha fazla göster / daha az göster)",
            "ar": "إعدادات آخر الأخبار (عرض المزيد / عرض أقل)",
            "ru": "Настройки ленты (показывать больше / показывать меньше)",
            "it": "Impostazioni del feed (mostra di più / mostra di meno)",
            "ro": "Setările fluxului (arată mai mult / arată mai puțin)",
            "es": "Controles de la sección de noticias (mostrar más / mostrar menos)",
            "sq": "Kontrollet e feed-it (shfaq më shumë / shfaq më pak)"
          },
          "description": {
            "en": "This table shows the actions you've taken to customise what content you see more or less of on Facebook.",
            "nl": "Deze tabel toont de acties die je hebt ondernomen om aan te passen welke content je meer of minder ziet op Facebook.",
            "de": "Diese Tabelle zeigt die Aktionen, mit denen Sie festgelegt haben, welche Inhalte Sie auf Facebook mehr oder weniger sehen.",
            "pl": "Ta tabela pokazuje działania, które podjąłeś/aś, aby dostosować, jakie treści widzisz częściej lub rzadziej na Facebooku.",
            "tr": "Bu tablo, Facebook'ta hangi içerikleri daha fazla veya daha az gördüğünü özelleştirmek için yaptığın işlemleri gösterir.",
            "ar": "يعرض هذا الجدول الإجراءات التي اتخذتها لتخصيص المحتوى الذي تراه أكثر أو أقل على فيسبوك.",
            "ru": "В этой таблице показаны действия, которые вы предприняли, чтобы настроить, какого контента вы видите больше или меньше на Facebook.",
            "it": "Questa tabella mostra le azioni che hai intrapreso per personalizzare quali contenuti vedi di più o di meno su Facebook.",
            "ro": "Acest tabel arată acțiunile pe care le-ai întreprins pentru a personaliza ce conținut vezi mai mult sau mai puțin pe Facebook.",
            "es": "Esta tabla muestra las acciones que has tomado para personalizar qué contenido ves más o menos en Facebook.",
            "sq": "Kjo tabelë tregon veprimet që ke ndërmarrë për të personalizuar përmbajtjen që sheh më shumë ose më pak në Facebook."
          },
          "headers": {
            "Action": {
              "en": "Action",
              "nl": "Actie",
              "de": "Aktion",
              "pl": "Działanie",
              "tr": "Eylem",
              "ar": "الإجراء",
              "ru": "Действие",
              "it": "Azione",
              "ro": "Acțiune",
              "es": "Acción",
              "sq": "Veprimi"
            },
            "Content": {
              "en": "Content",
              "nl": "Inhoud",
              "de": "Inhalt",
              "pl": "Treść",
              "tr": "İçerik",
              "ar": "المحتوى",
              "ru": "Содержание",
              "it": "Contenuto",
              "ro": "Conținut",
              "es": "Contenido",
              "sq": "Përmbajtja"
            },
            "Date": {
              "en": "Date",
              "nl": "Datum",
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
    result = reader.json("preferences/feed/controls.json")
    if not result.found:
        return pd.DataFrame()
    d = result.data

    out = pd.DataFrame()
    datapoints = []

    try:
        groups = d["controls"]  # pyright: ignore
        for group in groups:
            action = eh.fix_latin1_string(group.get("name", ""))
            for entry in group.get("entries", []):
                denested = eh.dict_denester(entry)
                datapoints.append((
                    action,
                    eh.fix_latin1_string(eh.find_item(denested, "value")),
                    eh.epoch_to_iso(eh.find_item(denested, "timestamp"), errors=errors),
                ))

        out = pd.DataFrame(datapoints, columns=["Action", "Content", "Date"])  # pyright: ignore

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return out


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
) -> list[tuple[str, str, str, str]]:
    """Flatten a ``label_values`` (or plain dict) structure into settings rows.

    Returns ``(category, setting, value, date)`` tuples.

    Parameters
    ----------
    scalars_only:
        Do not descend into nested ``dict`` / ``vec`` values (used for files
        whose nested values are lists of other people).
    nested_only:
        Only keep values found inside a nested ``dict`` / ``vec`` (used to keep
        a city/region/country block but drop a sibling postal code).
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


def _timestamp_events(
    data, category: str, errors: Counter, *, item_timestamp_only: bool = False
) -> list[tuple[str, str, str, str]]:
    """Return ``(category, event, date, detail)`` for every timestamp in a ``label_values`` file.

    Only labels and timestamps are used, never values, so identifiers cannot leak.
    With *item_timestamp_only* nested timestamps (e.g. carrier updates) are ignored.
    """
    events: list[tuple[str, str, str, str]] = []
    items = data if isinstance(data, list) else [data]

    def collect(node, path: list, found: list) -> None:
        if isinstance(node, list):
            for child in node:
                collect(child, path, found)
            return
        if not isinstance(node, dict):
            return
        label = node.get("label") or node.get("title")
        new_path = path + [_clean_text(label)] if label else path
        if node.get("timestamp_value"):
            found.append((new_path, node["timestamp_value"]))
        for key in ("label_values", "vec", "dict"):
            if key in node:
                collect(node[key], new_path, found)

    for item in items:
        if not isinstance(item, dict):
            continue
        found: list = []
        if not item_timestamp_only:
            collect(item.get("label_values", []), [], found)
        if found:
            for path, ts in found:
                events.append((category, _PATH_SEP.join(path) or category, eh.epoch_to_iso(ts, errors=errors), ""))
        elif isinstance(item.get("timestamp"), (int, float)) and item["timestamp"] > 0:
            events.append((category, category, eh.epoch_to_iso(item["timestamp"], errors=errors), ""))
    return events


def _dig(obj, dotted: str):
    for part in dotted.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


#: Plain-dict security files: (file, list key, category, event key or constant, timestamp key, detail key)
_SECURITY_DICT_SOURCES = [
    ("security_and_login_information/account_activity.json", "account_activity_v2", "Account-Aktivität", "action", "timestamp", "site_name"),
    ("security_and_login_information/ip_address_activity.json", "used_ip_address_v2", "IP-Adress-Aktivität", "action", "timestamp", None),
    ("security_and_login_information/logins_and_logouts.json", "account_accesses_v2", "An- und Abmeldungen", "action", "timestamp", "site"),
    ("security_and_login_information/record_details.json", "admin_records_v2", "Protokolldetails", "event", "session.created_timestamp", None),
    ("security_and_login_information/where_you're_logged_in.json", "active_sessions_v2", "Wo Sie angemeldet sind", "=Aktive Sitzung", "created_timestamp", "session_type"),
    ("security_and_login_information/email_address_verifications.json", "contact_verifications_v2", "Kontaktverifizierungen", "=Kontakt verifiziert", "verification_time", None),
]

#: German labels for the English event names and session types Meta uses in the
#: older plain-dict security files (unknown values are kept as exported).
_EVENT_LABELS = {
    "Session updated": "Sitzung aktualisiert",
    "Login": "Login",
    "Login checks failed": "Login-Prüfungen fehlgeschlagen",
    "Nonce Successful Validation": "Erfolgreiche Nonce-Validierung",
    "Requested Password Reset": "Passwort-Zurücksetzung angefordert",
    "Checkpoint": "Checkpoint",
    "Checkpoint completed": "Checkpoint abgeschlossen",
    "Password Change": "Passwortänderung",
}
_SESSION_TYPE_LABELS = {
    "web": "Web",
    "mobile_web": "Mobiles Web",
    "iphone": "iPhone",
    "android": "Android",
    "instagram": "Instagram",
    "messenger": "Messenger",
}

#: ``label_values`` security files: only labels and timestamps are kept.
_SECURITY_LV_SOURCES = [
    ("security_and_login_information/device_login_cookies.json", "Geräte-Login-Cookies", False),
    ("security_and_login_information/information_about_your_last_login.json", "Letzte Anmeldung", False),
    ("security_and_login_information/login_messages_we_have_shown.json", "Angezeigte Login-Hinweise", False),
    ("security_and_login_information/registration_information.json", "Registrierung", False),
    ("security_and_login_information/two-factor_authentication.json", "Zwei-Faktor-Authentifizierung", False),
    ("security_and_login_information/your_profile_confirmation_information.json", "Profilbestätigung", False),
    ("security_and_login_information/your_recent_profile_recovery_successes.json", "Profilwiederherstellung", True),
]

_RE_DMY = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")


def _active_day_to_iso(text: str) -> str:
    m = _RE_DMY.fullmatch(text.strip())
    if not m:
        return text.strip()
    day, month, year = (int(g) for g in m.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def ad_settings_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract ad-related and off-Facebook activity settings from Facebook.

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
          "summary": "Each row is one setting or permission related to advertising on Facebook or to activity of other websites and apps shared with Facebook (ad preferences, apps granted permissions, off-Meta activity settings).",
          "source_file": "ads_information/ad_preferences.json, apps_and_websites_off_of_facebook/permissions_you_have_granted_to_apps.json, apps_and_websites_off_of_facebook/your_activity_off_meta_technologies_settings.json",
          "columns": {
            "Category": "Which Facebook settings file the row comes from.",
            "Setting": "Name of the setting, as displayed in the participant's Facebook language (nested settings are joined with ' › ').",
            "Value": "Value of the setting. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed.",
            "Date": "ISO 8601 timestamp of when the settings record was last updated (empty when not available)."
          }
        }

    Table config::

        {
          "id": "facebook_ad_settings",
          "title": {
            "en": "Ad and off-Facebook activity settings",
            "nl": "Advertentie-instellingen en activiteit buiten Facebook",
            "de": "Werbeeinstellungen und Aktivitäten außerhalb von Facebook",
            "pl": "Ustawienia reklam i aktywności poza Facebookiem",
            "tr": "Reklam ve Facebook dışı etkinlik ayarları",
            "ar": "إعدادات الإعلانات والنشاط خارج فيسبوك",
            "ru": "Настройки рекламы и активности вне Facebook",
            "it": "Impostazioni degli annunci e dell'attività fuori da Facebook",
            "ro": "Setări pentru reclame și activitatea din afara Facebook",
            "es": "Ajustes de anuncios y de la actividad fuera de Facebook",
            "sq": "Cilësimet e reklamave dhe aktivitetit jashtë Facebook"
          },
          "description": {
            "en": "This table shows the settings and permissions related to the ads you see on Facebook and to the activity of other websites and apps that is shared with Facebook.",
            "nl": "Deze tabel toont de instellingen en machtigingen met betrekking tot de advertenties die je op Facebook ziet en de activiteit van andere websites en apps die met Facebook wordt gedeeld.",
            "de": "Diese Tabelle zeigt die Einstellungen und Berechtigungen zu den Werbeanzeigen, die Sie auf Facebook sehen, sowie zu Aktivitäten anderer Websites und Apps, die mit Facebook geteilt werden.",
            "pl": "Ta tabela pokazuje ustawienia i uprawnienia dotyczące reklam wyświetlanych na Facebooku oraz aktywności innych witryn i aplikacji udostępnianej Facebookowi.",
            "tr": "Bu tablo, Facebook'ta gördüğün reklamlarla ve diğer web sitelerinin ve uygulamaların Facebook ile paylaşılan etkinliğiyle ilgili ayarları ve izinleri gösterir.",
            "ar": "يعرض هذا الجدول الإعدادات والأذونات المتعلقة بالإعلانات التي تراها على فيسبوك ونشاط المواقع والتطبيقات الأخرى الذي تتم مشاركته مع فيسبوك.",
            "ru": "В этой таблице показаны настройки и разрешения, связанные с рекламой, которую вы видите на Facebook, и с активностью других сайтов и приложений, передаваемой Facebook.",
            "it": "Questa tabella mostra le impostazioni e le autorizzazioni relative agli annunci che vedi su Facebook e all'attività di altri siti web e app condivisa con Facebook.",
            "ro": "Acest tabel arată setările și permisiunile legate de reclamele pe care le vezi pe Facebook și de activitatea altor site-uri și aplicații partajată cu Facebook.",
            "es": "Esta tabla muestra los ajustes y permisos relacionados con los anuncios que ves en Facebook y con la actividad de otros sitios web y aplicaciones que se comparte con Facebook.",
            "sq": "Kjo tabelë tregon cilësimet dhe lejet që lidhen me reklamat që sheh në Facebook dhe me aktivitetin e faqeve të tjera dhe aplikacioneve që ndahet me Facebook."
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
        ("ads_information/ad_preferences.json", "Werbepräferenzen", {}),
        ("apps_and_websites_off_of_facebook/permissions_you_have_granted_to_apps.json", "Apps mit Berechtigungen", {}),
        ("apps_and_websites_off_of_facebook/your_activity_off_meta_technologies_settings.json", "Einstellungen zu Aktivitäten außerhalb von Meta", {}),
    ])


def preference_settings_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract Facebook preference and notification settings.

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
          "summary": "Each row is one preference or setting from the participant's Facebook preferences (feed, language, login alerts, memorialization, notifications, video, reels, stories, camera roll, device push, autoplay).",
          "source_file": "preferences/feed/reduce.json and preferences/preferences/*.json (language_settings_history, login_alerts_settings, memorialization_settings, notification_settings, preferred_language, reels_preferences, video_settings, your_camera_roll_controls, your_device_push_settings, your_facebook_story_preferences, your_video_autoplay_settings, dark_mode_settings, languages_you_may_know, your_accessibility_settings, your_audio_settings, your_fundraiser_settings, your_story_highlights, your_watch_settings)",
          "columns": {
            "Category": "Which Facebook preferences file the row comes from.",
            "Setting": "Name of the preference, as displayed in the participant's Facebook language (nested settings are joined with ' › ').",
            "Value": "Value of the preference. For grouped settings (e.g. notification channels) several 'name: value' pairs are joined with '; '. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed.",
            "Date": "ISO 8601 timestamp of when the preference record was last updated (empty when not available)."
          }
        }

    Table config::

        {
          "id": "facebook_preference_settings",
          "title": {
            "en": "Preferences and notification settings",
            "nl": "Voorkeuren en meldingsinstellingen",
            "de": "Einstellungen und Benachrichtigungen",
            "pl": "Preferencje i ustawienia powiadomień",
            "tr": "Tercihler ve bildirim ayarları",
            "ar": "التفضيلات وإعدادات الإشعارات",
            "ru": "Предпочтения и настройки уведомлений",
            "it": "Preferenze e impostazioni delle notifiche",
            "ro": "Preferințe și setări de notificare",
            "es": "Preferencias y ajustes de notificaciones",
            "sq": "Preferencat dhe cilësimet e njoftimeve"
          },
          "description": {
            "en": "This table shows your Facebook preferences, such as your feed, language, notification, video, reels and story settings.",
            "nl": "Deze tabel toont je Facebook-voorkeuren, zoals je instellingen voor feed, taal, meldingen, video's, reels en verhalen.",
            "de": "Diese Tabelle zeigt Ihre Facebook-Einstellungen, zum Beispiel für Feed, Sprache, Benachrichtigungen, Videos, Reels und Storys.",
            "pl": "Ta tabela pokazuje Twoje preferencje na Facebooku, takie jak ustawienia aktualności, języka, powiadomień, filmów, rolek i relacji.",
            "tr": "Bu tablo, akış, dil, bildirim, video, reels ve hikaye ayarların gibi Facebook tercihlerini gösterir.",
            "ar": "يعرض هذا الجدول تفضيلاتك على فيسبوك، مثل إعدادات آخر الأخبار واللغة والإشعارات والفيديو وريلز والقصص.",
            "ru": "В этой таблице показаны ваши предпочтения на Facebook, например настройки ленты, языка, уведомлений, видео, Reels и историй.",
            "it": "Questa tabella mostra le tue preferenze su Facebook, come le impostazioni di feed, lingua, notifiche, video, reel e storie.",
            "ro": "Acest tabel arată preferințele tale de pe Facebook, precum setările pentru flux, limbă, notificări, videoclipuri, reels și povești.",
            "es": "Esta tabla muestra tus preferencias de Facebook, como los ajustes de la sección de noticias, idioma, notificaciones, vídeo, reels e historias.",
            "sq": "Kjo tabelë tregon preferencat e tua në Facebook, si cilësimet e feed-it, gjuhës, njoftimeve, videove, reels dhe story."
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
    base = "preferences/preferences/"
    return _settings_df(reader, errors, [
        ("preferences/feed/reduce.json", "Feed: Inhalte reduzieren", {}),
        (base + "language_settings_history.json", "Verlauf der Spracheinstellungen", {}),
        (base + "preferred_language.json", "Bevorzugte Sprache", {}),
        (base + "login_alerts_settings.json", "Login-Benachrichtigungen", {}),
        (base + "memorialization_settings.json", "Gedenkzustand", {}),
        (base + "notification_settings.json", "Benachrichtigungen", {}),
        (base + "reels_preferences.json", "Reels", {}),
        (base + "video_settings.json", "Video", {}),
        (base + "your_video_autoplay_settings.json", "Video-Autoplay", {}),
        (base + "your_camera_roll_controls.json", "Kamerarolle", {}),
        (base + "your_device_push_settings.json", "Push-Einstellungen des Geräts", {}),
        (base + "your_facebook_story_preferences.json", "Storys", {}),
        (base + "dark_mode_settings.json", "Dunkelmodus", {}),
        (base + "languages_you_may_know.json", "Sprachen, die Sie vielleicht sprechen", {}),
        (base + "your_accessibility_settings.json", "Barrierefreiheit", {}),
        (base + "your_audio_settings.json", "Audio", {}),
        (base + "your_fundraiser_settings.json", "Spendenaktionen", {}),
        (base + "your_story_highlights.json", "Story-Highlights", {}),
        (base + "your_watch_settings.json", "Watch", {}),
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
        Columns: ``Category``, ``Event``, ``Date``, ``Detail``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is one security- or login-related event (log-ins, log-outs, session updates, verifications). Only the type of event, the time and (where available) the site or session type are kept; IP addresses, user agents, devices, cookies, contact details and carrier information are never extracted.",
          "source_file": "security_and_login_information/*.json (account_activity, browser_cookies, device_login_cookies, email_address_verifications, information_about_your_last_login, ip_address_activity, login_messages_we_have_shown, logins_and_logouts, record_details, registration_information, two-factor_authentication, where_you're_logged_in, your_profile_confirmation_information, your_recent_profile_recovery_successes)",
          "columns": {
            "Category": "Which security file the event comes from.",
            "Event": "Type of event (e.g. log-in, session updated), as displayed in the export.",
            "Date": "ISO 8601 timestamp of the event.",
            "Detail": "Website or session type the event relates to (empty when not available)."
          }
        }

    Table config::

        {
          "id": "facebook_security_and_login_events",
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
            "en": "This table shows when you logged in or out of Facebook and other security-related events. Only the type of event and the time are included. IP addresses, devices, cookies and contact details are not included.",
            "nl": "Deze tabel toont wanneer je bent in- of uitgelogd bij Facebook en andere beveiligingsgebeurtenissen. Alleen het type gebeurtenis en het tijdstip zijn opgenomen. IP-adressen, apparaten, cookies en contactgegevens zijn niet opgenomen.",
            "de": "Diese Tabelle zeigt, wann Sie sich bei Facebook an- oder abgemeldet haben, sowie weitere sicherheitsrelevante Ereignisse. Es sind nur die Art des Ereignisses und der Zeitpunkt enthalten. IP-Adressen, Geräte, Cookies und Kontaktdaten sind nicht enthalten.",
            "pl": "Ta tabela pokazuje, kiedy logowałeś/aś się i wylogowywałeś/aś z Facebooka, oraz inne zdarzenia związane z bezpieczeństwem. Uwzględniono tylko rodzaj zdarzenia i czas. Adresy IP, urządzenia, pliki cookie i dane kontaktowe nie są uwzględnione.",
            "tr": "Bu tablo, Facebook'a ne zaman giriş yaptığını veya çıkış yaptığını ve diğer güvenlikle ilgili olayları gösterir. Yalnızca olay türü ve zaman dahildir. IP adresleri, cihazlar, çerezler ve iletişim bilgileri dahil değildir.",
            "ar": "يعرض هذا الجدول متى سجّلت الدخول إلى فيسبوك أو خرجت منه، وأحداث الأمان الأخرى. يتضمن نوع الحدث ووقته فقط. لا تتضمن البيانات عناوين IP والأجهزة وملفات تعريف الارتباط وبيانات الاتصال.",
            "ru": "В этой таблице показано, когда вы входили в Facebook и выходили из него, а также другие события безопасности. Включены только тип события и время. IP-адреса, устройства, файлы cookie и контактные данные не включены.",
            "it": "Questa tabella mostra quando hai effettuato l'accesso o sei uscito da Facebook e altri eventi legati alla sicurezza. Sono inclusi solo il tipo di evento e l'orario. Indirizzi IP, dispositivi, cookie e dati di contatto non sono inclusi.",
            "ro": "Acest tabel arată când te-ai conectat sau te-ai deconectat de la Facebook și alte evenimente legate de securitate. Sunt incluse doar tipul evenimentului și ora. Adresele IP, dispozitivele, cookie-urile și datele de contact nu sunt incluse.",
            "es": "Esta tabla muestra cuándo iniciaste o cerraste sesión en Facebook y otros eventos relacionados con la seguridad. Solo se incluyen el tipo de evento y la hora. No se incluyen direcciones IP, dispositivos, cookies ni datos de contacto.",
            "sq": "Kjo tabelë tregon kur ke hyrë ose dalë nga Facebook dhe ngjarje të tjera që lidhen me sigurinë. Përfshihen vetëm lloji i ngjarjes dhe koha. Adresat IP, pajisjet, cookies dhe të dhënat e kontaktit nuk përfshihen."
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
            },
            "Detail": {
              "en": "Detail",
              "nl": "Detail",
              "de": "Detail",
              "pl": "Szczegóły",
              "tr": "Ayrıntı",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettaglio",
              "ro": "Detaliu",
              "es": "Detalle",
              "sq": "Detaje"
            }
          }
        }
    """
    rows: list[tuple[str, str, str, str]] = []

    try:
        # Plain-dict files: pick event, timestamp and (optionally) site only.
        for path, key, category, event_key, ts_key, detail_key in _SECURITY_DICT_SOURCES:
            data = _read_json_data(reader, errors, path)
            if not isinstance(data, dict):
                continue
            for entry in data.get(key, []):
                if not isinstance(entry, dict):
                    continue
                if event_key.startswith("="):
                    event = event_key[1:]
                else:
                    event = _clean_text(entry.get(event_key, ""))
                    event = _EVENT_LABELS.get(event, event)
                ts = _dig(entry, ts_key)
                if not ts:
                    continue
                detail = _clean_text(entry.get(detail_key, "")) if detail_key else ""
                if detail_key == "session_type":
                    detail = _SESSION_TYPE_LABELS.get(detail, detail)
                if _is_sensitive_value(detail):
                    detail = ""
                rows.append((category, event, eh.epoch_to_iso(ts, errors=errors), detail))

        # Browser cookies: only the times the cookies were used, never the cookie ids.
        data = _read_json_data(reader, errors, "security_and_login_information/browser_cookies.json")
        if isinstance(data, dict):
            for timestamps in (data.get("datr_stats_v2") or {}).values():
                for ts in timestamps:
                    rows.append(("Browser Cookies", "Browser Cookie genutzt", eh.epoch_to_iso(ts, errors=errors), ""))

        # label_values files: labels + timestamps only, values are never read.
        for path, category, item_only in _SECURITY_LV_SOURCES:
            data = _read_json_data(reader, errors, path)
            if data is not None:
                rows.extend(_timestamp_events(data, category, errors, item_timestamp_only=item_only))

    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["Category", "Event", "Date", "Detail"])


def privacy_settings_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the Facebook privacy settings.

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
          "summary": "Each row is one Facebook privacy setting (who can see posts, find the participant or send friend requests, etc.).",
          "source_file": "preferences/preferences/privacy_settings.json",
          "columns": {
            "Category": "Which Facebook file the row comes from.",
            "Setting": "Name of the setting, as displayed in the participant's Facebook language (nested settings are joined with ' › ').",
            "Value": "Value of the setting. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed.",
            "Date": "ISO 8601 timestamp of when the record was last updated (empty when not available)."
          }
        }

    Table config::

        {
          "id": "facebook_privacy_settings",
          "title": {
            "en": "Privacy settings",
            "nl": "Privacy-instellingen",
            "de": "Datenschutzeinstellungen",
            "pl": "Ustawienia prywatności",
            "tr": "Gizlilik ayarları",
            "ar": "إعدادات الخصوصية",
            "ru": "Настройки конфиденциальности",
            "it": "Impostazioni sulla privacy",
            "ro": "Setări de confidențialitate",
            "es": "Ajustes de privacidad",
            "sq": "Cilësimet e privatësisë"
          },
          "description": {
            "en": "This table shows your Facebook privacy settings, for example who can see your posts, find you or send you friend requests.",
            "nl": "Deze tabel toont je privacy-instellingen op Facebook, bijvoorbeeld wie je berichten kan zien, je kan vinden of je een vriendschapsverzoek kan sturen.",
            "de": "Diese Tabelle zeigt Ihre Datenschutzeinstellungen auf Facebook, zum Beispiel wer Ihre Beiträge sehen, Sie finden oder Ihnen Freundschaftsanfragen senden kann.",
            "pl": "Ta tabela pokazuje Twoje ustawienia prywatności na Facebooku, na przykład kto może widzieć Twoje posty, znajdować Cię lub wysyłać Ci zaproszenia do grona znajomych.",
            "tr": "Bu tablo, gönderilerini kimin görebileceği, seni kimin bulabileceği veya sana kimin arkadaşlık isteği gönderebileceği gibi Facebook gizlilik ayarlarını gösterir.",
            "ar": "يعرض هذا الجدول إعدادات الخصوصية على فيسبوك، مثل من يمكنه رؤية منشوراتك أو العثور عليك أو إرسال طلبات صداقة إليك.",
            "ru": "В этой таблице показаны ваши настройки конфиденциальности на Facebook, например кто может видеть ваши публикации, находить вас или отправлять вам запросы в друзья.",
            "it": "Questa tabella mostra le tue impostazioni sulla privacy su Facebook, ad esempio chi può vedere i tuoi post, trovarti o inviarti richieste di amicizia.",
            "ro": "Acest tabel arată setările tale de confidențialitate de pe Facebook, de exemplu cine îți poate vedea postările, te poate găsi sau îți poate trimite cereri de prietenie.",
            "es": "Esta tabla muestra tus ajustes de privacidad de Facebook, por ejemplo quién puede ver tus publicaciones, encontrarte o enviarte solicitudes de amistad.",
            "sq": "Kjo tabelë tregon cilësimet e tua të privatësisë në Facebook, për shembull kush mund t'i shohë postimet e tua, të të gjejë ose të të dërgojë kërkesa miqësie."
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
        ("preferences/preferences/privacy_settings.json", "Datenschutzeinstellungen", {}),
    ])


def consents_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the consents the participant gave to Facebook.

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
          "summary": "Each row is one consent record the participant gave to Facebook (terms of use, processing of sensitive data, ad partners, ...), with its status and date where available.",
          "source_file": "logged_information/other_logged_information/consents.json",
          "columns": {
            "Category": "Which Facebook file the row comes from.",
            "Setting": "Name of the setting, as displayed in the participant's Facebook language (nested settings are joined with ' › ').",
            "Value": "Value of the setting. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed.",
            "Date": "ISO 8601 timestamp of when the record was last updated (empty when not available)."
          }
        }

    Table config::

        {
          "id": "facebook_consents",
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
            "en": "This table shows which consents (for example to the terms of use or to the processing of sensitive data) you gave to Facebook and when.",
            "nl": "Deze tabel toont welke toestemmingen (bijvoorbeeld voor de gebruiksvoorwaarden of de verwerking van gevoelige gegevens) je aan Facebook hebt gegeven en wanneer.",
            "de": "Diese Tabelle zeigt, welche Einwilligungen (zum Beispiel zu den Nutzungsbedingungen oder zur Verarbeitung sensibler Daten) Sie Facebook erteilt haben und wann.",
            "pl": "Ta tabela pokazuje, jakich zgód (na przykład na regulamin lub przetwarzanie danych wrażliwych) udzieliłeś/aś Facebookowi i kiedy.",
            "tr": "Bu tablo, Facebook'a hangi onayları (örneğin kullanım koşulları veya hassas verilerin işlenmesi için) ne zaman verdiğini gösterir.",
            "ar": "يعرض هذا الجدول الموافقات التي منحتها لفيسبوك (مثل شروط الاستخدام أو معالجة البيانات الحساسة) ومتى منحتها.",
            "ru": "В этой таблице показано, какие согласия (например, на условия использования или обработку конфиденциальных данных) вы дали Facebook и когда.",
            "it": "Questa tabella mostra quali consensi (ad esempio ai termini d'uso o al trattamento di dati sensibili) hai dato a Facebook e quando.",
            "ro": "Acest tabel arată ce consimțăminte (de exemplu pentru termenii de utilizare sau prelucrarea datelor sensibile) ai dat Facebook și când.",
            "es": "Esta tabla muestra qué consentimientos (por ejemplo, a las condiciones de uso o al tratamiento de datos sensibles) diste a Facebook y cuándo.",
            "sq": "Kjo tabelë tregon cilat pëlqime (për shembull për kushtet e përdorimit ose përpunimin e të dhënave të ndjeshme) i ke dhënë Facebook dhe kur."
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
        ("logged_information/other_logged_information/consents.json", "Einwilligungen", {}),
    ])


def location_and_time_zone_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract location settings, time zone, privacy jurisdiction and coarse location.

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
        Columns: ``Category``, ``Setting``, ``Value``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is a location-related setting or record: location services setting, time zone, privacy jurisdiction, and the primary / primary public location (city, region, country). Postal codes are not extracted.",
          "source_file": "logged_information/location/location_services_setting.json, timezone.json, your_privacy_jurisdiction.json, primary_location.json, primary_public_location.json",
          "columns": {
            "Category": "Which Facebook file the row comes from.",
            "Setting": "Name of the setting, as displayed in the participant's Facebook language (nested settings are joined with ' › ').",
            "Value": "Value of the setting (city, region, country, time zone, ...). Postal codes are not extracted."
          }
        }

    Table config::

        {
          "id": "facebook_location_and_time_zone",
          "title": {
            "en": "Location and time zone",
            "nl": "Locatie en tijdzone",
            "de": "Standort und Zeitzone",
            "pl": "Lokalizacja i strefa czasowa",
            "tr": "Konum ve saat dilimi",
            "ar": "الموقع والمنطقة الزمنية",
            "ru": "Местоположение и часовой пояс",
            "it": "Posizione e fuso orario",
            "ro": "Locație și fus orar",
            "es": "Ubicación y zona horaria",
            "sq": "Vendndodhja dhe zona kohore"
          },
          "description": {
            "en": "This table shows your location settings, your time zone, your privacy jurisdiction and the general area (city, region, country) Facebook has on record for you. Your postal code is not included.",
            "nl": "Deze tabel toont je locatie-instellingen, je tijdzone, je privacyrechtsgebied en het algemene gebied (stad, regio, land) dat Facebook van je heeft vastgelegd. Je postcode is niet opgenomen.",
            "de": "Diese Tabelle zeigt Ihre Standorteinstellungen, Ihre Zeitzone, Ihre Datenschutz-Zuständigkeit und das allgemeine Gebiet (Stadt, Region, Land), das Facebook für Sie gespeichert hat. Ihre Postleitzahl ist nicht enthalten.",
            "pl": "Ta tabela pokazuje Twoje ustawienia lokalizacji, strefę czasową, jurysdykcję w zakresie prywatności oraz ogólny obszar (miasto, region, kraj) zapisany przez Facebooka. Kod pocztowy nie jest uwzględniony.",
            "tr": "Bu tablo, konum ayarlarını, saat dilimini, gizlilik yargı alanını ve Facebook'un senin için kaydettiği genel bölgeyi (şehir, bölge, ülke) gösterir. Posta kodun dahil değildir.",
            "ar": "يعرض هذا الجدول إعدادات الموقع ومنطقتك الزمنية واختصاص الخصوصية الخاص بك والمنطقة العامة (المدينة والمنطقة والبلد) المسجلة لدى فيسبوك. لا يتضمن الرمز البريدي.",
            "ru": "В этой таблице показаны ваши настройки местоположения, часовой пояс, юрисдикция по защите данных и общая местность (город, регион, страна), записанная Facebook. Почтовый индекс не включён.",
            "it": "Questa tabella mostra le tue impostazioni di posizione, il fuso orario, la giurisdizione sulla privacy e l'area generale (città, regione, paese) registrata da Facebook. Il CAP non è incluso.",
            "ro": "Acest tabel arată setările tale de locație, fusul orar, jurisdicția de confidențialitate și zona generală (oraș, regiune, țară) înregistrată de Facebook. Codul poștal nu este inclus.",
            "es": "Esta tabla muestra tus ajustes de ubicación, tu zona horaria, tu jurisdicción de privacidad y el área general (ciudad, región, país) que Facebook tiene registrada. No se incluye tu código postal.",
            "sq": "Kjo tabelë tregon cilësimet e vendndodhjes, zonën tënde kohore, juridiksionin e privatësisë dhe zonën e përgjithshme (qyteti, rajoni, shteti) që Facebook ka regjistruar për ty. Kodi postar nuk përfshihet."
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
            }
          }
        }
    """
    loc = "logged_information/location/"
    df = _settings_df(reader, errors, [
        (loc + "location_services_setting.json", "Standortdienste", {}),
        (loc + "timezone.json", "Zeitzone", {}),
        (loc + "your_privacy_jurisdiction.json", "Datenschutz-Zuständigkeit", {}),
        # City / region / country only: the sibling postal code is dropped.
        (loc + "primary_location.json", "Hauptstandort", {"nested_only": True}),
        (loc + "primary_public_location.json", "Öffentlicher Hauptstandort", {"nested_only": True}),
    ])
    # None of these records carry a meaningful timestamp, so the Date column is dropped.
    return df.drop(columns=["Date"]) if not df.empty else df


def active_days_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the days on which the participant was active on Facebook (one row per day).

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
        Columns: ``Date``, ``Platforms``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is a day on which the participant was active on Facebook, with the platform(s) used that day (e.g. website, app). One row per day.",
          "source_file": "security_and_login_information/your_facebook_activity_history.json",
          "columns": {
            "Date": "Day of activity as ISO 8601 date (YYYY-MM-DD); left as written in the export when the date format is not recognised.",
            "Platforms": "Platform(s) the participant was active on that day (e.g. website or Facebook app), as displayed in the export, separated by commas."
          }
        }

    Table config::

        {
          "id": "facebook_active_days",
          "title": {
            "en": "Days you were active on Facebook",
            "nl": "Dagen waarop je actief was op Facebook",
            "de": "An welchen Tagen Sie auf Facebook aktiv waren",
            "pl": "Dni, w których byłeś/aś aktywny/a na Facebooku",
            "tr": "Facebook'ta aktif olduğun günler",
            "ar": "الأيام التي كنت نشطًا فيها على فيسبوك",
            "ru": "Дни, когда вы были активны на Facebook",
            "it": "Giorni in cui sei stato attivo su Facebook",
            "ro": "Zilele în care ai fost activ pe Facebook",
            "es": "Días en los que estuviste activo en Facebook",
            "sq": "Ditët kur ke qenë aktiv në Facebook"
          },
          "description": {
            "en": "This table lists the days on which you were active on Facebook and whether that was on the website or in the app.",
            "nl": "Deze tabel toont de dagen waarop je actief was op Facebook en of dat op de website of in de app was.",
            "de": "Diese Tabelle zeigt die Tage, an denen Sie auf Facebook aktiv waren, und ob das auf der Website oder in der App war.",
            "pl": "Ta tabela pokazuje dni, w których byłeś/aś aktywny/a na Facebooku, oraz czy było to w witrynie, czy w aplikacji.",
            "tr": "Bu tablo, Facebook'ta hangi günlerde aktif olduğunu ve bunun web sitesinde mi yoksa uygulamada mı olduğunu gösterir.",
            "ar": "يعرض هذا الجدول الأيام التي كنت نشطًا فيها على فيسبوك وما إذا كان ذلك على الموقع الإلكتروني أو في التطبيق.",
            "ru": "В этой таблице показаны дни, когда вы были активны на Facebook, и то, на веб-сайте или в приложении.",
            "it": "Questa tabella elenca i giorni in cui sei stato attivo su Facebook e se è avvenuto sul sito web o nell'app.",
            "ro": "Acest tabel arată zilele în care ai fost activ pe Facebook și dacă a fost pe site sau în aplicație.",
            "es": "Esta tabla muestra los días en los que estuviste activo en Facebook y si fue en el sitio web o en la aplicación.",
            "sq": "Kjo tabelë tregon ditët kur ke qenë aktiv në Facebook dhe nëse ishte në faqen e internetit apo në aplikacion."
          },
          "headers": {
            "Date": {
              "en": "Date",
              "nl": "Datum",
              "de": "Zeitstempel",
              "pl": "Data",
              "tr": "Tarih",
              "ar": "التاريخ",
              "ru": "Дата",
              "it": "Data",
              "ro": "Data",
              "es": "Fecha",
              "sq": "Data"
            },
            "Platforms": {
              "en": "Platforms",
              "nl": "Platforms",
              "de": "Plattformen",
              "pl": "Platformy",
              "tr": "Platformlar",
              "ar": "المنصات",
              "ru": "Платформы",
              "it": "Piattaforme",
              "ro": "Platforme",
              "es": "Plataformas",
              "sq": "Platformat"
            }
          },
          "visualizations": [
            {
              "title": {
                "en": "Active days per month",
                "nl": "Actieve dagen per maand",
                "de": "Aktive Tage pro Monat",
                "pl": "Dni aktywności w miesiącu",
                "tr": "Aylara göre aktif günler",
                "ar": "الأيام النشطة لكل شهر",
                "ru": "Активные дни по месяцам",
                "it": "Giorni attivi per mese",
                "ro": "Zile active pe lună",
                "es": "Días activos por mes",
                "sq": "Ditët aktive për muaj"
              },
              "type": "bar",
              "group": {
                "column": "Date",
                "dateFormat": "month",
                "label": {
                  "en": "Month",
                  "nl": "Maand",
                  "de": "Monat",
                  "pl": "Miesiąc",
                  "tr": "Ay",
                  "ar": "الشهر",
                  "ru": "Месяц",
                  "it": "Mese",
                  "ro": "Lună",
                  "es": "Mes",
                  "sq": "Muaji"
                }
              },
              "values": [
                {
                  "label": {
                    "en": "Active days",
                    "nl": "Actieve dagen",
                    "de": "Aktive Tage",
                    "pl": "Dni aktywności",
                    "tr": "Aktif günler",
                    "ar": "الأيام النشطة",
                    "ru": "Активные дни",
                    "it": "Giorni attivi",
                    "ro": "Zile active",
                    "es": "Días activos",
                    "sq": "Ditë aktive"
                  },
                  "aggregate": "count"
                }
              ]
            },
            {
              "title": {
                "en": "Active days by weekday",
                "nl": "Actieve dagen per weekdag",
                "de": "Aktive Tage nach Wochentag",
                "pl": "Dni aktywności według dnia tygodnia",
                "tr": "Haftanın gününe göre aktif günler",
                "ar": "الأيام النشطة حسب يوم الأسبوع",
                "ru": "Активные дни по дням недели",
                "it": "Giorni attivi per giorno della settimana",
                "ro": "Zile active pe zilele săptămânii",
                "es": "Días activos por día de la semana",
                "sq": "Ditët aktive sipas ditës së javës"
              },
              "type": "bar",
              "group": {
                "column": "Date",
                "dateFormat": "weekday_cycle",
                "label": {
                  "en": "Weekday",
                  "nl": "Weekdag",
                  "de": "Wochentag",
                  "pl": "Dzień tygodnia",
                  "tr": "Haftanın günü",
                  "ar": "يوم الأسبوع",
                  "ru": "День недели",
                  "it": "Giorno della settimana",
                  "ro": "Ziua săptămânii",
                  "es": "Día de la semana",
                  "sq": "Dita e javës"
                }
              },
              "values": [
                {
                  "label": {
                    "en": "Active days",
                    "nl": "Actieve dagen",
                    "de": "Aktive Tage",
                    "pl": "Dni aktywności",
                    "tr": "Aktif günler",
                    "ar": "الأيام النشطة",
                    "ru": "Активные дни",
                    "it": "Giorni attivi",
                    "ro": "Zile active",
                    "es": "Días activos",
                    "sq": "Ditë aktive"
                  },
                  "aggregate": "count"
                }
              ]
            }
          ]
        }
    """
    data = _read_json_data(reader, errors, "security_and_login_information/your_facebook_activity_history.json")
    platforms_by_day: dict[str, set] = {}

    try:
        for item in data if isinstance(data, list) else []:
            platform, days = "", []
            for lv in item.get("label_values", []):
                if "vec" in lv:
                    days = [v.get("value", "") for v in lv["vec"] if isinstance(v, dict)]
                elif not platform and lv.get("value"):
                    platform = _clean_text(lv["value"])
            for day in days:
                platforms_by_day.setdefault(_active_day_to_iso(_clean_text(day)), set()).add(platform or "Facebook")
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    if not platforms_by_day:
        return pd.DataFrame()
    rows = [(day, ", ".join(sorted(p))) for day, p in platforms_by_day.items()]
    return pd.DataFrame(rows, columns=["Date", "Platforms"]).sort_values("Date", ascending=False).reset_index(drop=True)


def profile_visits_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the profiles and pages the participant visited.

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
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is a Facebook profile or page the participant visited.",
          "source_file": "logged_information/interactions/profile_visits.json",
          "columns": {
            "Name": "Name of the visited profile or page.",
            "Date": "ISO 8601 timestamp of the visit."
          }
        }

    Table config::

        {
          "id": "facebook_profile_visits",
          "title": {
            "en": "Profiles you visited",
            "nl": "Profielen die je hebt bezocht",
            "de": "Von Ihnen besuchte Profile",
            "pl": "Odwiedzone przez Ciebie profile",
            "tr": "Ziyaret ettiğin profiller",
            "ar": "الملفات الشخصية التي زرتها",
            "ru": "Посещённые вами профили",
            "it": "Profili che hai visitato",
            "ro": "Profiluri pe care le-ai vizitat",
            "es": "Perfiles que visitaste",
            "sq": "Profilet që ke vizituar"
          },
          "description": {
            "en": "This table shows the Facebook profiles and pages you visited and when.",
            "nl": "Deze tabel toont de Facebook-profielen en -pagina's die je hebt bezocht en wanneer.",
            "de": "Diese Tabelle zeigt die Facebook-Profile und -Seiten, die Sie besucht haben, und wann.",
            "pl": "Ta tabela pokazuje profile i strony na Facebooku, które odwiedziłeś/aś, oraz kiedy.",
            "tr": "Bu tablo, ziyaret ettiğin Facebook profillerini ve sayfalarını ve ne zaman ziyaret ettiğini gösterir.",
            "ar": "يعرض هذا الجدول ملفات فيسبوك وصفحاته التي زرتها ومتى زرتها.",
            "ru": "В этой таблице показаны профили и страницы Facebook, которые вы посещали, и когда.",
            "it": "Questa tabella mostra i profili e le pagine di Facebook che hai visitato e quando.",
            "ro": "Acest tabel arată profilurile și paginile de Facebook pe care le-ai vizitat și când.",
            "es": "Esta tabla muestra los perfiles y páginas de Facebook que visitaste y cuándo.",
            "sq": "Kjo tabelë tregon profilet dhe faqet e Facebook që ke vizituar dhe kur."
          },
          "headers": {
            "Name": {
              "en": "Name",
              "nl": "Naam",
              "de": "Name",
              "pl": "Nazwa",
              "tr": "Ad",
              "ar": "الاسم",
              "ru": "Название",
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
    data = _read_json_data(reader, errors, "logged_information/interactions/profile_visits.json")
    rows: list[tuple[str, str]] = []

    try:
        for item in data if isinstance(data, list) else []:
            name = next((_clean_text(lv["value"]) for lv in item.get("label_values", [])
                         if isinstance(lv, dict) and lv.get("value")), "")
            ts = item.get("timestamp")
            rows.append((name, eh.epoch_to_iso(ts, errors=errors) if ts else ""))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return pd.DataFrame(rows, columns=["Name", "Date"]) if rows else pd.DataFrame()


def link_history_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the links the participant opened through Facebook.

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
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is a link the participant opened through Facebook.",
          "source_file": "your_facebook_activity/other_activity/link_history.json",
          "columns": {
            "URL": "URL of the opened link.",
            "Date": "ISO 8601 timestamp of the visit."
          }
        }

    Table config::

        {
          "id": "facebook_link_history",
          "title": {
            "en": "Links you opened through Facebook",
            "nl": "Links die je via Facebook hebt geopend",
            "de": "Über Facebook geöffnete Links",
            "pl": "Linki otwarte przez Facebooka",
            "tr": "Facebook üzerinden açtığın bağlantılar",
            "ar": "الروابط التي فتحتها عبر فيسبوك",
            "ru": "Ссылки, открытые через Facebook",
            "it": "Link aperti tramite Facebook",
            "ro": "Linkuri deschise prin Facebook",
            "es": "Enlaces que abriste a través de Facebook",
            "sq": "Lidhjet që ke hapur përmes Facebook"
          },
          "description": {
            "en": "This table shows the links you opened through Facebook and when.",
            "nl": "Deze tabel toont de links die je via Facebook hebt geopend en wanneer.",
            "de": "Diese Tabelle zeigt die Links, die Sie über Facebook geöffnet haben, und wann.",
            "pl": "Ta tabela pokazuje linki, które otworzyłeś/aś przez Facebooka, oraz kiedy.",
            "tr": "Bu tablo, Facebook üzerinden açtığın bağlantıları ve ne zaman açtığını gösterir.",
            "ar": "يعرض هذا الجدول الروابط التي فتحتها عبر فيسبوك ومتى فتحتها.",
            "ru": "В этой таблице показаны ссылки, которые вы открывали через Facebook, и когда.",
            "it": "Questa tabella mostra i link che hai aperto tramite Facebook e quando.",
            "ro": "Acest tabel arată linkurile pe care le-ai deschis prin Facebook și când.",
            "es": "Esta tabla muestra los enlaces que abriste a través de Facebook y cuándo.",
            "sq": "Kjo tabelë tregon lidhjet që ke hapur përmes Facebook dhe kur."
          },
          "headers": {
            "URL": {
              "en": "URL",
              "nl": "URL",
              "de": "URL",
              "pl": "URL",
              "tr": "URL",
              "ar": "الرابط (URL)",
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
    data = _read_json_data(reader, errors, "your_facebook_activity/other_activity/link_history.json")
    rows: list[tuple[str, str]] = []

    try:
        for item in [data] if isinstance(data, dict) else data or []:
            url = ""
            for lv in item.get("label_values", []):
                if isinstance(lv, dict) and (lv.get("href") or lv.get("value")):
                    url = lv.get("href") or lv.get("value")
                    break
            ts = item.get("timestamp")
            rows.append((url, eh.epoch_to_iso(ts, errors=errors) if ts else ""))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return pd.DataFrame(rows, columns=["URL", "Date"]) if rows else pd.DataFrame()


def fundraiser_posts_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract the number of fundraiser posts the participant likely viewed.

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
        Columns: ``Posts``, ``Date``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Number of fundraiser posts Facebook estimates the participant likely viewed.",
          "source_file": "your_facebook_activity/fundraisers/fundraiser_posts_you_likely_viewed.json",
          "columns": {
            "Posts": "Number of fundraiser posts the participant likely viewed.",
            "Date": "ISO 8601 timestamp of the record (empty when not available)."
          }
        }

    Table config::

        {
          "id": "facebook_fundraiser_posts_viewed",
          "title": {
            "en": "Fundraiser posts you likely viewed",
            "nl": "Fondsenwervingsberichten die je waarschijnlijk hebt bekeken",
            "de": "Spendenaktionen, die Sie vermutlich angesehen haben",
            "pl": "Posty o zbiórkach, które prawdopodobnie wyświetliłeś/aś",
            "tr": "Muhtemelen görüntülediğin bağış kampanyası gönderileri",
            "ar": "منشورات جمع التبرعات التي ربما شاهدتها",
            "ru": "Публикации о сборе средств, которые вы, вероятно, просматривали",
            "it": "Post di raccolte fondi che probabilmente hai visualizzato",
            "ro": "Postări despre strângeri de fonduri pe care probabil le-ai vizualizat",
            "es": "Publicaciones de recaudaciones de fondos que probablemente viste",
            "sq": "Postimet e mbledhjeve të fondeve që ke parë me gjasë"
          },
          "description": {
            "en": "This table shows how many fundraiser posts Facebook estimates you viewed.",
            "nl": "Deze tabel toont hoeveel fondsenwervingsberichten je volgens Facebook waarschijnlijk hebt bekeken.",
            "de": "Diese Tabelle zeigt, wie viele Beiträge zu Spendenaktionen Sie laut Facebook vermutlich angesehen haben.",
            "pl": "Ta tabela pokazuje, ile postów o zbiórkach według szacunków Facebooka wyświetliłeś/aś.",
            "tr": "Bu tablo, Facebook'un tahminine göre kaç bağış kampanyası gönderisini görüntülediğini gösterir.",
            "ar": "يعرض هذا الجدول عدد منشورات جمع التبرعات التي يقدّر فيسبوك أنك شاهدتها.",
            "ru": "В этой таблице показано, сколько публикаций о сборе средств, по оценке Facebook, вы просмотрели.",
            "it": "Questa tabella mostra quanti post di raccolte fondi, secondo la stima di Facebook, hai visualizzato.",
            "ro": "Acest tabel arată câte postări despre strângeri de fonduri estimează Facebook că ai vizualizat.",
            "es": "Esta tabla muestra cuántas publicaciones de recaudaciones de fondos Facebook estima que viste.",
            "sq": "Kjo tabelë tregon sa postime mbledhjesh fondesh vlerëson Facebook se ke parë."
          },
          "headers": {
            "Posts": {
              "en": "Number of posts",
              "nl": "Aantal berichten",
              "de": "Anzahl der Beiträge",
              "pl": "Liczba postów",
              "tr": "Gönderi sayısı",
              "ar": "عدد المنشورات",
              "ru": "Количество публикаций",
              "it": "Numero di post",
              "ro": "Număr de postări",
              "es": "Número de publicaciones",
              "sq": "Numri i postimeve"
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
    data = _read_json_data(reader, errors, "your_facebook_activity/fundraisers/fundraiser_posts_you_likely_viewed.json")
    rows: list[tuple[str, str]] = []

    try:
        for item in [data] if isinstance(data, dict) else data or []:
            value = next((_clean_text(lv["value"]) for lv in item.get("label_values", [])
                          if isinstance(lv, dict) and lv.get("value") not in (None, "")), "")
            ts = item.get("timestamp")
            if value:
                rows.append((value, eh.epoch_to_iso(ts, errors=errors) if ts else ""))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return pd.DataFrame(rows, columns=["Posts", "Date"]) if rows else pd.DataFrame()


def friend_suggestions_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Extract how many friend suggestions Facebook generated (counts and dates only, no names).

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
        Columns: ``Suggested``, ``Date``.
        Empty DataFrame when none of the source files are present.

    Table documentation::

        {
          "summary": "Each row is a batch of friend suggestions Facebook generated for the participant: the number of suggested people and the date. The names of the suggested people are not extracted.",
          "source_file": "connections/friends/people_you_may_know.json, connections/friends/suggested_friends.json",
          "columns": {
            "Suggested": "Number of people suggested as friends.",
            "Date": "ISO 8601 timestamp of the suggestion (empty when not available)."
          }
        }

    Table config::

        {
          "id": "facebook_friend_suggestions",
          "title": {
            "en": "Friend suggestions you received",
            "nl": "Vriendsuggesties die je hebt ontvangen",
            "de": "Erhaltene Freundschaftsvorschläge",
            "pl": "Otrzymane propozycje znajomych",
            "tr": "Aldığın arkadaş önerileri",
            "ar": "اقتراحات الأصدقاء التي تلقيتها",
            "ru": "Полученные рекомендации друзей",
            "it": "Suggerimenti di amicizia ricevuti",
            "ro": "Sugestii de prieteni primite",
            "es": "Sugerencias de amistad que recibiste",
            "sq": "Sugjerimet e miqve që ke marrë"
          },
          "description": {
            "en": "This table shows how many friend suggestions Facebook generated for you and when. The names of the suggested people are not included.",
            "nl": "Deze tabel toont hoeveel vriendsuggesties Facebook voor je heeft gegenereerd en wanneer. De namen van de voorgestelde personen zijn niet opgenomen.",
            "de": "Diese Tabelle zeigt, wie viele Freundschaftsvorschläge Facebook für Sie erstellt hat und wann. Die Namen der vorgeschlagenen Personen sind nicht enthalten.",
            "pl": "Ta tabela pokazuje, ile propozycji znajomych wygenerował dla Ciebie Facebook i kiedy. Imiona i nazwiska proponowanych osób nie są uwzględnione.",
            "tr": "Bu tablo, Facebook'un senin için kaç arkadaş önerisi oluşturduğunu ve ne zaman oluşturduğunu gösterir. Önerilen kişilerin adları dahil değildir.",
            "ar": "يعرض هذا الجدول عدد اقتراحات الأصدقاء التي أنشأها فيسبوك لك ومتى. لا تتضمن البيانات أسماء الأشخاص المقترحين.",
            "ru": "В этой таблице показано, сколько рекомендаций друзей Facebook сформировал для вас и когда. Имена рекомендованных людей не включены.",
            "it": "Questa tabella mostra quanti suggerimenti di amicizia Facebook ha generato per te e quando. I nomi delle persone suggerite non sono inclusi.",
            "ro": "Acest tabel arată câte sugestii de prieteni a generat Facebook pentru tine și când. Numele persoanelor sugerate nu sunt incluse.",
            "es": "Esta tabla muestra cuántas sugerencias de amistad generó Facebook para ti y cuándo. No se incluyen los nombres de las personas sugeridas.",
            "sq": "Kjo tabelë tregon sa sugjerime miqsh ka gjeneruar Facebook për ty dhe kur. Emrat e personave të sugjeruar nuk përfshihen."
          },
          "headers": {
            "Suggested": {
              "en": "Number of suggested people",
              "nl": "Aantal voorgestelde personen",
              "de": "Anzahl vorgeschlagener Personen",
              "pl": "Liczba proponowanych osób",
              "tr": "Önerilen kişi sayısı",
              "ar": "عدد الأشخاص المقترحين",
              "ru": "Количество рекомендованных людей",
              "it": "Numero di persone suggerite",
              "ro": "Număr de persoane sugerate",
              "es": "Número de personas sugeridas",
              "sq": "Numri i personave të sugjeruar"
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
    rows: list[tuple[str, str]] = []

    try:
        data = _read_json_data(reader, errors, "connections/friends/people_you_may_know.json")
        for item in [data] if isinstance(data, dict) else data or []:
            ts = item.get("timestamp")
            for lv in item.get("label_values", []):
                if isinstance(lv, dict) and isinstance(lv.get("vec"), list):
                    rows.append((str(len(lv["vec"])), eh.epoch_to_iso(ts, errors=errors) if ts else ""))

        data = _read_json_data(reader, errors, "connections/friends/suggested_friends.json")
        for item in [data] if isinstance(data, dict) else data or []:
            stamps = [lv["timestamp_value"] for lv in item.get("label_values", [])
                      if isinstance(lv, dict) and lv.get("timestamp_value")]
            if stamps:
                rows.append(("1", eh.epoch_to_iso(max(stamps), errors=errors)))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1

    return pd.DataFrame(rows, columns=["Suggested", "Date"]) if rows else pd.DataFrame()



# ---------------------------------------------------------------------------
# Single-file record tables (files also covered by the inference_* pipeline)
#
# Each table reads exactly one export file and shows one row per record: the
# first timestamp of the record as Date, and all remaining fields flattened
# into Details as "label: value" pairs.  Values that look like identifiers
# (IP address, e-mail, phone number, cookie, user agent, long numeric ids)
# are dropped.
# ---------------------------------------------------------------------------

_RECORD_STRUCT_KEYS = _LV_KEYS | {"string_list_data", "string_map_data", "media_list_data"}
_RECORD_SKIP_KEYS = {"fbid", "media", "media_list_data", "uri", "thumbnail"}


def _standard_records(data) -> list:
    """Split a parsed JSON file into records (same rules as the inference pipeline)."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if "label_values" in data or "string_map_data" in data:
        return [data]
    lists = [value for value in data.values() if isinstance(value, list)]
    if len(lists) == 1:
        return lists[0]
    return [data]


def _is_epoch(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _record_timestamp(record):
    """Return the first (breadth-first) epoch timestamp found in *record*."""
    queue = [record]
    while queue:
        node = queue.pop(0)
        if isinstance(node, dict):
            for key, value in node.items():
                if ("timestamp" in str(key).lower() or key == "time") and _is_epoch(value):
                    return value
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)
    return None


def _record_details(record, errors: Counter, skip_ts=None) -> str:
    """Flatten *record* into '; '-joined 'label: value' pairs.

    Timestamps equal to *skip_ts* (the record date, shown in its own column)
    are not repeated.
    """
    parts: list[str] = []

    def add(path: list, value, *, is_timestamp: bool = False) -> None:
        if is_timestamp and value == skip_ts:
            return
        text = eh.epoch_to_iso(value, errors=errors) if is_timestamp else _clean_text(value)
        if not text or _is_sensitive_value(text):
            return
        label = _PATH_SEP.join(p for p in path if p)
        parts.append(f"{label}: {text}" if label else text)

    def walk(node, path: list) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, path)
            return
        if not isinstance(node, dict):
            add(path, node)
            return
        if node.keys() & _RECORD_STRUCT_KEYS:
            label = node.get("label")
            title = node.get("title")
            if not label and title and ("dict" in node or "vec" in node):
                label = title  # a titled group of labelled values
            elif title:
                add(path + ["Titel"], title)
            new_path = path + [_clean_text(label)] if label else path
            if "value" in node:
                add(new_path, node["value"])
            elif node.get("href"):
                add(new_path, node["href"])
            if "value" in node and node.get("href") and node["href"] != node["value"]:
                add(new_path + ["Link"], node["href"])
            if _is_epoch(node.get("timestamp_value")):
                add(new_path, node["timestamp_value"], is_timestamp=True)
            for key in ("label_values", "string_list_data", "vec", "dict"):
                if key in node:
                    walk(node[key], new_path)
            string_map = node.get("string_map_data")
            if isinstance(string_map, dict):
                for key, value in string_map.items():
                    walk(value, new_path + [_clean_text(key)])
            return
        for key, value in node.items():
            if key in _RECORD_SKIP_KEYS:
                continue
            if "timestamp" in str(key).lower():
                if _is_epoch(value):
                    add(path + [_clean_text(key)], value, is_timestamp=True)
                continue
            walk(value, path + [_clean_text(key)])

    walk(record, [])
    return "; ".join(parts)


def _split_repeated_labels(entries: list) -> list[list]:
    """Split a flat label_values list into chunks whenever a label repeats.

    Some files (e.g. snoozed or unfollowed profiles) store every entry as
    consecutive ``Name`` / ``Zeit`` items inside one record.
    """
    chunks: list[list] = []
    current: list = []
    seen: set = set()
    for entry in entries:
        label = entry.get("label") if isinstance(entry, dict) else None
        if label is not None and label in seen:
            chunks.append(current)
            current, seen = [], set()
        current.append(entry)
        if label is not None:
            seen.add(label)
    if current:
        chunks.append(current)
    return chunks


def _name_ts_df(reader: ZipArchiveReader, errors: Counter, path: str) -> pd.DataFrame:
    """``Name / Date`` table for plain ``[{"name": ..., "timestamp": ...}]`` lists."""
    data = _read_json_data(reader, errors, path)
    if data is None:
        return pd.DataFrame()
    rows: list[tuple[str, str]] = []
    try:
        for record in _standard_records(data):
            if not isinstance(record, dict):
                continue
            name = _clean_text(record.get("name") or record.get("title") or "")
            ts = record.get("timestamp")
            if name or _is_epoch(ts):
                rows.append((name, eh.epoch_to_iso(ts, errors=errors) if _is_epoch(ts) else ""))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return pd.DataFrame(rows, columns=["Name", "Date"]) if rows else pd.DataFrame()


def _label_pairs_df(reader: ZipArchiveReader, errors: Counter, path: str) -> pd.DataFrame:
    """``Name / Date`` table for flat, repeating ``label_values`` (Name, Zeit, Name, Zeit, ...).

    Exact duplicates (same name and time), which Meta sometimes exports twice,
    are listed only once.
    """
    data = _read_json_data(reader, errors, path)
    if data is None:
        return pd.DataFrame()
    rows: list[tuple[str, str]] = []
    seen: set = set()
    try:
        for record in _standard_records(data):
            if not isinstance(record, dict):
                continue
            for chunk in _split_repeated_labels(record.get("label_values", [])):
                name = next((_clean_text(e["value"]) for e in chunk
                             if isinstance(e, dict) and e.get("value") not in (None, "")), "")
                ts = next((e["timestamp_value"] for e in chunk
                           if isinstance(e, dict) and _is_epoch(e.get("timestamp_value"))), None)
                row = (name, eh.epoch_to_iso(ts, errors=errors) if ts else "")
                if (name or ts) and row not in seen:
                    seen.add(row)
                    rows.append(row)
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return pd.DataFrame(rows, columns=["Name", "Date"]) if rows else pd.DataFrame()


_BOOL_TEXTS = {"richtig", "falsch", "true", "false", "waar", "onwaar", "vrai", "faux", "vero", "falso", "verdadero"}


def _post_audiences_df(reader: ZipArchiveReader, errors: Counter, path: str) -> pd.DataFrame:
    """``List / Deactivated / Group / Member`` table, one row per list member.

    Lists without members are shown as one row with empty Group and Member.
    """
    data = _read_json_data(reader, errors, path)
    if data is None:
        return pd.DataFrame()
    rows: list[tuple[str, str, str, str]] = []
    try:
        for record in _standard_records(data):
            if not isinstance(record, dict):
                continue
            list_name, deactivated, members = "", "", []
            for entry in record.get("label_values", []):
                if not isinstance(entry, dict):
                    continue
                if "dict" in entry:
                    group = _clean_text(entry.get("title") or entry.get("label") or "")
                    for member in entry.get("dict") or []:
                        values = [_clean_text(v["value"]) for v in (member.get("dict") or [member])
                                  if isinstance(v, dict) and v.get("value") not in (None, "")]
                        for value in values:
                            members.append((group, value))
                elif "value" in entry:
                    value = entry["value"]
                    text = _clean_text(value)
                    if isinstance(value, bool) or text.lower() in _BOOL_TEXTS:
                        deactivated = text
                    elif not list_name:
                        list_name = text
            if members:
                rows.extend((list_name, deactivated, group, member) for group, member in members)
            elif list_name:
                rows.append((list_name, deactivated, "", ""))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    return (pd.DataFrame(rows, columns=["List", "Deactivated", "Group", "Member"])
            if rows else pd.DataFrame())


def _expand_records(record):
    """Yield ``(group_label, item, fallback_ts)`` rows for one record.

    Meta often stores a whole list (places, profiles, ...) inside a single
    record as ``label_values -> [{label, vec: [...]}]``.  Each element of such
    a list becomes its own row, so participants see every entry separately and
    can deselect it.  Records without such a list are returned as one row.
    """
    groups = []
    if isinstance(record, dict) and isinstance(record.get("label_values"), list):
        for entry in record["label_values"]:
            if isinstance(entry, dict) and isinstance(entry.get("vec"), list) and entry["vec"]:
                groups.append((entry.get("label") or entry.get("title") or "", entry["vec"]))
    if not groups:
        entries = record.get("label_values") if isinstance(record, dict) else None
        if isinstance(entries, list):
            chunks = _split_repeated_labels(entries)
            if len(chunks) > 1:
                for chunk in chunks:
                    yield "", {"label_values": chunk}, None
                return
        yield "", record, None
        return
    record_ts = record.get("timestamp") if _is_epoch(record.get("timestamp")) else None
    show_label = len(groups) > 1
    for label, items in groups:
        for item in items:
            yield (_clean_text(label) if show_label else ""), item, record_ts


def _records_df(reader: ZipArchiveReader, errors: Counter, path: str) -> pd.DataFrame:
    """Build a ``Date / Details`` table from one export file, one row per entry."""
    data = _read_json_data(reader, errors, path)
    if data is None:
        return pd.DataFrame()
    rows: list[tuple[str, str]] = []
    try:
        for record in _standard_records(data):
            for label, item, fallback_ts in _expand_records(record):
                ts = _record_timestamp(item) or fallback_ts
                details = _record_details(item, errors, skip_ts=ts)
                if label and details:
                    details = f"{label}: {details}"
                if not details and ts is None:
                    continue
                rows.append((eh.epoch_to_iso(ts, errors=errors) if ts else "", details))
    except Exception as e:
        logger.error("Exception caught: %s", e)
        errors[type(e).__name__] += 1
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["Date", "Details"])


def _count_df(reader: ZipArchiveReader, errors: Counter, path: str) -> pd.DataFrame:
    """Return a one-cell ``Count`` table with the first number found in *path*."""
    data = _read_json_data(reader, errors, path)
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



def groups_and_events_youve_visited_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a Facebook group or event the participant visited.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is a Facebook group or event the participant visited.",
          "source_file": "logged_information/interactions/groups_and_events_you've_visited.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_groups_and_events_youve_visited",
          "title": {
            "en": "Groups and events you visited",
            "nl": "Groepen en evenementen die je hebt bezocht",
            "de": "Besuchte Gruppen und Veranstaltungen",
            "pl": "Odwiedzone grupy i wydarzenia",
            "tr": "Ziyaret ettiğin gruplar ve etkinlikler",
            "ar": "المجموعات والفعاليات التي زرتها",
            "ru": "Посещённые группы и мероприятия",
            "it": "Gruppi ed eventi visitati",
            "ro": "Grupuri și evenimente vizitate",
            "es": "Grupos y eventos que visitaste",
            "sq": "Grupet dhe ngjarjet që ke vizituar"
          },
          "description": {
            "en": "This table shows the Facebook groups and events you visited.",
            "nl": "Deze tabel toont de Facebook-groepen en -evenementen die je hebt bezocht.",
            "de": "Diese Tabelle zeigt die Facebook-Gruppen und -Veranstaltungen, die Sie besucht haben.",
            "pl": "Ta tabela pokazuje grupy i wydarzenia na Facebooku, które odwiedziłeś/aś.",
            "tr": "Bu tablo, ziyaret ettiğin Facebook gruplarını ve etkinliklerini gösterir.",
            "ar": "يعرض هذا الجدول مجموعات وفعاليات فيسبوك التي زرتها.",
            "ru": "В этой таблице показаны группы и мероприятия Facebook, которые вы посещали.",
            "it": "Questa tabella mostra i gruppi e gli eventi di Facebook che hai visitato.",
            "ro": "Acest tabel arată grupurile și evenimentele de pe Facebook pe care le-ai vizitat.",
            "es": "Esta tabla muestra los grupos y eventos de Facebook que visitaste.",
            "sq": "Kjo tabelë tregon grupet dhe ngjarjet në Facebook që ke vizituar."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "logged_information/interactions/groups_and_events_you've_visited.json")


def your_comment_edits_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is an edit the participant made to one of their Facebook comments.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is an edit the participant made to one of their Facebook comments.",
          "source_file": "your_facebook_activity/comments_and_reactions/your_comment_edits.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_your_comment_edits",
          "title": {
            "en": "Your comment edits",
            "nl": "Bewerkingen van je reacties",
            "de": "Ihre bearbeiteten Kommentare",
            "pl": "Twoje edycje komentarzy",
            "tr": "Yorum düzenlemelerin",
            "ar": "تعديلات تعليقاتك",
            "ru": "Ваши изменения комментариев",
            "it": "Le modifiche ai tuoi commenti",
            "ro": "Editările comentariilor tale",
            "es": "Tus ediciones de comentarios",
            "sq": "Ndryshimet e komenteve të tua"
          },
          "description": {
            "en": "This table shows the edits you made to your comments on Facebook.",
            "nl": "Deze tabel toont de bewerkingen die je aan je reacties op Facebook hebt gedaan.",
            "de": "Diese Tabelle zeigt, wie Sie Ihre Kommentare auf Facebook bearbeitet haben.",
            "pl": "Ta tabela pokazuje edycje Twoich komentarzy na Facebooku.",
            "tr": "Bu tablo, Facebook'taki yorumlarında yaptığın düzenlemeleri gösterir.",
            "ar": "يعرض هذا الجدول التعديلات التي أجريتها على تعليقاتك في فيسبوك.",
            "ru": "В этой таблице показаны изменения, которые вы вносили в свои комментарии на Facebook.",
            "it": "Questa tabella mostra le modifiche che hai apportato ai tuoi commenti su Facebook.",
            "ro": "Acest tabel arată editările pe care le-ai făcut comentariilor tale pe Facebook.",
            "es": "Esta tabla muestra las ediciones que hiciste a tus comentarios en Facebook.",
            "sq": "Kjo tabelë tregon ndryshimet që ke bërë në komentet e tua në Facebook."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "your_facebook_activity/comments_and_reactions/your_comment_edits.json")


def stories_you_reacted_to_with_a_sticker_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a Facebook Story the participant reacted to with a sticker.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is a Facebook Story the participant reacted to with a sticker.",
          "source_file": "your_facebook_activity/stories/stories_you_reacted_to_with_a_sticker.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_stories_you_reacted_to_with_a_sticker",
          "title": {
            "en": "Sticker reactions to stories",
            "nl": "Stickerreacties op stories",
            "de": "Sticker-Reaktionen auf Storys",
            "pl": "Reakcje naklejkami na relacje",
            "tr": "Hikayelere çıkartma tepkileri",
            "ar": "التفاعل مع القصص بالملصقات",
            "ru": "Реакции стикерами на истории",
            "it": "Reazioni con sticker alle storie",
            "ro": "Reacții cu stickere la povești",
            "es": "Reacciones con stickers a historias",
            "sq": "Reagime me ngjitëse ndaj stories"
          },
          "description": {
            "en": "This table shows the Facebook stories you reacted to with a sticker.",
            "nl": "Deze tabel toont de Facebook-stories waarop je met een sticker hebt gereageerd.",
            "de": "Diese Tabelle zeigt die Facebook-Storys, auf die Sie mit einem Sticker reagiert haben.",
            "pl": "Ta tabela pokazuje relacje na Facebooku, na które zareagowałeś/aś naklejką.",
            "tr": "Bu tablo, çıkartmayla tepki verdiğin Facebook hikayelerini gösterir.",
            "ar": "يعرض هذا الجدول قصص فيسبوك التي تفاعلت معها بملصق.",
            "ru": "В этой таблице показаны истории Facebook, на которые вы отреагировали стикером.",
            "it": "Questa tabella mostra le storie di Facebook a cui hai reagito con uno sticker.",
            "ro": "Acest tabel arată poveștile de pe Facebook la care ai reacționat cu un sticker.",
            "es": "Esta tabla muestra las historias de Facebook a las que reaccionaste con un sticker.",
            "sq": "Kjo tabelë tregon stories në Facebook ndaj të cilave ke reaguar me ngjitëse."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "your_facebook_activity/stories/stories_you_reacted_to_with_a_sticker.json")


def your_educational_notification_interactions_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is an interaction with an educational notice shown in Facebook search.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is an interaction with an educational notice shown in Facebook search.",
          "source_file": "logged_information/search/your_educational_notification_interactions.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_your_educational_notification_interactions",
          "title": {
            "en": "Interactions with notices in search",
            "nl": "Interacties met meldingen in zoeken",
            "de": "Interaktionen mit Hinweisen in der Suche",
            "pl": "Interakcje z komunikatami w wyszukiwarce",
            "tr": "Aramadaki bildirimlerle etkileşimler",
            "ar": "التفاعلات مع الإشعارات في البحث",
            "ru": "Взаимодействия с уведомлениями в поиске",
            "it": "Interazioni con avvisi nella ricerca",
            "ro": "Interacțiuni cu notificări în căutare",
            "es": "Interacciones con avisos en la búsqueda",
            "sq": "Ndërveprime me njoftimet në kërkim"
          },
          "description": {
            "en": "This table shows your interactions with educational notices that Facebook displayed in search.",
            "nl": "Deze tabel toont je interacties met informatieve meldingen die Facebook in de zoekfunctie heeft getoond.",
            "de": "Diese Tabelle zeigt Ihre Interaktionen mit Hinweisen, die Facebook in der Suche angezeigt hat.",
            "pl": "Ta tabela pokazuje Twoje interakcje z komunikatami edukacyjnymi wyświetlanymi przez Facebooka w wyszukiwarce.",
            "tr": "Bu tablo, Facebook'un aramada gösterdiği bilgilendirici bildirimlerle etkileşimlerini gösterir.",
            "ar": "يعرض هذا الجدول تفاعلاتك مع الإشعارات التوعوية التي عرضها فيسبوك في البحث.",
            "ru": "В этой таблице показаны ваши взаимодействия с информационными уведомлениями, которые Facebook показывал в поиске.",
            "it": "Questa tabella mostra le tue interazioni con gli avvisi informativi mostrati da Facebook nella ricerca.",
            "ro": "Acest tabel arată interacțiunile tale cu notificările informative afișate de Facebook în căutare.",
            "es": "Esta tabla muestra tus interacciones con avisos informativos que Facebook mostró en la búsqueda.",
            "sq": "Kjo tabelë tregon ndërveprimet e tua me njoftimet informuese që Facebook shfaqi në kërkim."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "logged_information/search/your_educational_notification_interactions.json")


def received_friend_requests_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a friend request the participant received.

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
          "summary": "Each row is a friend request the participant received.",
          "source_file": "connections/friends/received_friend_requests.json",
          "columns": {
            "Name": "Name of the person, page or profile.",
            "Date": "ISO 8601 timestamp of the entry."
          }
        }

    Table config::

        {
          "id": "facebook_received_friend_requests",
          "title": {
            "en": "Friend requests you received",
            "nl": "Ontvangen vriendschapsverzoeken",
            "de": "Erhaltene Freundschaftsanfragen",
            "pl": "Otrzymane zaproszenia do znajomych",
            "tr": "Aldığın arkadaşlık istekleri",
            "ar": "طلبات الصداقة التي تلقيتها",
            "ru": "Полученные запросы в друзья",
            "it": "Richieste di amicizia ricevute",
            "ro": "Cereri de prietenie primite",
            "es": "Solicitudes de amistad recibidas",
            "sq": "Kërkesat për miqësi që ke marrë"
          },
          "description": {
            "en": "This table shows the friend requests you received on Facebook.",
            "nl": "Deze tabel toont de vriendschapsverzoeken die je op Facebook hebt ontvangen.",
            "de": "Diese Tabelle zeigt die Freundschaftsanfragen, die Sie auf Facebook erhalten haben.",
            "pl": "Ta tabela pokazuje zaproszenia do znajomych otrzymane na Facebooku.",
            "tr": "Bu tablo, Facebook'ta aldığın arkadaşlık isteklerini gösterir.",
            "ar": "يعرض هذا الجدول طلبات الصداقة التي تلقيتها على فيسبوك.",
            "ru": "В этой таблице показаны запросы в друзья, полученные вами на Facebook.",
            "it": "Questa tabella mostra le richieste di amicizia che hai ricevuto su Facebook.",
            "ro": "Acest tabel arată cererile de prietenie primite pe Facebook.",
            "es": "Esta tabla muestra las solicitudes de amistad que recibiste en Facebook.",
            "sq": "Kjo tabelë tregon kërkesat për miqësi që ke marrë në Facebook."
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
    return _name_ts_df(reader, errors, "connections/friends/received_friend_requests.json")


def sent_friend_requests_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a friend request the participant sent.

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
          "summary": "Each row is a friend request the participant sent.",
          "source_file": "connections/friends/sent_friend_requests.json",
          "columns": {
            "Name": "Name of the person, page or profile.",
            "Date": "ISO 8601 timestamp of the entry."
          }
        }

    Table config::

        {
          "id": "facebook_sent_friend_requests",
          "title": {
            "en": "Friend requests you sent",
            "nl": "Verstuurde vriendschapsverzoeken",
            "de": "Gesendete Freundschaftsanfragen",
            "pl": "Wysłane zaproszenia do znajomych",
            "tr": "Gönderdiğin arkadaşlık istekleri",
            "ar": "طلبات الصداقة التي أرسلتها",
            "ru": "Отправленные запросы в друзья",
            "it": "Richieste di amicizia inviate",
            "ro": "Cereri de prietenie trimise",
            "es": "Solicitudes de amistad enviadas",
            "sq": "Kërkesat për miqësi që ke dërguar"
          },
          "description": {
            "en": "This table shows the friend requests you sent on Facebook.",
            "nl": "Deze tabel toont de vriendschapsverzoeken die je op Facebook hebt verstuurd.",
            "de": "Diese Tabelle zeigt die Freundschaftsanfragen, die Sie auf Facebook gesendet haben.",
            "pl": "Ta tabela pokazuje zaproszenia do znajomych wysłane na Facebooku.",
            "tr": "Bu tablo, Facebook'ta gönderdiğin arkadaşlık isteklerini gösterir.",
            "ar": "يعرض هذا الجدول طلبات الصداقة التي أرسلتها على فيسبوك.",
            "ru": "В этой таблице показаны запросы в друзья, отправленные вами на Facebook.",
            "it": "Questa tabella mostra le richieste di amicizia che hai inviato su Facebook.",
            "ro": "Acest tabel arată cererile de prietenie trimise pe Facebook.",
            "es": "Esta tabla muestra las solicitudes de amistad que enviaste en Facebook.",
            "sq": "Kjo tabelë tregon kërkesat për miqësi që ke dërguar në Facebook."
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
    return _name_ts_df(reader, errors, "connections/friends/sent_friend_requests.json")


def rejected_friend_requests_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a friend request the participant rejected.

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
          "summary": "Each row is a friend request the participant rejected.",
          "source_file": "connections/friends/rejected_friend_requests.json",
          "columns": {
            "Name": "Name of the person, page or profile.",
            "Date": "ISO 8601 timestamp of the entry."
          }
        }

    Table config::

        {
          "id": "facebook_rejected_friend_requests",
          "title": {
            "en": "Friend requests you rejected",
            "nl": "Geweigerde vriendschapsverzoeken",
            "de": "Abgelehnte Freundschaftsanfragen",
            "pl": "Odrzucone zaproszenia do znajomych",
            "tr": "Reddettiğin arkadaşlık istekleri",
            "ar": "طلبات الصداقة التي رفضتها",
            "ru": "Отклонённые запросы в друзья",
            "it": "Richieste di amicizia rifiutate",
            "ro": "Cereri de prietenie respinse",
            "es": "Solicitudes de amistad rechazadas",
            "sq": "Kërkesat për miqësi që ke refuzuar"
          },
          "description": {
            "en": "This table shows the friend requests you rejected on Facebook.",
            "nl": "Deze tabel toont de vriendschapsverzoeken die je op Facebook hebt geweigerd.",
            "de": "Diese Tabelle zeigt die Freundschaftsanfragen, die Sie auf Facebook abgelehnt haben.",
            "pl": "Ta tabela pokazuje zaproszenia do znajomych odrzucone na Facebooku.",
            "tr": "Bu tablo, Facebook'ta reddettiğin arkadaşlık isteklerini gösterir.",
            "ar": "يعرض هذا الجدول طلبات الصداقة التي رفضتها على فيسبوك.",
            "ru": "В этой таблице показаны запросы в друзья, отклонённые вами на Facebook.",
            "it": "Questa tabella mostra le richieste di amicizia che hai rifiutato su Facebook.",
            "ro": "Acest tabel arată cererile de prietenie respinse pe Facebook.",
            "es": "Esta tabla muestra las solicitudes de amistad que rechazaste en Facebook.",
            "sq": "Kjo tabelë tregon kërkesat për miqësi që ke refuzuar në Facebook."
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
    return _name_ts_df(reader, errors, "connections/friends/rejected_friend_requests.json")


def removed_friends_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a friend the participant removed.

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
          "summary": "Each row is a friend the participant removed.",
          "source_file": "connections/friends/removed_friends.json",
          "columns": {
            "Name": "Name of the person, page or profile.",
            "Date": "ISO 8601 timestamp of the entry."
          }
        }

    Table config::

        {
          "id": "facebook_removed_friends",
          "title": {
            "en": "Friends you removed",
            "nl": "Verwijderde vrienden",
            "de": "Entfernte Freund*innen",
            "pl": "Usunięci znajomi",
            "tr": "Çıkardığın arkadaşlar",
            "ar": "الأصدقاء الذين أزلتهم",
            "ru": "Удалённые друзья",
            "it": "Amici rimossi",
            "ro": "Prieteni eliminați",
            "es": "Amigos que eliminaste",
            "sq": "Miqtë që ke hequr"
          },
          "description": {
            "en": "This table shows the friends you removed on Facebook.",
            "nl": "Deze tabel toont de vrienden die je op Facebook hebt verwijderd.",
            "de": "Diese Tabelle zeigt die Freund*innen, die Sie auf Facebook entfernt haben.",
            "pl": "Ta tabela pokazuje znajomych usuniętych przez Ciebie na Facebooku.",
            "tr": "Bu tablo, Facebook'ta çıkardığın arkadaşları gösterir.",
            "ar": "يعرض هذا الجدول الأصدقاء الذين أزلتهم على فيسبوك.",
            "ru": "В этой таблице показаны друзья, которых вы удалили на Facebook.",
            "it": "Questa tabella mostra gli amici che hai rimosso su Facebook.",
            "ro": "Acest tabel arată prietenii pe care i-ai eliminat pe Facebook.",
            "es": "Esta tabla muestra los amigos que eliminaste en Facebook.",
            "sq": "Kjo tabelë tregon miqtë që ke hequr në Facebook."
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
    return _name_ts_df(reader, errors, "connections/friends/removed_friends.json")


def your_post_audiences_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is an audience setting the participant used for their posts.

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
        Columns: ``List``, ``Deactivated``, ``Group``, ``Member``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is an audience setting the participant used for their posts.",
          "source_file": "connections/friends/your_post_audiences.json",
          "columns": {
            "List": "Name of the custom audience list.",
            "Deactivated": "Whether the list is deactivated, as exported.",
            "Group": "Part of the list the member belongs to (e.g. members, rejected suggested members).",
            "Member": "Name of the person in the list (empty when the list has no members)."
          }
        }

    Table config::

        {
          "id": "facebook_your_post_audiences",
          "title": {
            "en": "Audiences for your posts",
            "nl": "Doelgroepen van je berichten",
            "de": "Zielgruppen Ihrer Beiträge",
            "pl": "Odbiorcy Twoich postów",
            "tr": "Gönderilerinin kitleleri",
            "ar": "جمهور منشوراتك",
            "ru": "Аудитории ваших публикаций",
            "it": "Pubblico dei tuoi post",
            "ro": "Publicul postărilor tale",
            "es": "Público de tus publicaciones",
            "sq": "Audienca e postimeve të tua"
          },
          "description": {
            "en": "This table shows which audiences you selected for your Facebook posts.",
            "nl": "Deze tabel toont welke doelgroepen je voor je Facebook-berichten hebt gekozen.",
            "de": "Diese Tabelle zeigt, welche Zielgruppen Sie für Ihre Facebook-Beiträge ausgewählt haben.",
            "pl": "Ta tabela pokazuje, jakich odbiorców wybrałeś/aś dla swoich postów na Facebooku.",
            "tr": "Bu tablo, Facebook gönderilerin için seçtiğin kitleleri gösterir.",
            "ar": "يعرض هذا الجدول الجمهور الذي اخترته لمنشوراتك على فيسبوك.",
            "ru": "В этой таблице показано, какие аудитории вы выбирали для своих публикаций на Facebook.",
            "it": "Questa tabella mostra quale pubblico hai scelto per i tuoi post su Facebook.",
            "ro": "Acest tabel arată ce public ai ales pentru postările tale de pe Facebook.",
            "es": "Esta tabla muestra qué público elegiste para tus publicaciones de Facebook.",
            "sq": "Kjo tabelë tregon cilën audiencë ke zgjedhur për postimet e tua në Facebook."
          },
          "headers": {
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
            },
            "Deactivated": {
              "en": "Deactivated",
              "nl": "Gedeactiveerd",
              "de": "Deaktiviert",
              "pl": "Dezaktywowana",
              "tr": "Devre dışı",
              "ar": "معطّلة",
              "ru": "Отключён",
              "it": "Disattivato",
              "ro": "Dezactivată",
              "es": "Desactivada",
              "sq": "Çaktivizuar"
            },
            "Group": {
              "en": "Group",
              "nl": "Groep",
              "de": "Gruppe",
              "pl": "Grupa",
              "tr": "Grup",
              "ar": "المجموعة",
              "ru": "Группа",
              "it": "Gruppo",
              "ro": "Grup",
              "es": "Grupo",
              "sq": "Grupi"
            },
            "Member": {
              "en": "Member",
              "nl": "Lid",
              "de": "Mitglied",
              "pl": "Członek",
              "tr": "Üye",
              "ar": "العضو",
              "ru": "Участник",
              "it": "Membro",
              "ro": "Membru",
              "es": "Miembro",
              "sq": "Anëtari"
            }
          }
        }
    """
    return _post_audiences_df(reader, errors, "connections/friends/your_post_audiences.json")


def locations_of_interest_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a location Facebook associates with the participant's interests.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is a location Facebook associates with the participant's interests.",
          "source_file": "logged_information/other_logged_information/locations_of_interest.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_locations_of_interest",
          "title": {
            "en": "Locations of interest",
            "nl": "Interessante locaties",
            "de": "Interessante Orte",
            "pl": "Interesujące lokalizacje",
            "tr": "İlgi duyulan konumlar",
            "ar": "المواقع التي تهمك",
            "ru": "Интересные места",
            "it": "Luoghi di interesse",
            "ro": "Locații de interes",
            "es": "Ubicaciones de interés",
            "sq": "Vendndodhjet me interes"
          },
          "description": {
            "en": "This table shows locations that Facebook associates with your interests.",
            "nl": "Deze tabel toont locaties die Facebook met je interesses in verband brengt.",
            "de": "Diese Tabelle zeigt Orte, die Facebook mit Ihren Interessen in Verbindung bringt.",
            "pl": "Ta tabela pokazuje lokalizacje, które Facebook wiąże z Twoimi zainteresowaniami.",
            "tr": "Bu tablo, Facebook'un ilgi alanlarınla ilişkilendirdiği konumları gösterir.",
            "ar": "يعرض هذا الجدول المواقع التي يربطها فيسبوك باهتماماتك.",
            "ru": "В этой таблице показаны места, которые Facebook связывает с вашими интересами.",
            "it": "Questa tabella mostra i luoghi che Facebook associa ai tuoi interessi.",
            "ro": "Acest tabel arată locațiile pe care Facebook le asociază cu interesele tale.",
            "es": "Esta tabla muestra las ubicaciones que Facebook asocia con tus intereses.",
            "sq": "Kjo tabelë tregon vendndodhjet që Facebook lidh me interesat e tua."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "logged_information/other_logged_information/locations_of_interest.json")


def apps_detected_from_your_activity_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is an app Meta detected from the participant's activity.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is an app Meta detected from the participant's activity.",
          "source_file": "ads_information/apps_detected_from_your_activity.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_apps_detected_from_your_activity",
          "title": {
            "en": "Apps detected from your activity",
            "nl": "Apps herkend uit je activiteit",
            "de": "Aus Ihren Aktivitäten erkannte Apps",
            "pl": "Aplikacje wykryte na podstawie aktywności",
            "tr": "Etkinliğinden tespit edilen uygulamalar",
            "ar": "التطبيقات المكتشفة من نشاطك",
            "ru": "Приложения, обнаруженные по вашей активности",
            "it": "App rilevate dalla tua attività",
            "ro": "Aplicații detectate din activitatea ta",
            "es": "Aplicaciones detectadas a partir de tu actividad",
            "sq": "Aplikacionet e zbuluara nga aktiviteti yt"
          },
          "description": {
            "en": "This table shows apps that Meta detected from your activity.",
            "nl": "Deze tabel toont apps die Meta uit je activiteit heeft herkend.",
            "de": "Diese Tabelle zeigt Apps, die Meta aus Ihren Aktivitäten erkannt hat.",
            "pl": "Ta tabela pokazuje aplikacje, które Meta wykryła na podstawie Twojej aktywności.",
            "tr": "Bu tablo, Meta'nın etkinliğinden tespit ettiği uygulamaları gösterir.",
            "ar": "يعرض هذا الجدول التطبيقات التي اكتشفتها Meta من نشاطك.",
            "ru": "В этой таблице показаны приложения, которые Meta обнаружила по вашей активности.",
            "it": "Questa tabella mostra le app che Meta ha rilevato dalla tua attività.",
            "ro": "Acest tabel arată aplicațiile pe care Meta le-a detectat din activitatea ta.",
            "es": "Esta tabla muestra las aplicaciones que Meta detectó a partir de tu actividad.",
            "sq": "Kjo tabelë tregon aplikacionet që Meta ka zbuluar nga aktiviteti yt."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "ads_information/apps_detected_from_your_activity.json")


def apps_detected_on_your_account_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is an app Meta detected on the participant's account.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is an app Meta detected on the participant's account.",
          "source_file": "ads_information/apps_detected_on_your_account.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_apps_detected_on_your_account",
          "title": {
            "en": "Apps detected on your account",
            "nl": "Apps herkend in je account",
            "de": "In Ihrem Konto erkannte Apps",
            "pl": "Aplikacje wykryte na Twoim koncie",
            "tr": "Hesabında tespit edilen uygulamalar",
            "ar": "التطبيقات المكتشفة في حسابك",
            "ru": "Приложения, обнаруженные в вашем аккаунте",
            "it": "App rilevate sul tuo account",
            "ro": "Aplicații detectate în contul tău",
            "es": "Aplicaciones detectadas en tu cuenta",
            "sq": "Aplikacionet e zbuluara në llogarinë tënde"
          },
          "description": {
            "en": "This table shows apps that Meta detected on your account.",
            "nl": "Deze tabel toont apps die Meta in je account heeft herkend.",
            "de": "Diese Tabelle zeigt Apps, die Meta in Ihrem Konto erkannt hat.",
            "pl": "Ta tabela pokazuje aplikacje, które Meta wykryła na Twoim koncie.",
            "tr": "Bu tablo, Meta'nın hesabında tespit ettiği uygulamaları gösterir.",
            "ar": "يعرض هذا الجدول التطبيقات التي اكتشفتها Meta في حسابك.",
            "ru": "В этой таблице показаны приложения, которые Meta обнаружила в вашем аккаунте.",
            "it": "Questa tabella mostra le app che Meta ha rilevato sul tuo account.",
            "ro": "Acest tabel arată aplicațiile pe care Meta le-a detectat în contul tău.",
            "es": "Esta tabla muestra las aplicaciones que Meta detectó en tu cuenta.",
            "sq": "Kjo tabelë tregon aplikacionet që Meta ka zbuluar në llogarinë tënde."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "ads_information/apps_detected_on_your_account.json")


def ads_about_meta_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is an ad about Meta shown to the participant.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is an ad about Meta shown to the participant.",
          "source_file": "ads_information/ads_about_meta.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_ads_about_meta",
          "title": {
            "en": "Ads about Meta",
            "nl": "Advertenties over Meta",
            "de": "Werbung über Meta",
            "pl": "Reklamy dotyczące Meta",
            "tr": "Meta hakkındaki reklamlar",
            "ar": "إعلانات عن Meta",
            "ru": "Реклама о Meta",
            "it": "Inserzioni su Meta",
            "ro": "Reclame despre Meta",
            "es": "Anuncios sobre Meta",
            "sq": "Reklama për Meta"
          },
          "description": {
            "en": "This table shows ads about Meta that were shown to you.",
            "nl": "Deze tabel toont advertenties over Meta die aan je zijn getoond.",
            "de": "Diese Tabelle zeigt Werbung über Meta, die Ihnen angezeigt wurde.",
            "pl": "Ta tabela pokazuje reklamy dotyczące Meta, które zostały Ci wyświetlone.",
            "tr": "Bu tablo, sana gösterilen Meta hakkındaki reklamları gösterir.",
            "ar": "يعرض هذا الجدول الإعلانات عن Meta التي عُرضت لك.",
            "ru": "В этой таблице показана реклама о Meta, которую вам показывали.",
            "it": "Questa tabella mostra le inserzioni su Meta che ti sono state mostrate.",
            "ro": "Acest tabel arată reclamele despre Meta care ți-au fost afișate.",
            "es": "Esta tabla muestra los anuncios sobre Meta que se te mostraron.",
            "sq": "Kjo tabelë tregon reklamat për Meta që të janë shfaqur."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "ads_information/ads_about_meta.json")


def meta_ad_library_profiles_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a profile of the participant listed in Meta's Ad Library.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is a profile of the participant listed in Meta's Ad Library.",
          "source_file": "ads_information/meta_ad_library_profiles.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_meta_ad_library_profiles",
          "title": {
            "en": "Profiles in the Ad Library",
            "nl": "Profielen in de advertentiebibliotheek",
            "de": "Profile in der Werbebibliothek",
            "pl": "Profile w Bibliotece reklam",
            "tr": "Reklam Kütüphanesi'ndeki profiller",
            "ar": "الملفات في مكتبة الإعلانات",
            "ru": "Профили в Библиотеке рекламы",
            "it": "Profili nella Libreria inserzioni",
            "ro": "Profiluri în Biblioteca de reclame",
            "es": "Perfiles en la Biblioteca de anuncios",
            "sq": "Profilet në Bibliotekën e reklamave"
          },
          "description": {
            "en": "This table shows your profiles that appear in Meta's Ad Library.",
            "nl": "Deze tabel toont je profielen die in de advertentiebibliotheek van Meta staan.",
            "de": "Diese Tabelle zeigt Ihre Profile, die in der Werbebibliothek von Meta erscheinen.",
            "pl": "Ta tabela pokazuje Twoje profile widoczne w Bibliotece reklam Meta.",
            "tr": "Bu tablo, Meta'nın Reklam Kütüphanesi'nde görünen profillerini gösterir.",
            "ar": "يعرض هذا الجدول ملفاتك الشخصية التي تظهر في مكتبة إعلانات Meta.",
            "ru": "В этой таблице показаны ваши профили, которые есть в Библиотеке рекламы Meta.",
            "it": "Questa tabella mostra i tuoi profili presenti nella Libreria inserzioni di Meta.",
            "ro": "Acest tabel arată profilurile tale care apar în Biblioteca de reclame Meta.",
            "es": "Esta tabla muestra tus perfiles que aparecen en la Biblioteca de anuncios de Meta.",
            "sq": "Kjo tabelë tregon profilet e tua që shfaqen në Bibliotekën e reklamave të Meta."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "ads_information/meta_ad_library_profiles.json")


def off_meta_activity_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a website or app that shared the participant's activity with Meta, with the recorded events.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is a website or app that shared the participant's activity with Meta, with the recorded events.",
          "source_file": "apps_and_websites_off_of_facebook/your_activity_off_meta_technologies.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_off_meta_activity",
          "title": {
            "en": "Activity off Meta technologies",
            "nl": "Activiteit buiten Meta-technologieën",
            "de": "Aktivitäten außerhalb von Meta-Technologien",
            "pl": "Aktywność poza technologiami Meta",
            "tr": "Meta teknolojileri dışındaki etkinlikler",
            "ar": "النشاط خارج تقنيات Meta",
            "ru": "Активность вне технологий Meta",
            "it": "Attività al di fuori delle tecnologie Meta",
            "ro": "Activitate în afara tehnologiilor Meta",
            "es": "Actividad fuera de las tecnologías de Meta",
            "sq": "Aktiviteti jashtë teknologjive të Meta"
          },
          "description": {
            "en": "This table shows which websites and apps shared your activity with Meta.",
            "nl": "Deze tabel toont welke websites en apps je activiteit met Meta hebben gedeeld.",
            "de": "Diese Tabelle zeigt, welche Websites und Apps Ihre Aktivitäten mit Meta geteilt haben.",
            "pl": "Ta tabela pokazuje, które witryny i aplikacje udostępniły Twoją aktywność Meta.",
            "tr": "Bu tablo, hangi web sitelerinin ve uygulamaların etkinliğini Meta ile paylaştığını gösterir.",
            "ar": "يعرض هذا الجدول المواقع والتطبيقات التي شاركت نشاطك مع Meta.",
            "ru": "В этой таблице показано, какие сайты и приложения передавали вашу активность Meta.",
            "it": "Questa tabella mostra quali siti web e app hanno condiviso la tua attività con Meta.",
            "ro": "Acest tabel arată ce site-uri și aplicații au partajat activitatea ta cu Meta.",
            "es": "Esta tabla muestra qué sitios web y aplicaciones compartieron tu actividad con Meta.",
            "sq": "Kjo tabelë tregon cilat faqe dhe aplikacione kanë ndarë aktivitetin tënd me Meta."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "apps_and_websites_off_of_facebook/your_activity_off_meta_technologies.json")


def connected_apps_and_websites_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is an app or website connected to the participant's Facebook account.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is an app or website connected to the participant's Facebook account.",
          "source_file": "apps_and_websites_off_of_facebook/connected_apps_and_websites.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_connected_apps_and_websites",
          "title": {
            "en": "Connected apps and websites",
            "nl": "Gekoppelde apps en websites",
            "de": "Verbundene Apps und Websites",
            "pl": "Połączone aplikacje i witryny",
            "tr": "Bağlı uygulamalar ve web siteleri",
            "ar": "التطبيقات والمواقع المتصلة",
            "ru": "Подключённые приложения и сайты",
            "it": "App e siti web collegati",
            "ro": "Aplicații și site-uri conectate",
            "es": "Aplicaciones y sitios web conectados",
            "sq": "Aplikacionet dhe faqet e lidhura"
          },
          "description": {
            "en": "This table shows the apps and websites connected to your Facebook account.",
            "nl": "Deze tabel toont de apps en websites die aan je Facebook-account zijn gekoppeld.",
            "de": "Diese Tabelle zeigt die Apps und Websites, die mit Ihrem Facebook-Konto verbunden sind.",
            "pl": "Ta tabela pokazuje aplikacje i witryny połączone z Twoim kontem na Facebooku.",
            "tr": "Bu tablo, Facebook hesabına bağlı uygulamaları ve web sitelerini gösterir.",
            "ar": "يعرض هذا الجدول التطبيقات والمواقع المتصلة بحسابك على فيسبوك.",
            "ru": "В этой таблице показаны приложения и сайты, подключённые к вашему аккаунту Facebook.",
            "it": "Questa tabella mostra le app e i siti web collegati al tuo account Facebook.",
            "ro": "Acest tabel arată aplicațiile și site-urile conectate la contul tău de Facebook.",
            "es": "Esta tabla muestra las aplicaciones y sitios web conectados a tu cuenta de Facebook.",
            "sq": "Kjo tabelë tregon aplikacionet dhe faqet e lidhura me llogarinë tënde në Facebook."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "apps_and_websites_off_of_facebook/connected_apps_and_websites.json")


def snooze_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a profile, page or group the participant snoozed in their feed.

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
          "summary": "Each row is a profile, page or group the participant snoozed in their feed.",
          "source_file": "preferences/feed/snooze.json",
          "columns": {
            "Name": "Name of the person, page or profile.",
            "Date": "ISO 8601 timestamp of the entry. Exact duplicates are listed once."
          }
        }

    Table config::

        {
          "id": "facebook_snooze",
          "title": {
            "en": "Snoozed in your feed",
            "nl": "Gesnoozed in je feed",
            "de": "Im Feed pausiert",
            "pl": "Wyciszone w aktualnościach",
            "tr": "Akışında ertelenenler",
            "ar": "ما أوقفته مؤقتًا في آخر الأخبار",
            "ru": "Временно скрытые в ленте",
            "it": "Silenziati nel feed",
            "ro": "Amânate în flux",
            "es": "Pausados en tu sección de noticias",
            "sq": "Të pezulluara në feed"
          },
          "description": {
            "en": "This table shows profiles, pages and groups you temporarily snoozed in your Facebook feed.",
            "nl": "Deze tabel toont profielen, pagina's en groepen die je tijdelijk hebt gesnoozed in je Facebook-feed.",
            "de": "Diese Tabelle zeigt Profile, Seiten und Gruppen, die Sie in Ihrem Facebook-Feed vorübergehend pausiert haben.",
            "pl": "Ta tabela pokazuje profile, strony i grupy, które tymczasowo wyciszyłeś/aś w aktualnościach.",
            "tr": "Bu tablo, Facebook akışında geçici olarak ertelediğin profilleri, sayfaları ve grupları gösterir.",
            "ar": "يعرض هذا الجدول الملفات والصفحات والمجموعات التي أوقفتها مؤقتًا في آخر الأخبار.",
            "ru": "В этой таблице показаны профили, страницы и группы, которые вы временно скрыли в ленте Facebook.",
            "it": "Questa tabella mostra profili, pagine e gruppi che hai silenziato temporaneamente nel feed di Facebook.",
            "ro": "Acest tabel arată profilurile, paginile și grupurile pe care le-ai amânat temporar în fluxul Facebook.",
            "es": "Esta tabla muestra los perfiles, páginas y grupos que pausaste temporalmente en tu sección de noticias.",
            "sq": "Kjo tabelë tregon profilet, faqet dhe grupet që ke pezulluar përkohësisht në feed-in e Facebook."
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
    return _label_pairs_df(reader, errors, "preferences/feed/snooze.json")


def unfollowed_profiles_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a profile the participant unfollowed in their feed.

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
          "summary": "Each row is a profile the participant unfollowed in their feed.",
          "source_file": "preferences/feed/unfollowed_profiles.json",
          "columns": {
            "Name": "Name of the person, page or profile.",
            "Date": "ISO 8601 timestamp of the entry. Exact duplicates are listed once."
          }
        }

    Table config::

        {
          "id": "facebook_unfollowed_profiles",
          "title": {
            "en": "Profiles you unfollowed",
            "nl": "Ontvolgde profielen",
            "de": "Nicht mehr abonnierte Profile",
            "pl": "Profile, których już nie obserwujesz",
            "tr": "Takibi bıraktığın profiller",
            "ar": "الملفات التي ألغيت متابعتها",
            "ru": "Профили, от которых вы отписались",
            "it": "Profili che non segui più",
            "ro": "Profiluri pe care nu le mai urmărești",
            "es": "Perfiles que dejaste de seguir",
            "sq": "Profilet që nuk i ndjek më"
          },
          "description": {
            "en": "This table shows the profiles you unfollowed in your Facebook feed.",
            "nl": "Deze tabel toont de profielen die je in je Facebook-feed hebt ontvolgd.",
            "de": "Diese Tabelle zeigt die Profile, die Sie in Ihrem Facebook-Feed nicht mehr abonniert haben.",
            "pl": "Ta tabela pokazuje profile, których przestałeś/aś obserwować w aktualnościach.",
            "tr": "Bu tablo, Facebook akışında takibi bıraktığın profilleri gösterir.",
            "ar": "يعرض هذا الجدول الملفات التي ألغيت متابعتها في آخر الأخبار.",
            "ru": "В этой таблице показаны профили, от которых вы отписались в ленте Facebook.",
            "it": "Questa tabella mostra i profili che hai smesso di seguire nel feed di Facebook.",
            "ro": "Acest tabel arată profilurile pe care nu le mai urmărești în fluxul Facebook.",
            "es": "Esta tabla muestra los perfiles que dejaste de seguir en tu sección de noticias.",
            "sq": "Kjo tabelë tregon profilet që ke ndaluar së ndjekuri në feed-in e Facebook."
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
    return _label_pairs_df(reader, errors, "preferences/feed/unfollowed_profiles.json")


def memories_activity_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is an activity of the participant in Facebook Memories.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is an activity of the participant in Facebook Memories.",
          "source_file": "preferences/memories/your_activity.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_memories_activity",
          "title": {
            "en": "Your activity in Memories",
            "nl": "Je activiteit in Herinneringen",
            "de": "Ihre Aktivität in Erinnerungen",
            "pl": "Twoja aktywność we Wspomnieniach",
            "tr": "Anılar'daki etkinliğin",
            "ar": "نشاطك في الذكريات",
            "ru": "Ваша активность в Воспоминаниях",
            "it": "La tua attività nei Ricordi",
            "ro": "Activitatea ta în Amintiri",
            "es": "Tu actividad en Recuerdos",
            "sq": "Aktiviteti yt te Kujtimet"
          },
          "description": {
            "en": "This table shows your activity in Facebook Memories.",
            "nl": "Deze tabel toont je activiteit in Facebook Herinneringen.",
            "de": "Diese Tabelle zeigt Ihre Aktivität in den Facebook-Erinnerungen.",
            "pl": "Ta tabela pokazuje Twoją aktywność we Wspomnieniach na Facebooku.",
            "tr": "Bu tablo, Facebook Anılar'daki etkinliğini gösterir.",
            "ar": "يعرض هذا الجدول نشاطك في ذكريات فيسبوك.",
            "ru": "В этой таблице показана ваша активность в Воспоминаниях Facebook.",
            "it": "Questa tabella mostra la tua attività nei Ricordi di Facebook.",
            "ro": "Acest tabel arată activitatea ta în Amintirile Facebook.",
            "es": "Esta tabla muestra tu actividad en los Recuerdos de Facebook.",
            "sq": "Kjo tabelë tregon aktivitetin tënd te Kujtimet e Facebook."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "preferences/memories/your_activity.json")


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


def time_spent_watching_facebook_shows_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Time the participant spent watching Facebook shows.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Time the participant spent watching Facebook shows.",
          "source_file": "logged_information/interactions/time_spent_watching_facebook_shows.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_time_spent_watching_facebook_shows",
          "title": {
            "en": "Time spent watching Facebook shows",
            "nl": "Tijd besteed aan Facebook-shows",
            "de": "Zeit mit Facebook-Shows",
            "pl": "Czas oglądania programów na Facebooku",
            "tr": "Facebook şovlarını izlemeye harcanan süre",
            "ar": "الوقت المستغرق في مشاهدة عروض فيسبوك",
            "ru": "Время просмотра шоу на Facebook",
            "it": "Tempo trascorso a guardare show su Facebook",
            "ro": "Timp petrecut vizionând emisiuni Facebook",
            "es": "Tiempo viendo programas de Facebook",
            "sq": "Koha e kaluar duke parë emisione në Facebook"
          },
          "description": {
            "en": "This table shows how much time you spent watching shows on Facebook.",
            "nl": "Deze tabel toont hoeveel tijd je hebt besteed aan het bekijken van shows op Facebook.",
            "de": "Diese Tabelle zeigt, wie viel Zeit Sie mit Shows auf Facebook verbracht haben.",
            "pl": "Ta tabela pokazuje, ile czasu spędziłeś/aś na oglądaniu programów na Facebooku.",
            "tr": "Bu tablo, Facebook'ta şov izlemeye ne kadar zaman harcadığını gösterir.",
            "ar": "يعرض هذا الجدول الوقت الذي قضيته في مشاهدة العروض على فيسبوك.",
            "ru": "В этой таблице показано, сколько времени вы провели за просмотром шоу на Facebook.",
            "it": "Questa tabella mostra quanto tempo hai trascorso a guardare show su Facebook.",
            "ro": "Acest tabel arată cât timp ai petrecut vizionând emisiuni pe Facebook.",
            "es": "Esta tabla muestra cuánto tiempo pasaste viendo programas en Facebook.",
            "sq": "Kjo tabelë tregon sa kohë ke kaluar duke parë emisione në Facebook."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "logged_information/interactions/time_spent_watching_facebook_shows.json")


def milestone_notifications_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Each row is a milestone notification Facebook sent to the participant.

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
        Columns: ``Date``, ``Details``.
        Empty DataFrame when the file is absent or contains no records.

    Table documentation::

        {
          "summary": "Each row is a milestone notification Facebook sent to the participant.",
          "source_file": "logged_information/activity_messages/milestone_notifications.json",
          "columns": {
            "Date": "ISO 8601 timestamp of the entry (first timestamp found; empty when not available).",
            "Details": "All other fields of the entry as 'label: value' pairs joined with '; '. Lists inside a record are split so that every list element is its own row. Values that look like IP addresses, e-mail addresses, phone numbers, cookies or device identifiers are removed."
          }
        }

    Table config::

        {
          "id": "facebook_milestone_notifications",
          "title": {
            "en": "Milestone notifications",
            "nl": "Mijlpaalmeldingen",
            "de": "Meilenstein-Benachrichtigungen",
            "pl": "Powiadomienia o kamieniach milowych",
            "tr": "Dönüm noktası bildirimleri",
            "ar": "إشعارات المحطات المهمة",
            "ru": "Уведомления о достижениях",
            "it": "Notifiche sui traguardi",
            "ro": "Notificări despre repere",
            "es": "Notificaciones de hitos",
            "sq": "Njoftimet për arritjet"
          },
          "description": {
            "en": "This table shows the milestone notifications Facebook sent you.",
            "nl": "Deze tabel toont de mijlpaalmeldingen die Facebook je heeft gestuurd.",
            "de": "Diese Tabelle zeigt die Meilenstein-Benachrichtigungen, die Facebook Ihnen gesendet hat.",
            "pl": "Ta tabela pokazuje powiadomienia o kamieniach milowych, które wysłał Ci Facebook.",
            "tr": "Bu tablo, Facebook'un sana gönderdiği dönüm noktası bildirimlerini gösterir.",
            "ar": "يعرض هذا الجدول إشعارات المحطات المهمة التي أرسلها لك فيسبوك.",
            "ru": "В этой таблице показаны уведомления о достижениях, которые вам отправлял Facebook.",
            "it": "Questa tabella mostra le notifiche sui traguardi che Facebook ti ha inviato.",
            "ro": "Acest tabel arată notificările despre repere trimise de Facebook.",
            "es": "Esta tabla muestra las notificaciones de hitos que Facebook te envió.",
            "sq": "Kjo tabelë tregon njoftimet për arritjet që të ka dërguar Facebook."
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
            },
            "Details": {
              "en": "Details",
              "nl": "Details",
              "de": "Details",
              "pl": "Szczegóły",
              "tr": "Ayrıntılar",
              "ar": "التفاصيل",
              "ru": "Подробности",
              "it": "Dettagli",
              "ro": "Detalii",
              "es": "Detalles",
              "sq": "Detajet"
            }
          }
        }
    """
    return _records_df(reader, errors, "logged_information/activity_messages/milestone_notifications.json")


# ---------------------------------------------------------------------------
# Extractor registry & platform info
# ---------------------------------------------------------------------------

#: Mapping from the string names used in port_config.json to actual extractor functions.
EXTRACTOR_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "who_youve_followed_to_df": who_youve_followed_to_df,
    "facebook_reels_usage_to_df": facebook_reels_usage_to_df,
    "video_consumption_summary_to_df": video_consumption_summary_to_df,
    "last_28_days_to_df": last_28_days_to_df,
    "time_spent_on_facebook_to_df": time_spent_on_facebook_to_df,
    "your_search_history_to_df": your_search_history_to_df,
    "your_friends_to_df": your_friends_to_df,
    "ads_interests_to_df": ads_interests_to_df,
    "other_categories_used_to_reach_you_to_df": other_categories_used_to_reach_you_to_df,
    "advertisers_using_your_information_to_df": advertisers_using_your_information_to_df,
    "advertisers_interacted_with_to_df": advertisers_interacted_with_to_df,
    "ads_viewed_to_df": ads_viewed_to_df,
    "content_shown_in_feed_to_df": content_shown_in_feed_to_df,
    "recently_viewed_to_df": recently_viewed_to_df,
    "recently_visited_to_df": recently_visited_to_df,
    "pages_youve_liked_to_df": pages_youve_liked_to_df,
    "your_saved_items_to_df": your_saved_items_to_df,
    "comments_to_df": comments_to_df,
    "your_comment_active_days_to_df": your_comment_active_days_to_df,
    "story_reactions_to_df": story_reactions_to_df,
    "likes_and_reactions_base_to_df": likes_and_reactions_base_to_df,
    "controls_to_df": controls_to_df,
    "ad_settings_to_df": ad_settings_to_df,
    "preference_settings_to_df": preference_settings_to_df,
    "security_and_login_events_to_df": security_and_login_events_to_df,
    "privacy_settings_to_df": privacy_settings_to_df,
    "consents_to_df": consents_to_df,
    "location_and_time_zone_to_df": location_and_time_zone_to_df,
    "active_days_to_df": active_days_to_df,
    "profile_visits_to_df": profile_visits_to_df,
    "link_history_to_df": link_history_to_df,
    "fundraiser_posts_to_df": fundraiser_posts_to_df,
    "friend_suggestions_to_df": friend_suggestions_to_df,
    "groups_and_events_youve_visited_to_df": groups_and_events_youve_visited_to_df,
    "your_comment_edits_to_df": your_comment_edits_to_df,
    "stories_you_reacted_to_with_a_sticker_to_df": stories_you_reacted_to_with_a_sticker_to_df,
    "your_educational_notification_interactions_to_df": your_educational_notification_interactions_to_df,
    "received_friend_requests_to_df": received_friend_requests_to_df,
    "sent_friend_requests_to_df": sent_friend_requests_to_df,
    "rejected_friend_requests_to_df": rejected_friend_requests_to_df,
    "removed_friends_to_df": removed_friends_to_df,
    "your_post_audiences_to_df": your_post_audiences_to_df,
    "locations_of_interest_to_df": locations_of_interest_to_df,
    "apps_detected_from_your_activity_to_df": apps_detected_from_your_activity_to_df,
    "apps_detected_on_your_account_to_df": apps_detected_on_your_account_to_df,
    "ads_about_meta_to_df": ads_about_meta_to_df,
    "meta_ad_library_profiles_to_df": meta_ad_library_profiles_to_df,
    "off_meta_activity_to_df": off_meta_activity_to_df,
    "connected_apps_and_websites_to_df": connected_apps_and_websites_to_df,
    "snooze_to_df": snooze_to_df,
    "unfollowed_profiles_to_df": unfollowed_profiles_to_df,
    "memories_activity_to_df": memories_activity_to_df,
    "story_views_in_past_7_days_to_df": story_views_in_past_7_days_to_df,
    "time_spent_watching_facebook_shows_to_df": time_spent_watching_facebook_shows_to_df,
    "milestone_notifications_to_df": milestone_notifications_to_df,
}


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