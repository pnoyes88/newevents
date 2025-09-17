"""Streamlit app for surfacing upcoming weekend events via open web searches.

The app geocodes a town/state, determines the upcoming weekend window in the
local timezone, and then uses DuckDuckGo web searches to collect recent event
listings from popular event sites, local news outlets, and municipal calendars.
Users can emphasize specific categories, times of day, and focus keywords to
steer the search queries.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from io import StringIO
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

import requests
import streamlit as st
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:  # Optional dependency used for web search
    from duckduckgo_search import DDGS
except ImportError:  # pragma: no cover - handled gracefully at runtime
    DDGS = None  # type: ignore[assignment]


st.set_page_config(
    page_title="Weekend Event Finder",
    page_icon="🎟️",
    layout="wide",
)


@dataclass
class LocationResult:
    """Dataclass capturing the outcome of a geocoding request."""

    latitude: float
    longitude: float
    display_name: str
    timezone: str


@dataclass
class SearchResult:
    """Normalized representation of a web search hit."""

    title: str
    url: str
    snippet: str
    source_label: str
    domain: str
    query: str
    query_index: int
    rank: int


TIME_OF_DAY_KEYWORDS: Dict[str, List[str]] = {
    "Any": [],
    "Early Morning": ["early morning"],
    "Morning": ["morning"],
    "Afternoon": ["afternoon"],
    "Evening": ["evening"],
    "Late Night": ["night", "late night"],
}


CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Any": [],
    "Live Music & Concerts": ["concert", "live music"],
    "Festivals & Fairs": ["festival", "fair"],
    "Family & Kids": ["family", "kids"],
    "Food & Drink": ["food", "brewery", "wine", "tasting"],
    "Sports & Fitness": ["sports", "race", "tournament"],
    "Arts & Culture": ["art", "gallery", "theatre", "museum"],
    "Community & Civic": ["community", "library", "city event"],
    "Nightlife": ["nightlife", "club", "bar"],
    "Outdoor & Nature": ["outdoor", "park", "hike"],
}


FOCUS_KEYWORD_SETS: Dict[str, List[str]] = {
    "Free or low-cost": ["free", "no-cost"],
    "Family friendly": ["family friendly", "all ages"],
    "Pet friendly": ["pet friendly", "dogs welcome"],
    "Accessible": ["accessible", "ADA"],
    "Farmers markets": ["farmers market", "farm market"],
    "Workshops & classes": ["workshop", "class", "lesson"],
    "Volunteering": ["volunteer", "community service"],
}


SEARCH_TARGETS: Dict[str, str] = {
    "General web": "",
    "Eventbrite": "site:eventbrite.com",
    "Meetup": "site:meetup.com",
    "AllEvents": "site:allevents.in",
    "Eventful": "site:eventful.com",
    "EventCrazy": "site:eventcrazy.com",
    "Local news & media": "site:patch.com OR site:timeout.com OR site:newsbreak.com",
    "Chambers & tourism": "site:chamberofcommerce.com OR site:visitcity.com OR site:visitnc.com",
    "Municipal & .gov": "(site:.gov OR site:.org) (events OR calendar)",
    "Facebook events": "\"facebook.com/events\"",
}


DEFAULT_TARGETS: Sequence[str] = (
    "General web",
    "Eventbrite",
    "Meetup",
    "AllEvents",
    "Local news & media",
)


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_location(city: str, state: str) -> Optional[LocationResult]:
    """Resolve a city and state to coordinates and timezone information."""

    query = f"{city}, {state}"
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "WeekendEventFinder/1.0"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    results = response.json()
    if not results:
        return None

    top = results[0]
    latitude = float(top["lat"])
    longitude = float(top["lon"])
    display_name = top.get("display_name", query)

    timezone = fetch_timezone(latitude, longitude)
    return LocationResult(latitude, longitude, display_name, timezone)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_timezone(latitude: float, longitude: float) -> str:
    """Obtain a timezone name for the supplied coordinates."""

    try:
        response = requests.get(
            "https://timeapi.io/api/Time/current/coordinate",
            params={"latitude": latitude, "longitude": longitude},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        timezone = data.get("timeZone")
        if timezone:
            return timezone
    except requests.RequestException:
        pass

    # Fallback to UTC if no timezone could be found
    return "UTC"


def upcoming_weekend_range(tz_name: str) -> Tuple[datetime, datetime, datetime, datetime]:
    """Return the start and end of the upcoming weekend in both local and UTC."""

    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)
    # Calculate the next Saturday
    days_until_saturday = (5 - now.weekday()) % 7
    saturday_date = (now + timedelta(days=days_until_saturday)).date()
    sunday_date = saturday_date + timedelta(days=1)

    local_start = datetime.combine(saturday_date, time.min, tzinfo=tz)
    local_end = datetime.combine(sunday_date, time.max.replace(microsecond=0), tzinfo=tz)

    start_utc = local_start.astimezone(ZoneInfo("UTC"))
    end_utc = local_end.astimezone(ZoneInfo("UTC"))
    return local_start, local_end, start_utc, end_utc


def format_day(dt: datetime) -> str:
    """Return a month/day string without a leading zero."""

    return f"{dt.strftime('%B')} {dt.day}"


def normalize_url(url: str) -> Optional[str]:
    """Normalize URLs for deduplication purposes."""

    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))


def extract_domain(url: str) -> str:
    """Extract the domain portion of a URL."""

    try:
        return urlparse(url).netloc.lower() or url
    except ValueError:
        return url


def parse_extra_terms(raw_terms: str) -> List[str]:
    """Split a comma or semicolon separated string into query-ready terms."""

    if not raw_terms:
        return []
    parts = re.split(r"[;,]", raw_terms)
    return [part.strip() for part in parts if part.strip()]


def build_search_queries(
    city: str,
    state: str,
    weekend_start: datetime,
    weekend_end: datetime,
    category: str,
    time_of_day: str,
    focus_labels: Sequence[str],
    radius_miles: int,
    extra_terms: Sequence[str],
    targets: Sequence[str],
) -> Tuple[str, List[Tuple[str, str]]]:
    """Construct the base search string and a list of per-target queries."""

    base_terms: List[str] = [
        f"{city} {state}",
        "weekend events",
        "things to do",
        f"{weekend_start.strftime('%A')} {format_day(weekend_start)}",
        f"{weekend_end.strftime('%A')} {format_day(weekend_end)}",
        str(weekend_start.year),
    ]

    if weekend_start.month != weekend_end.month:
        base_terms.append(f"{format_day(weekend_start)}-{format_day(weekend_end)}")
    else:
        base_terms.append(f"{weekend_start.day}-{weekend_end.day}")

    if radius_miles:
        base_terms.append(f"within {radius_miles} miles")

    base_terms.extend(CATEGORY_KEYWORDS.get(category, []))
    base_terms.extend(TIME_OF_DAY_KEYWORDS.get(time_of_day, []))

    for label in focus_labels:
        base_terms.extend(FOCUS_KEYWORD_SETS.get(label, []))

    base_terms.extend(extra_terms)

    base_query = " ".join(term for term in base_terms if term)

    queries: List[Tuple[str, str]] = []
    for target in targets:
        modifier = SEARCH_TARGETS.get(target)
        if modifier is None:
            continue
        query = base_query if not modifier else f"{base_query} {modifier}".strip()
        queries.append((target, query))

    return base_query, queries


@st.cache_data(ttl=600)
def run_event_search(
    queries: Sequence[Tuple[str, str]],
    max_results_per_query: int,
) -> Tuple[List[SearchResult], Dict[str, str]]:
    """Execute the prepared queries and collate search hits."""

    if DDGS is None:
        raise RuntimeError(
            "The optional dependency 'duckduckgo_search' is not installed."
        )

    aggregated: List[SearchResult] = []
    errors: Dict[str, str] = {}
    seen_urls: set[str] = set()

    try:
        ddgs = DDGS()
    except Exception as exc:  # pragma: no cover - network/environment issue
        raise RuntimeError(f"Unable to initialize DuckDuckGo search client: {exc}") from exc

    with ddgs:
        for query_index, (label, query) in enumerate(queries):
            try:
                results_iter = ddgs.text(
                    query,
                    region="us-en",
                    safesearch="Moderate",
                    timelimit="w",
                    max_results=max_results_per_query,
                )
            except Exception as exc:  # pragma: no cover - network/environment issue
                errors[label] = str(exc)
                continue

            for rank, item in enumerate(results_iter, start=1):
                url = item.get("href") or item.get("url")
                if not url:
                    continue

                normalized = normalize_url(url)
                if not normalized or normalized in seen_urls:
                    continue
                seen_urls.add(normalized)

                title = (item.get("title") or url).strip()
                snippet = (item.get("body") or "").strip()

                aggregated.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source_label=label,
                        domain=extract_domain(url),
                        query=query,
                        query_index=query_index,
                        rank=rank,
                    )
                )

    aggregated.sort(key=lambda result: (result.query_index, result.rank))
    return aggregated, errors


def render_search_results(results: Iterable[SearchResult]) -> None:
    """Render search hits as rich cards in the Streamlit app."""

    for result in results:
        with st.container():
            st.subheader(result.title)
            if result.snippet:
                st.write(result.snippet)
            else:
                st.write("_No preview available. Open the link for details._")

            st.markdown(
                f"[Open event page ↗️]({result.url})",
                unsafe_allow_html=False,
            )
            st.caption(
                f"{result.domain} • via {result.source_label} • rank #{result.rank}"
            )
        st.divider()


def download_ready_table(records: Sequence[SearchResult]) -> str:
    """Serialize search results as CSV text for download."""

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "Title",
        "Source label",
        "Domain",
        "Rank within query",
        "Query",
        "Snippet",
        "URL",
    ])
    for record in records:
        writer.writerow(
            [
                record.title,
                record.source_label,
                record.domain,
                record.rank,
                record.query,
                record.snippet,
                record.url,
            ]
        )
    return buffer.getvalue()


# --- Streamlit UI ---

def main() -> None:
    st.title("🎟️ Weekend Event Finder")
    st.write(
        "Search across popular event directories, municipal calendars, and local "
        "media to see what's happening this upcoming weekend near your town."
    )

    if DDGS is None:
        st.error(
            "This app relies on the optional package `duckduckgo_search`. Install it "
            "with `pip install duckduckgo_search` and refresh the app."
        )
        return

    with st.sidebar:
        st.header("Search Settings")
        city = st.text_input("Town / City", placeholder="e.g. Asheville")
        state = st.text_input("State or region", placeholder="e.g. North Carolina")
        radius = st.slider(
            "Willing to travel (miles)",
            min_value=5,
            max_value=150,
            value=30,
            step=5,
        )

        time_of_day = st.selectbox(
            "Preferred time of day emphasis",
            list(TIME_OF_DAY_KEYWORDS.keys()),
        )
        category = st.selectbox(
            "Category emphasis",
            list(CATEGORY_KEYWORDS.keys()),
        )
        focus = st.multiselect(
            "Focus keywords",
            list(FOCUS_KEYWORD_SETS.keys()),
        )
        extra_keywords = st.text_input(
            "Additional keywords (comma separated)",
            placeholder="farmers market, jazz, date night",
        )

        target_choices = st.multiselect(
            "Sites to emphasize",
            list(SEARCH_TARGETS.keys()),
            default=list(DEFAULT_TARGETS),
        )
        max_results = st.slider(
            "Results to pull from each site",
            min_value=3,
            max_value=20,
            value=6,
        )

        run_search = st.button("Search upcoming weekend", type="primary")

    city = city.strip()
    state = state.strip()

    if not city or not state:
        st.info("Provide both a town/city and a state to continue.")
        return

    if not target_choices:
        st.warning("Select at least one search target to run queries.")
        return

    if not run_search:
        st.stop()

    with st.spinner("Geocoding location..."):
        location = geocode_location(city, state)
    if not location:
        st.error(
            "We couldn't resolve that location. Please check the spelling or try a nearby town."
        )
        return

    st.success(f"Searching near {location.display_name}")
    st.caption(f"Local timezone detected: {location.timezone}")

    local_start, local_end, _, _ = upcoming_weekend_range(location.timezone)
    st.write(
        f"Upcoming weekend window: **{local_start.strftime('%A %b %d')}** to "
        f"**{local_end.strftime('%A %b %d')}** ({location.timezone})"
    )

    extra_terms = parse_extra_terms(extra_keywords)
    base_query, queries = build_search_queries(
        city=city,
        state=state,
        weekend_start=local_start,
        weekend_end=local_end,
        category=category,
        time_of_day=time_of_day,
        focus_labels=focus,
        radius_miles=radius,
        extra_terms=extra_terms,
        targets=target_choices,
    )

    st.caption("Search terms are tuned to favour recent listings from the past week.")

    with st.expander("View generated search queries"):
        st.code(base_query, language="text")
        for label, query in queries:
            st.markdown(f"**{label}**")
            st.code(query, language="text")

    with st.spinner("Searching the web for event listings..."):
        try:
            results, errors = run_event_search(queries, max_results)
        except RuntimeError as exc:
            st.error(str(exc))
            return

    if errors:
        for label, message in errors.items():
            st.warning(f"{label}: {message}")

    if not results:
        st.warning("No event listings were found for the selected filters.")
        return

    st.success(f"Found {len(results)} event listings across {len(queries)} search queries.")

    render_search_results(results)

    csv_data = download_ready_table(results)
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="weekend-event-search-results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
