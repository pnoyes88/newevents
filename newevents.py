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
from datetime import date, datetime, time, timedelta
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


WEEKEND_OPTIONS: Dict[str, Tuple[int, int]] = {
    "Saturday & Sunday": (5, 2),
    "Friday evening through Sunday": (4, 3),
    "Long weekend (Friday to Monday)": (4, 4),
}


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_location(city: str, state: str) -> Optional[LocationResult]:
    """Resolve a city and state to coordinates and timezone information."""

    query = f"{city}, {state}"
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
                "extratags": 1,
            },
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

    timezone = (top.get("extratags") or {}).get("timezone")
    if not timezone:
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


def upcoming_weekend_range(
    tz_name: str, start_weekday: int, length_days: int
) -> Tuple[datetime, datetime, List[date]]:
    """Return the start/end datetimes and calendar days for the chosen weekend."""

    if length_days < 1:
        raise ValueError("length_days must be at least 1")

    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)
    days_until_start = (start_weekday - now.weekday()) % 7
    start_date = (now + timedelta(days=days_until_start)).date()

    weekend_dates = [start_date + timedelta(days=offset) for offset in range(length_days)]

    local_start = datetime.combine(weekend_dates[0], time.min, tzinfo=tz)
    local_end = datetime.combine(
        weekend_dates[-1], time.max.replace(microsecond=0), tzinfo=tz
    )

    return local_start, local_end, weekend_dates


def format_day(value: date) -> str:
    """Return a month/day string without a leading zero."""

    return f"{value.strftime('%B')} {value.day}"


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
    weekend_dates: Sequence[date],
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
    ]

    for day in weekend_dates:
        base_terms.append(f"{day.strftime('%A')} {format_day(day)}")

    if weekend_dates:
        years = sorted({day.year for day in weekend_dates})
        base_terms.extend(str(year) for year in years)

    if weekend_dates:
        first_day = weekend_dates[0]
        last_day = weekend_dates[-1]
        if first_day == last_day:
            base_terms.append(format_day(first_day))
        elif first_day.month != last_day.month:
            base_terms.append(f"{format_day(first_day)}-{format_day(last_day)}")
        else:
            base_terms.append(f"{first_day.day}-{last_day.day}")

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
        ddgs = DDGS(headers={"User-Agent": "WeekendEventFinder/1.0"})
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


def filter_search_results(
    results: Sequence[SearchResult],
    excluded_terms: Sequence[str],
    unique_domains: bool,
) -> List[SearchResult]:
    """Filter results by excluded keywords and deduplicate by domain if requested."""

    lowered_exclusions = [term.lower() for term in excluded_terms if term]
    seen_domains: set[str] = set()
    filtered: List[SearchResult] = []

    for result in results:
        haystack = " ".join([result.title, result.snippet, result.domain]).lower()
        if any(exclusion in haystack for exclusion in lowered_exclusions):
            continue

        if unique_domains:
            domain = result.domain
            if domain in seen_domains:
                continue
            seen_domains.add(domain)

        filtered.append(result)

    return filtered


def render_search_results(results: Iterable[SearchResult]) -> None:
    """Render search hits as rich cards in the Streamlit app."""

    for result in results:
        with st.container():
            st.subheader(result.title)
            st.caption(f"Source: {result.source_label} • Rank #{result.rank}")
            if result.snippet:
                st.write(result.snippet)
            else:
                st.write("_No preview available. Open the link for details._")

            st.markdown(
                f"[Open event page ↗️]({result.url})",
                unsafe_allow_html=False,
            )
            st.caption(f"Domain: {result.domain}")
            with st.expander("See the generated search query"):
                st.code(result.query, language="text")
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

        weekend_choice = st.selectbox(
            "Weekend window",
            list(WEEKEND_OPTIONS.keys()),
            index=1,
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
        excluded_keywords = st.text_input(
            "Exclude keywords (comma separated)",
            placeholder="bingo, virtual, webinar",
        )
        unique_domains = st.checkbox(
            "Limit to one result per website",
            value=True,
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

    start_weekday, weekend_length = WEEKEND_OPTIONS.get(
        weekend_choice, WEEKEND_OPTIONS["Saturday & Sunday"]
    )
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

    _, _, weekend_dates = upcoming_weekend_range(
        location.timezone, start_weekday, weekend_length
    )
    if weekend_dates:
        if len(weekend_dates) == 1:
            weekend_summary = weekend_dates[0].strftime("%A %b %d")
        else:
            start_label = weekend_dates[0].strftime("%A %b %d")
            end_label = weekend_dates[-1].strftime("%A %b %d")
            weekend_summary = f"{start_label} → {end_label}"
        focus_days = ", ".join(
            f"{day.strftime('%A')} ({format_day(day)})" for day in weekend_dates
        )
    else:
        weekend_summary = "Upcoming weekend"
        focus_days = ""

    st.write(
        f"Upcoming weekend window ({weekend_choice}): "
        f"**{weekend_summary}** ({location.timezone})"
    )
    if focus_days:
        st.caption(f"Focus days: {focus_days}")

    extra_terms = parse_extra_terms(extra_keywords)
    excluded_terms = parse_extra_terms(excluded_keywords)
    base_query, queries = build_search_queries(
        city=city,
        state=state,
        weekend_dates=weekend_dates,
        category=category,
        time_of_day=time_of_day,
        focus_labels=focus,
        radius_miles=radius,
        extra_terms=extra_terms,
        targets=target_choices,
    )

    st.caption("Search terms are tuned to favour recent listings from the past week.")
    if excluded_terms:
        st.caption(
            "Results containing these phrases will be skipped: "
            + ", ".join(excluded_terms)
        )

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

    filtered_results = filter_search_results(results, excluded_terms, unique_domains)
    removed_count = len(results) - len(filtered_results)
    if removed_count > 0:
        st.info(
            f"Filtered out {removed_count} results based on excluded keywords or domain limits."
        )

    if not filtered_results:
        st.warning(
            "All retrieved listings were filtered out. Try adjusting the excluded keywords "
            "or allowing multiple results per website."
        )
        return

    st.success(
        f"Surfaced {len(filtered_results)} event listings across {len(queries)} search queries."
    )

    render_search_results(filtered_results)

    csv_data = download_ready_table(filtered_results)
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="weekend-event-search-results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
