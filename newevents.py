"""Streamlit app for discovering upcoming weekend events near a town and state.

The app relies on the Ticketmaster Discovery API and OpenStreetMap's Nominatim
service to gather event and geolocation data. Users must supply their own
Ticketmaster API key within the interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import streamlit as st
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
class EventRecord:
    """Normalized event information for easy display and export."""

    name: str
    start_local: Optional[datetime]
    venue: str
    address: str
    category: str
    price: str
    distance_miles: Optional[float]
    url: str
    source: str


TIME_OF_DAY_WINDOWS: Dict[str, Tuple[time, time]] = {
    "Any": (time(0, 0), time(23, 59, 59)),
    "Early Morning": (time(5, 0), time(8, 59, 59)),
    "Morning": (time(9, 0), time(11, 59, 59)),
    "Afternoon": (time(12, 0), time(16, 59, 59)),
    "Evening": (time(17, 0), time(20, 59, 59)),
    "Night": (time(21, 0), time(23, 59, 59)),
}


CLASSIFICATION_OPTIONS = [
    "Any",
    "Music",
    "Sports",
    "Arts & Theatre",
    "Film",
    "Miscellaneous",
    "Family",
    "Festivals",
    "Hobbies",
    "Food & Drink",
]


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


@st.cache_data(ttl=900)
def query_ticketmaster(
    api_key: str,
    lat: float,
    lon: float,
    radius_miles: int,
    classification: Optional[str],
    keyword: Optional[str],
    start_utc: datetime,
    end_utc: datetime,
) -> Dict:
    """Query the Ticketmaster Discovery API for events."""

    params = {
        "apikey": api_key,
        "latlong": f"{lat},{lon}",
        "radius": radius_miles,
        "unit": "miles",
        "locale": "*",
        "size": 200,
        "sort": "date,asc",
        "startDateTime": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDateTime": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if classification and classification != "Any":
        params["classificationName"] = classification
    if keyword:
        params["keyword"] = keyword

    response = requests.get(
        "https://app.ticketmaster.com/discovery/v2/events.json",
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def parse_tm_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime strings returned by the Ticketmaster API."""

    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def within_time_window(check_time: time, window: Tuple[time, time]) -> bool:
    """Check whether a time is within a start/end window."""

    start, end = window
    if start <= end:
        return start <= check_time <= end
    # Handle overnight windows gracefully
    return check_time >= start or check_time <= end


def classify_price(event: Dict) -> Tuple[str, Optional[float]]:
    """Derive a readable price description and minimum price."""

    price_ranges = event.get("priceRanges") or []
    if not price_ranges:
        if event.get("promoter") and event["promoter"].get("name"):
            return "See site", None
        return "Not listed", None

    min_prices = []
    formatted_ranges = []
    for price in price_ranges:
        try:
            min_price = float(price.get("min"))
            max_price = float(price.get("max"))
            currency = price.get("currency", "USD")
        except (TypeError, ValueError):
            continue
        min_prices.append(min_price)
        if math.isclose(min_price, max_price):
            formatted_ranges.append(f"{min_price:.2f} {currency}")
        else:
            formatted_ranges.append(f"{min_price:.2f}-{max_price:.2f} {currency}")

    if not formatted_ranges:
        return "Not listed", None

    return ", ".join(formatted_ranges), min(min_prices) if min_prices else None


def transform_events(
    events_payload: Dict,
    tz_name: str,
    time_window: Tuple[time, time],
    include_free: bool,
    include_paid: bool,
    max_price: Optional[float],
    include_online: bool,
    include_in_person: bool,
) -> List[EventRecord]:
    """Normalize raw Ticketmaster results and apply filters."""

    embedded = events_payload.get("_embedded", {})
    events = embedded.get("events", [])

    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")

    normalized: List[EventRecord] = []

    for event in events:
        start_data = event.get("dates", {}).get("start", {})
        start_dt = parse_tm_datetime(start_data.get("dateTime"))
        if start_dt is None:
            continue
        local_start = start_dt.astimezone(tz)

        if not within_time_window(local_start.time(), time_window):
            continue

        classification_names = [
            class_item.get("name")
            for class_item in event.get("classifications", [])
            if class_item.get("name")
        ]
        category = classification_names[0] if classification_names else "Uncategorized"

        price_text, min_price = classify_price(event)
        if not include_free and (min_price is None or math.isclose(min_price, 0.0, abs_tol=0.01)):
            continue
        if not include_paid and min_price and min_price > 0.0:
            continue
        if max_price is not None and min_price is not None and min_price > max_price:
            continue

        venues = event.get("_embedded", {}).get("venues", [])
        venue_name = venues[0].get("name") if venues else "Venue TBA"
        address_parts = []
        if venues:
            venue = venues[0]
            for key in ["address", "city", "state", "country"]:
                piece = venue.get(key)
                if isinstance(piece, dict):
                    name = piece.get("name") or piece.get("line1") or piece.get("line2")
                    if name:
                        address_parts.append(name)
                elif isinstance(piece, str):
                    address_parts.append(piece)
        address = ", ".join(dict.fromkeys(filter(None, address_parts))) or "Address TBA"

        is_virtual = any(venue.get("type") == "venue" and venue.get("name", "").lower() == "virtual venue" for venue in venues)
        if is_virtual and not include_online:
            continue
        if not is_virtual and not include_in_person:
            continue

        distance_miles = None
        if event.get("distance") is not None:
            try:
                distance_miles = float(event["distance"])
            except (TypeError, ValueError):
                distance_miles = None

        normalized.append(
            EventRecord(
                name=event.get("name", "Unnamed event"),
                start_local=local_start,
                venue=venue_name,
                address=address,
                category=category,
                price=price_text,
                distance_miles=distance_miles,
                url=event.get("url", ""),
                source="Ticketmaster",
            )
        )

    normalized.sort(key=lambda record: (record.start_local or datetime.max))
    return normalized


def render_event_cards(records: Iterable[EventRecord]) -> None:
    """Render event details as cards in the main area."""

    for record in records:
        with st.container():
            st.subheader(record.name)
            columns = st.columns([2, 1])
            with columns[0]:
                if record.start_local:
                    st.write(
                        f"**When:** {record.start_local.strftime('%A, %B %d %Y at %I:%M %p')}"
                    )
                st.write(f"**Where:** {record.venue}")
                st.write(f"**Address:** {record.address}")
                st.write(f"**Category:** {record.category}")
                st.write(f"**Price:** {record.price}")
                if record.distance_miles is not None:
                    st.write(f"**Approx. distance:** {record.distance_miles:.1f} miles")
            with columns[1]:
                if record.url:
                    st.markdown(
                        f"[View details ↗️]({record.url})",
                        unsafe_allow_html=False,
                    )
                st.caption(f"Source: {record.source}")
        st.divider()


def download_ready_table(records: Iterable[EventRecord]) -> str:
    """Serialize event records as CSV text for download."""

    headers = [
        "Name",
        "Local Start",
        "Venue",
        "Address",
        "Category",
        "Price",
        "Distance (miles)",
        "URL",
        "Source",
    ]
    lines = [",".join(headers)]
    for record in records:
        local_start = (
            record.start_local.strftime("%Y-%m-%d %H:%M") if record.start_local else ""
        )
        distance = f"{record.distance_miles:.2f}" if record.distance_miles is not None else ""
        row = [
            record.name.replace(",", " "),
            local_start,
            record.venue.replace(",", " "),
            record.address.replace(",", " "),
            record.category.replace(",", " "),
            record.price.replace(",", ";"),
            distance,
            record.url,
            record.source,
        ]
        lines.append(",".join(row))
    return "\n".join(lines)


# --- Streamlit UI ---

def main() -> None:
    st.title("🎟️ Weekend Event Finder")
    st.write(
        "Discover concerts, sports, food festivals, and more happening this weekend "
        "near your town. Provide a Ticketmaster API key to search the Discovery API "
        "for upcoming events."
    )

    with st.sidebar:
        st.header("Search Settings")
        api_key = st.text_input("Ticketmaster API key", type="password")
        st.caption(
            "Get a free key from https://developer.ticketmaster.com/. It is stored only "
            "in your current session."
        )

        city = st.text_input("Town / City", placeholder="e.g. Asheville")
        state = st.text_input("State or region", placeholder="e.g. North Carolina")
        keyword = st.text_input("Keyword (optional)", placeholder="music, comedy, art...")

        time_of_day = st.selectbox("Preferred time of day", list(TIME_OF_DAY_WINDOWS.keys()))
        classification = st.selectbox("Category", CLASSIFICATION_OPTIONS)
        radius = st.slider("Search radius (miles)", min_value=5, max_value=150, value=30, step=5)

        include_online = st.checkbox("Include online / virtual events", value=True)
        include_in_person = st.checkbox("Include in-person events", value=True)

        filter_free = st.checkbox("Show free events", value=True)
        filter_paid = st.checkbox("Show paid events", value=True)
        max_price = st.slider(
            "Maximum minimum ticket price",
            min_value=0,
            max_value=500,
            value=500,
            step=5,
            help="Filter out events whose lowest listed price exceeds this amount.",
        )
        if max_price == 500:
            max_price_value: Optional[float] = None
        else:
            max_price_value = float(max_price)

        run_search = st.button("Search for weekend events", type="primary")

    if not api_key:
        st.info("Enter a valid Ticketmaster API key in the sidebar to start searching.")
        return

    if not city or not state:
        st.info("Provide both a town/city and a state to continue.")
        return

    if not (include_online or include_in_person):
        st.warning("Enable at least one of online or in-person events to see results.")
        return

    if not (filter_free or filter_paid):
        st.warning("Enable at least one of free or paid events to see results.")
        return

    if not run_search:
        st.stop()

    with st.spinner("Geocoding location..."):
        location = geocode_location(city.strip(), state.strip())
    if not location:
        st.error(
            "We couldn't resolve that location. Please check the spelling or try a nearby city."
        )
        return

    st.success(f"Searching within {radius} miles of {location.display_name}")
    st.caption(f"Local timezone detected: {location.timezone}")

    local_start, local_end, start_utc, end_utc = upcoming_weekend_range(location.timezone)
    st.write(
        f"Upcoming weekend window: **{local_start.strftime('%A %b %d')}** to "
        f"**{local_end.strftime('%A %b %d')}** ({location.timezone})"
    )

    try:
        with st.spinner("Contacting Ticketmaster..."):
            raw_results = query_ticketmaster(
                api_key=api_key,
                lat=location.latitude,
                lon=location.longitude,
                radius_miles=radius,
                classification=classification,
                keyword=keyword.strip() if keyword else None,
                start_utc=start_utc,
                end_utc=end_utc,
            )
    except requests.HTTPError as exc:
        st.error(
            "Ticketmaster returned an error. Double-check your API key and try again.\n"
            f"Details: {exc}"
        )
        return
    except requests.RequestException as exc:
        st.error(f"We couldn't reach Ticketmaster right now: {exc}")
        return

    time_window = TIME_OF_DAY_WINDOWS.get(time_of_day, TIME_OF_DAY_WINDOWS["Any"])

    filtered_events = transform_events(
        events_payload=raw_results,
        tz_name=location.timezone,
        time_window=time_window,
        include_free=filter_free,
        include_paid=filter_paid,
        max_price=max_price_value,
        include_online=include_online,
        include_in_person=include_in_person,
    )

    if not filtered_events:
        st.warning("No events matched your filters for the upcoming weekend.")
        return

    st.success(f"Found {len(filtered_events)} matching events!")

    render_event_cards(filtered_events)

    csv_data = download_ready_table(filtered_events)
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="weekend-events.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
