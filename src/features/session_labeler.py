from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo('America/New_York')

PREMARKET_START = time(4, 0)
REGULAR_START = time(9, 30)
REGULAR_END = time(16, 0)
AFTERHOURS_END = time(20, 0)


def _to_datetime(value):
    if isinstance(value, datetime):
        return value

    if hasattr(value, 'to_pydatetime'):
        return value.to_pydatetime()

    return datetime.fromtimestamp(int(value), tz=EASTERN)


def to_eastern_datetime(value):
    dt = _to_datetime(value)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=EASTERN)

    return dt.astimezone(EASTERN)


def session_from_datetime(dt):
    if dt.weekday() >= 5:
        return 'closed'

    t = dt.time()

    if PREMARKET_START <= t < REGULAR_START:
        return 'premarket'

    if REGULAR_START <= t < REGULAR_END:
        return 'regular'

    if REGULAR_END <= t < AFTERHOURS_END:
        return 'afterhours'

    return 'closed'


def session_from_time(value):
    dt = to_eastern_datetime(value)
    return session_from_datetime(dt)


def _next_weekday(d):
    x = d + timedelta(days=1)
    while x.weekday() >= 5:
        x = x + timedelta(days=1)
    return x


def regular_close_datetime(event_dt):
    """16:00 ET close for this session, or the next session if afterhours/weekend."""
    dt = to_eastern_datetime(event_dt)
    session = session_from_datetime(dt)

    if session in ('premarket', 'regular'):
        day = dt.date()
    elif session == 'afterhours':
        day = _next_weekday(dt.date())
    elif dt.weekday() < 5 and dt.time() < PREMARKET_START:
        day = dt.date()
    else:
        day = _next_weekday(dt.date())

    return datetime(day.year, day.month, day.day, 16, 0, tzinfo=EASTERN)
