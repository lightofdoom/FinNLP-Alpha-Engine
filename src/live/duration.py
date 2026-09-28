"""Parse session end times. Naive inputs are read as US market time (America/New_York)."""

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo('America/New_York')

DATETIME_FORMATS = (
    '%m/%d/%y %I:%M %p',
    '%m/%d/%Y %I:%M %p',
    '%m/%d/%y %I:%M:%S %p',
    '%m/%d/%Y %I:%M:%S %p',
    '%m/%d/%y %H:%M',
    '%m/%d/%Y %H:%M',
    '%Y-%m-%d %I:%M %p',
    '%Y-%m-%d %H:%M',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%dT%H:%M',
    '%Y-%m-%dT%H:%M:%S',
    '%m/%d/%y',
    '%m/%d/%Y',
    '%Y-%m-%d',
)

TIME_ONLY_FORMATS = (
    '%I:%M %p',
    '%I:%M:%S %p',
    '%H:%M',
    '%H:%M:%S',
)

DURATION_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*([smhd])', re.IGNORECASE)

UNIT_SECONDS = {
    's': 1.0,
    'm': 60.0,
    'h': 3600.0,
    'd': 86400.0,
}


def parse_duration(text):
    """'90m', '2h', '1h30m', '45s', '1d' -> timedelta."""
    if text is None:
        return None
    raw = str(text).strip().lower()
    if raw == '':
        return None

    matches = DURATION_PATTERN.findall(raw)
    if matches:
        stripped = DURATION_PATTERN.sub('', raw).replace(' ', '')
        if stripped != '':
            raise ValueError('Could not parse duration: %s' % text)
        total = 0.0
        for amount, unit in matches:
            total += float(amount) * UNIT_SECONDS[unit.lower()]
        if total <= 0:
            raise ValueError('Duration must be positive: %s' % text)
        return timedelta(seconds=total)

    try:
        minutes = float(raw)
    except ValueError:
        raise ValueError('Could not parse duration: %s' % text)
    if minutes <= 0:
        raise ValueError('Duration must be positive: %s' % text)
    return timedelta(minutes=minutes)


def parse_until(text, now=None, tz=EASTERN):
    """'8/25/26 12:47 AM', '2026-08-25 00:47', or a bare time -> aware datetime."""
    if text is None:
        return None
    raw = str(text).strip()
    if raw == '':
        return None

    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=tz)

    try:
        parsed = datetime.fromisoformat(raw)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=tz)
    except ValueError:
        pass

    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=tz)
        except ValueError:
            continue

    for fmt in TIME_ONLY_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        candidate = now.replace(
            hour=parsed.hour,
            minute=parsed.minute,
            second=parsed.second,
            microsecond=0,
        )
        if candidate <= now:
            candidate = candidate + timedelta(days=1)
        return candidate

    raise ValueError(
        'Could not parse end time: %s (try "8/25/26 12:47 AM" or "2026-08-25 00:47")' % text
    )


def resolve_end_time(until=None, duration=None, now=None, tz=EASTERN):
    """until wins over duration. Returns None for run-until-interrupted."""
    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=tz)

    if until is not None and str(until).strip() != '':
        end = parse_until(until, now=now, tz=tz)
        if end <= now:
            raise ValueError('End time %s is already in the past' % end.isoformat())
        return end

    span = parse_duration(duration) if duration is not None else None
    if span is not None:
        return now + span

    return None


def format_remaining(delta):
    seconds = int(max(0, delta.total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return '%dh %dm %ds' % (hours, minutes, secs)
    if minutes:
        return '%dm %ds' % (minutes, secs)
    return '%ds' % secs
