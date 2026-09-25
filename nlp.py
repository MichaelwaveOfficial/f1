import re


def normalize_text(value):
    """Lowercase and replace punctuation with spaces."""
    return " ".join(
        re.findall(r"[a-z0-9]+", value.casefold())
    )


def contains_phrase(text, phrase):
    """Match a normalized whole phrase, not arbitrary characters."""
    return f" {phrase} " in f" {text} "


def resolve_team(car_model, driver_db):

    normalized_model = normalize_text(car_model)

    # Include database keys themselves as aliases.
    candidates = []

    for team in driver_db:

        aliases = TEAM_ALIASES.get(team, [])
        aliases = [team, *aliases]

        for alias in aliases:

            normalized_alias = normalize_text(alias)
            candidates.append((
                normalized_alias,
                team,
            ))

    # Prefer longer, more specific aliases.
    candidates.sort(
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, team in candidates:

        if contains_phrase(normalized_model, alias):
            return team

    return None
