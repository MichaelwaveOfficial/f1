from config import settings
from nlp import normalize_text
from utils import load_json


driver_db = load_json(settings.DRIVER_JSON)
team_aliases = load_json(settings.TEAM_ALIASES_JSON)


def canonical_team_name(team):
    if not team:
        return None

    normalized_team = normalize_text(str(team))
    candidates = []
    for canonical_name, aliases in team_aliases.items():
        alias_candidates = [canonical_name, *aliases]
        for alias in alias_candidates:
            normalized_alias = normalize_text(alias)
            if normalized_alias:
                candidates.append((len(normalized_alias), normalized_alias, canonical_name))

    # Classifier labels often include a car/model suffix, e.g. "Ferrari SF-26".
    # Match the longest whole team phrase inside that label.
    for _, alias, canonical_name in sorted(candidates, reverse=True):
        if normalized_team == alias or f" {alias} " in f" {normalized_team} ":
            return canonical_name
    return None


def lookup_driver(team, tcam_color):
    """Resolve the driver from the canonical team and black/yellow T-cam."""
    canonical_team = canonical_team_name(team)
    normalized_tcam = normalize_text(str(tcam_color or "unknown"))
    driver_record = (
        driver_db
        .get(canonical_team, {})
        .get(normalized_tcam)
    )

    if not driver_record:
        return {
            "driver": "Unknown Driver",
            "number": None,
        }

    return {
        "driver": driver_record["driver"],
        "number": driver_record["number"],
    }
