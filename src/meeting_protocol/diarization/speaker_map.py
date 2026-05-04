def parse_speaker_map(spec: str) -> dict[str, str]:
    """Parse 'SPEAKER_00=Yogev,SPEAKER_01=Tom' into {'SPEAKER_00': 'Yogev', ...}.

    Empty input → empty dict. Whitespace around commas and equals is tolerated.
    Malformed pairs raise ValueError."""
    if not spec or not spec.strip():
        return {}
    out: dict[str, str] = {}
    for raw_pair in spec.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"speaker-map entry {pair!r} missing '='")
        label, _, name = pair.partition("=")
        label = label.strip()
        name = name.strip()
        if not label or not name:
            raise ValueError(f"speaker-map entry {pair!r} has empty label or name")
        out[label] = name
    return out
