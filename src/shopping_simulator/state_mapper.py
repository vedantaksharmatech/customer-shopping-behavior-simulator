"""Map source events to the four conceptual model states."""
EVENT_TO_STATE = {"view": "BROWSING", "addtocart": "CART", "transaction": "PURCHASED"}


def map_event(event: str) -> str:
    """Map one recognized source event; reject unknown event names."""
    try:
        return EVENT_TO_STATE[event]
    except KeyError as exc:
        raise ValueError(f"Unsupported event type: {event!r}") from exc
