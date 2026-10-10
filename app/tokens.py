import re

TOKEN = re.compile(r"\{\{([pf]):(\d+)\}\}")


def sanitize_name(name: str) -> str:
    name = re.sub(r"[{}<>\\`]", "", name or "").strip()
    return re.sub(r"\s+", " ", name)[:50]


def load_people() -> dict[int, dict]:
    from .db import conn
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT person_id, display_name, merged_into, deleted FROM people")
        return {row["person_id"]: row for row in cur.fetchall()}


def root_id(pid: int, people: dict[int, dict]) -> int:
    seen = set()
    while people.get(pid, {}).get("merged_into") and pid not in seen:
        seen.add(pid); pid = people[pid]["merged_into"]
    return pid


def display(pid: int, people: dict[int, dict]) -> str:
    person = people.get(root_id(pid, people))
    return "a person" if not person or person["deleted"] else (person["display_name"] or f"User {person['person_id']}")


def render(template: str | None, people: dict[int, dict]) -> str:
    if not template: return ""
    return TOKEN.sub(lambda m: "an unidentified person" if m[1] == "f" else display(int(m[2]), people), template)


def join_names(names: list[str]) -> str:
    return "" if not names else names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def presence_template(tokens: list[str]) -> str:
    return "No identified people in this image." if not tokens else f"{join_names(tokens)} {'is' if len(tokens) == 1 else 'are'} present in this image."


def summarise_group(names: list[str], max_names: int = 4) -> str:
    return join_names(names) if len(names) <= max_names else join_names(names[:max_names - 1] + [f"{len(names) - max_names + 1} others"])
