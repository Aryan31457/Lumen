from pathlib import Path
import sys

# Running ``python sql/clean.py`` puts sql/ rather than the repository root
# on sys.path, so add the root before importing the application package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import conn
from app.qdrant_store import client, ensure_collections


def main():
    with conn() as db, db.cursor() as cur:
        cur.execute("TRUNCATE image_people, faces, images, people RESTART IDENTITY CASCADE")

    qdrant = client()
    for name in ("faces", "photos"):
        if qdrant.collection_exists(name):
            qdrant.delete_collection(name)

    ensure_collections()
    print("App data cleared; empty collections recreated.")


if __name__ == "__main__":
    main()
