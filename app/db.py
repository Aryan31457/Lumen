from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row
from .config import settings


@contextmanager
def conn():
    """Yield a transaction; psycopg commits on normal context exit."""
    with psycopg.connect(settings.database_url, row_factory=dict_row) as connection:
        yield connection
