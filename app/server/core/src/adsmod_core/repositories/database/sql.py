from __future__ import annotations

import sqlalchemy
from sqlalchemy.sql.elements import TextClause

###############################################################################
def build_postgres_create_database_sql(
    database_name: str,
) -> TextClause:
    safe_database = database_name.replace('"', '""')
    return sqlalchemy.text(
        f"CREATE DATABASE \"{safe_database}\" WITH ENCODING 'UTF8' TEMPLATE template0"
    )

###############################################################################
def build_postgres_database_exists_sql() -> TextClause:
    return sqlalchemy.text("SELECT 1 FROM pg_database WHERE datname=:name")
