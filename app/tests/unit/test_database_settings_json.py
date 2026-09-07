from __future__ import annotations

from adsmod_common.config import DatabaseConfig, load_config
from pathlib import Path

CANONICAL_CONFIGURATION_FILE = Path("app/resources/adsmod.json")


###############################################################################
def project(payload: dict[str, object]) -> DatabaseConfig:
    canonical = load_config(
        CANONICAL_CONFIGURATION_FILE
    ).application.database.model_dump(mode="python")
    canonical.update(payload)
    return DatabaseConfig.model_validate(canonical)

###############################################################################
def test_db_embedded_json_configuration() -> None:
    database = project(
        {
            "embedded_database": True,
            "connect_timeout": 45,
            "insert_batch_size": 6000,
        }
    )

    assert database.embedded_database is True
    assert database.engine == "postgres"
    assert database.host is None
    assert database.port == 5432
    assert database.database_name is None
    assert database.username is None
    assert database.password is None
    assert database.ssl is False
    assert database.ssl_ca is None
    assert database.connect_timeout == 45
    assert database.insert_batch_size == 6000

###############################################################################
def test_db_external_json_configuration() -> None:
    database = project(
        {
            "embedded_database": False,
            "engine": "postgres",
            "host": "external-db.example.com",
            "port": 6543,
            "database_name": "external_adsmod",
            "username": "external_user",
            "password": "external_password",
            "ssl": True,
            "ssl_ca": "/tmp/ca.pem",
            "connect_timeout": 45,
            "insert_batch_size": 6000,
        }
    )

    assert database.embedded_database is False
    assert database.engine == "postgres"
    assert database.host == "external-db.example.com"
    assert database.port == 6543
    assert database.database_name == "external_adsmod"
    assert database.username == "external_user"
    assert database.password == "external_password"
    assert database.ssl is True
    assert database.ssl_ca == "/tmp/ca.pem"
    assert database.connect_timeout == 45
    assert database.insert_batch_size == 6000

###############################################################################
def test_db_settings_use_canonical_values_when_no_override_is_given() -> None:
    database = project({})

    assert database.embedded_database is True
    assert database.engine == "postgres"
    assert database.host is None
    assert database.port == 5432
    assert database.database_name is None
    assert database.username is None
    assert database.password is None
    assert database.ssl is False
    assert database.ssl_ca is None
    assert database.connect_timeout == 30
    assert database.insert_batch_size == 5000

###############################################################################
def test_db_settings_allow_minimal_external_payload() -> None:
    database = project({"embedded_database": False, "password": ""})
    assert database.embedded_database is False
    assert database.password is None
