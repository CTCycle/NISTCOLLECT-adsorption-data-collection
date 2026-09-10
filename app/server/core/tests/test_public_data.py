from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError

from adsmod_common.config import DatabaseConfig
from adsmod_core.providers.cod import CODProvider
from adsmod_core.providers.pubchem import PubChemProvider
from adsmod_core.providers.public_data import ProviderUnavailableError
from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.public_data import PublicDataRepository
from adsmod_core.repositories.schemas import Base
from adsmod_core.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
    Observation,
)
from adsmod_core.repositories.schemas.public_data import (
    DataSource,
    Reference,
    SourceRecord,
    SourceRecordReference,
    Structure,
    StructureSourceRecord,
)


###############################################################################
def _manager() -> DatabaseManager:
    manager = DatabaseManager(
        DatabaseConfig(
            embedded_database=True,
            connect_timeout=5,
            insert_batch_size=100,
            sqlite_path=":memory:",
        )
    )
    Base.metadata.create_all(manager.engine)
    return manager


###############################################################################
def _count_selects(manager: DatabaseManager, operation) -> int:  # type: ignore[no-untyped-def]
    statements: list[str] = []

    def capture_selects(
        conn, cursor, statement, parameters, context, executemany  # type: ignore[no-untyped-def]
    ) -> None:
        del conn, cursor, parameters, context, executemany
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(manager.engine, "before_cursor_execute", capture_selects)
    try:
        operation()
    finally:
        event.remove(manager.engine, "before_cursor_execute", capture_selects)
    return len(statements)


###############################################################################
def _seed_listing_rows(repository: PublicDataRepository) -> int:
    with repository.database.transaction() as session:
        dataset = Dataset(name="Public data performance", source="nist")
        session.add(dataset)
        session.flush()
        first_structure_id = 0
        for index in range(2):
            material = Adsorbent(
                key=f"listing-material-{index}",
                name=f"Listing material {index}",
            )
            adsorbate = Adsorbate(
                key=f"listing-adsorbate-{index}",
                name=f"Listing adsorbate {index}",
            )
            session.add_all([material, adsorbate])
            session.flush()
            isotherm = Isotherm(
                dataset_id=dataset.id,
                external_key=f"listing-isotherm-{index}",
                name=f"Listing isotherm {index}",
                adsorbent_id=material.id,
                temperature_original=298.15,
                temperature_original_unit="K",
                temperature_k=298.15,
                pressure_basis="absolute",
            )
            session.add(isotherm)
            session.flush()
            component = IsothermComponent(
                isotherm_id=isotherm.id,
                position=1,
                adsorbate_id=adsorbate.id,
                mole_fraction=1.0,
            )
            session.add(component)
            session.flush()
            session.add(
                Observation(
                    isotherm_id=isotherm.id,
                    component_id=component.id,
                    sequence_index=0,
                    pressure_original=1.0,
                    pressure_original_unit="bar",
                    pressure_canonical=100_000.0,
                    pressure_canonical_unit="Pa",
                    uptake_original=1.0,
                    uptake_original_unit="mol/kg",
                    uptake_mol_kg=1.0,
                )
            )
            structure = Structure(
                adsorbent_id=material.id,
                name=f"Listing structure {index}",
                formula="C",
                format="cif",
                content=f"data_listing_{index}",
                content_sha256=f"{index + 1:064d}",
                has_coordinates=False,
            )
            session.add(structure)
            session.flush()
            if index == 0:
                first_structure_id = structure.id

    for index in range(2):
        with repository.database.session_factory() as session:
            material_id = session.scalar(
                select(Adsorbent.id).where(Adsorbent.key == f"listing-material-{index}")
            )
            adsorbate_id = session.scalar(
                select(Adsorbate.id).where(Adsorbate.key == f"listing-adsorbate-{index}")
            )
        assert material_id is not None
        assert adsorbate_id is not None
        repository.link_adsorbent_record(
            source_key="nist",
            adsorbent_id=material_id,
            external_id=f"material-{index}",
        )
        repository.link_adsorbate_record(
            source_key="nist",
            adsorbate_id=adsorbate_id,
            external_id=f"adsorbate-{index}",
        )
        repository.link_isotherm_record(
            source_key="nist",
            dataset_id=1,
            external_id=f"listing-isotherm-{index}",
        )
    return first_structure_id


###############################################################################
def test_provider_registry_and_source_identity_constraints() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
        assert {row["key"] for row in repository.source_rows()} == {
            "nist",
            "pubchem",
            "cod",
        }

        with manager.transaction() as session:
            pubchem_id = session.scalar(select(DataSource.id).where(DataSource.key == "pubchem"))
            assert pubchem_id is not None
            session.add(
                SourceRecord(
                    source_id=pubchem_id,
                    record_type="chemical",
                    external_id="123",
                    source_url="https://pubchem.ncbi.nlm.nih.gov/compound/123",
                    raw_metadata={},
                )
            )
        with manager.transaction() as session:
            pubchem_id = session.scalar(select(DataSource.id).where(DataSource.key == "pubchem"))
            session.add(
                SourceRecord(
                    source_id=pubchem_id,
                    record_type="chemical",
                    external_id="123",
                    raw_metadata={},
                )
            )
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
            else:
                raise AssertionError("duplicate source identity was accepted")
    finally:
        manager.dispose()


###############################################################################
def test_pubchem_resolution_normalizes_properties_without_network(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    provider = PubChemProvider(parallel_requests=1)

    async def fake_request(method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        del method, kwargs
        if "/property/" in url:
            return httpx.Response(
                200,
                json={
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 297,
                                "Title": "Methane",
                                "IUPACName": "methane",
                                "MolecularFormula": "CH4",
                                "MolecularWeight": "16.043",
                                "SMILES": "C",
                                "ConnectivitySMILES": "C",
                                "InChI": "InChI=1S/CH4/h1H4",
                                "InChIKey": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                                "TPSA": 0,
                                "HBondDonorCount": 0,
                                "HBondAcceptorCount": 0,
                            }
                        ]
                    }
                },
            )
        if "/synonyms/" in url:
            return httpx.Response(
                200,
                json={
                    "InformationList": {
                        "Information": [{"CID": 297, "Synonym": ["Methane", "Marsh gas"]}]
                    }
                },
            )
        return httpx.Response(200, text="3D SDF")

    monkeypatch.setattr(provider, "_request", fake_request)
    payload = asyncio.run(provider.resolve("methane"))

    assert payload["cid"] == "297"
    assert payload["inchi_key"] == "VNWKTOKETHGBQD-UHFFFAOYSA-N"
    assert payload["elemental_composition"] == {"C": 1.0, "H": 4.0}
    assert payload["descriptors"]["tpsa_angstrom2"] == 0.0
    assert payload["conformer_3d_url"] is not None


###############################################################################
def test_pubchem_resolution_maps_successful_malformed_json_to_provider_error(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    provider = PubChemProvider(parallel_requests=1)

    async def fake_request(method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        del method, url, kwargs
        return httpx.Response(200, text="{not valid json")

    monkeypatch.setattr(provider, "_request", fake_request)

    with pytest.raises(ProviderUnavailableError, match="malformed JSON"):
        asyncio.run(provider.resolve("methane"))


###############################################################################
def test_pubchem_empty_query_preserves_local_validation_error() -> None:
    provider = PubChemProvider(parallel_requests=1)

    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(provider.resolve("  "))


###############################################################################
def test_pubchem_upsert_uses_strong_identity_and_does_not_merge_by_name() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
        with manager.transaction() as session:
            session.add(
                Adsorbate(
                    key="upload-methane",
                    name="Methane",
                    inchi_key=None,
                )
            )

        pubchem_id = repository.upsert_pubchem_compound(
            {
                "cid": "297",
                "name": "Methane",
                "preferred_name": "methane",
                "formula": "CH4",
                "molecular_weight": 16.043,
                "smiles": "C",
                "connectivity_smiles": "C",
                "inchi": "InChI=1S/CH4/h1H4",
                "inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                "synonyms": ["Methane"],
                "descriptors": {},
                "elemental_composition": {"C": 1.0, "H": 4.0},
                "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/297",
                "raw_metadata": {},
            }
        )
        with manager.session_factory() as session:
            records = session.scalars(select(Adsorbate).order_by(Adsorbate.id)).all()
            assert len(records) == 2
            assert records[1].id == pubchem_id
            assert records[1].inchi_key == "VNWKTOKETHGBQD-UHFFFAOYSA-N"
    finally:
        manager.dispose()


###############################################################################
def test_cod_atom_parser_normalizes_fractional_coordinates() -> None:
    cif = """
data_test
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.125 0.250 0.375 1.0
O1 O 0.500(2) 0.625 0.750 0.5
"""
    atoms = CODProvider.parse_atoms(cif)
    assert atoms == [
        {
            "sequence_index": 0,
            "label": "C1",
            "element": "C",
            "fractional_x": 0.125,
            "fractional_y": 0.25,
            "fractional_z": 0.375,
            "occupancy": 1.0,
        },
        {
            "sequence_index": 1,
            "label": "O1",
            "element": "O",
            "fractional_x": 0.5,
            "fractional_y": 0.625,
            "fractional_z": 0.75,
            "occupancy": 0.5,
        },
    ]


###############################################################################
def test_cod_search_maps_successful_malformed_json_to_provider_error(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    provider = CODProvider(
        request_timeout_seconds=1.0,
        retry_attempts=1,
        max_interactive_results=10,
    )

    async def fake_request(method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        del method, url
        if kwargs["params"]["format"] == "count":
            return httpx.Response(200, text="1")
        return httpx.Response(200, text="{not valid json")

    monkeypatch.setattr(provider, "_request", fake_request)

    with pytest.raises(ProviderUnavailableError, match="malformed JSON"):
        asyncio.run(provider.search(text="silica"))


###############################################################################
def test_cod_search_validation_errors_remain_local_value_errors() -> None:
    provider = CODProvider(
        request_timeout_seconds=1.0,
        retry_attempts=1,
        max_interactive_results=10,
    )

    with pytest.raises(ValueError, match="at least three characters"):
        asyncio.run(provider.search(text="Si"))
    with pytest.raises(ValueError, match="Provide a COD ID"):
        asyncio.run(provider.search())


###############################################################################
def test_cod_reimport_without_material_preserves_existing_association() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
        with manager.transaction() as session:
            adsorbent = Adsorbent(
                key="test-silica",
                name="Test silica",
                normalized_name="test silica",
            )
            session.add(adsorbent)
            session.flush()
            adsorbent_id = adsorbent.id

        metadata = {
            "cod_id": "4502440",
            "name": "Test silica",
            "formula": "SiO2",
        }
        structure_id = repository.upsert_cod_structure(
            metadata=metadata,
            cif_text="data_test",
            atoms=[],
            adsorbent_id=adsorbent_id,
        )
        assert (
            repository.upsert_cod_structure(
                metadata=metadata,
                cif_text="data_test",
                atoms=[],
                adsorbent_id=None,
            )
            == structure_id
        )

        with manager.session_factory() as session:
            structure = session.get(Structure, structure_id)
            assert structure is not None
            assert structure.adsorbent_id == adsorbent_id
    finally:
        manager.dispose()


###############################################################################
def test_public_data_list_queries_do_not_scale_with_page_size() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
        _seed_listing_rows(repository)

        assert _count_selects(
            manager,
            lambda: repository.list_adsorption(page=1, page_size=2),
        ) <= 6
        assert _count_selects(
            manager,
            lambda: repository.list_materials(page=1, page_size=2),
        ) <= 4
        assert _count_selects(
            manager,
            lambda: repository.list_chemicals(page=1, page_size=2),
        ) <= 5
        assert _count_selects(
            manager,
            lambda: repository.list_structures(page=1, page_size=2),
        ) <= 6
    finally:
        manager.dispose()


###############################################################################
def test_structure_reference_matches_primary_provenance_record() -> None:
    manager = _manager()
    try:
        repository = PublicDataRepository(manager)
        repository.ensure_sources()
        structure_id = _seed_listing_rows(repository)
        with manager.transaction() as session:
            nist_id = session.scalar(select(DataSource.id).where(DataSource.key == "nist"))
            cod_id = session.scalar(select(DataSource.id).where(DataSource.key == "cod"))
            assert nist_id is not None
            assert cod_id is not None
            older = SourceRecord(
                source_id=cod_id,
                record_type="structure",
                external_id="older-structure",
                retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                raw_metadata={},
            )
            newer = SourceRecord(
                source_id=nist_id,
                record_type="structure",
                external_id="newer-structure",
                retrieved_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                raw_metadata={},
            )
            session.add_all([older, newer])
            session.flush()
            session.add_all(
                [
                    StructureSourceRecord(
                        source_record_id=older.id,
                        structure_id=structure_id,
                    ),
                    StructureSourceRecord(
                        source_record_id=newer.id,
                        structure_id=structure_id,
                    ),
                ]
            )
            older_reference = Reference(doi="10.1000/older")
            newer_reference = Reference(doi="10.1000/newer")
            session.add_all([older_reference, newer_reference])
            session.flush()
            session.add_all(
                [
                    SourceRecordReference(
                        source_record_id=older.id,
                        reference_id=older_reference.id,
                    ),
                    SourceRecordReference(
                        source_record_id=newer.id,
                        reference_id=newer_reference.id,
                    ),
                ]
            )

        view = repository.get_structure(structure_id)
        assert view["source"] == "nist"
        assert view["external_id"] == "newer-structure"
        assert view["doi"] == "10.1000/newer"
    finally:
        manager.dispose()
