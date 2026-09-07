from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from adsmod_core.repositories.schemas.types import JSONSequence


Base = declarative_base()


###############################################################################
class SequenceModel(Base):
    __tablename__ = "test_data"

    id = Column(Integer, primary_key=True)
    sequence = Column(JSONSequence)


###############################################################################
@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as current_session:
        yield current_session


###############################################################################
def test_json_sequence_round_trip(session) -> None:  # type: ignore[no-untyped-def]
    data = [1.1, 2.2, 3.3]
    session.add(SequenceModel(sequence=data))
    session.commit()

    retrieved = session.query(SequenceModel).first()
    assert retrieved is not None
    assert retrieved.sequence == data
    assert isinstance(retrieved.sequence, list)


###############################################################################
def test_json_sequence_empty_list(session) -> None:  # type: ignore[no-untyped-def]
    session.add(SequenceModel(sequence=[]))
    session.commit()

    retrieved = session.query(SequenceModel).first()
    assert retrieved is not None
    assert retrieved.sequence == []


###############################################################################
def test_json_sequence_none(session) -> None:  # type: ignore[no-untyped-def]
    session.add(SequenceModel(sequence=None))
    session.commit()

    retrieved = session.query(SequenceModel).first()
    assert retrieved is not None
    assert retrieved.sequence is None


###############################################################################
def test_string_payload_raises_for_json_sequence(session) -> None:  # type: ignore[no-untyped-def]
    session.execute(SequenceModel.__table__.insert().values(sequence="1.1, 2.2, 3.3"))
    session.commit()

    with pytest.raises(ValueError, match="Invalid JSONSequence payload"):
        session.query(SequenceModel).first()
