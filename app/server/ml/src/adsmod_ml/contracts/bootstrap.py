from __future__ import annotations

from pydantic import BaseModel

###############################################################################
class ServiceStatusResponse(BaseModel):
    status: str
