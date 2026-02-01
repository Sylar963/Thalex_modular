from typing import Optional
from ...domain.interfaces import StorageGateway


class BaseRepository:
    def __init__(self, storage_gateway: Optional[StorageGateway] = None):
        self.storage = storage_gateway
