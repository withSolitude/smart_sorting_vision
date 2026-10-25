
"""
StorageBox

单个储物盒状态管理。
"""

from dataclasses import dataclass
from typing import Optional

from .object_info import ObjectInfo


@dataclass
class StorageBox:

    box_id: int

    object_key: Optional[str] = None

    object_info: Optional[ObjectInfo] = None

    count: int = 0


    def empty(self) -> bool:

        return self.object_key is None



    def match(self, obj: ObjectInfo) -> bool:

        if self.object_key is None:
            return False

        return self.object_key == obj.key()



    def add(self, obj: ObjectInfo):

        if self.empty():

            self.object_key = obj.key()

            self.object_info = obj


        self.count += 1
