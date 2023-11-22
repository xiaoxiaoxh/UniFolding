import dataclasses
import json
import minio
from typing import List, Any, Dict, Optional

import pymongo
from pydantic import BaseModel, Field

from .config import BackendConfig


class LogRequest(BaseModel):
    identifiers: Optional[List[str]]
    extra_filter: Optional[Dict[str, Any]]


class LogPoint(BaseModel):
    identifier: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    metadata_bak: List[Dict[str, Any]] = Field(default_factory=list)
    annotations: Dict[str, Optional[Dict[str, Any]]] = Field(default_factory=dict)
    annotators: List[str] = Field(default_factory=list)

    def __post_init__(self):
        pass

    def to_dict(self):
        return {
            "identifier": self.identifier,
            "metadata": self.metadata,
            "metadata_bak": self.metadata_bak,
            "annotations": self.annotations,
            "annotators": self.annotators,
        }

    def from_dict(self, d: Dict[str, Any]):
        self.identifier = d["identifier"] if "identifier" in d else self.identifier
        self.metadata = d["metadata"] if "metadata" in d else self.metadata
        self.metadata_bak = d["metadata_bak"] if "metadata_bak" in d else self.metadata_bak
        self.annotations = d["annotations"] if "annotations" in d else self.annotations
        self.annotators = d["annotators"] if "annotators" in d else self.annotators
        return self

    def to_json(self):
        return json.dumps(self.to_dict())

    def from_json(self, s: str):
        return self.from_dict(json.loads(s))


class AnnotationPoint(BaseModel):
    annotator: str
    annotation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotator": self.annotator,
            "annotation": self.annotation
        }


@dataclasses.dataclass
class BackendContext:
    opt: BackendConfig
    db: Optional[pymongo.MongoClient]
    oss: Optional[minio.Minio]
    num_replicas: int = dataclasses.field(default=1)

    def __repr__(self):
        return str({
            "option": self.opt.__repr__(),
            "num_replicas": self.num_replicas,
            "db": self.db.__repr__(),
            "oss": self.oss.__repr__()
        })
