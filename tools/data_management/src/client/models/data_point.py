import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

import attr
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.data_point_annotations import DataPointAnnotations
    from ..models.data_point_metadata import DataPointMetadata


T = TypeVar("T", bound="DataPoint")


@attr.s(auto_attribs=True)
class DataPoint:
    """
    Attributes:
        identifier (Union[Unset, str]):  Default: ''.
        timestamp (Union[Unset, datetime.datetime]):
        metadata (Union[Unset, DataPointMetadata]):
        annotations (Union[Unset, DataPointAnnotations]):
        annotators (Union[Unset, List[str]]):
    """

    identifier: Union[Unset, str] = ""
    timestamp: Union[Unset, datetime.datetime] = UNSET
    metadata: Union[Unset, "DataPointMetadata"] = UNSET
    annotations: Union[Unset, "DataPointAnnotations"] = UNSET
    annotators: Union[Unset, List[str]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        identifier = self.identifier
        timestamp: Union[Unset, str] = UNSET
        if not isinstance(self.timestamp, Unset):
            timestamp = self.timestamp.isoformat()

        metadata: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        annotations: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.annotations, Unset):
            annotations = self.annotations.to_dict()

        annotators: Union[Unset, List[str]] = UNSET
        if not isinstance(self.annotators, Unset):
            annotators = self.annotators

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if identifier is not UNSET:
            field_dict["identifier"] = identifier
        if timestamp is not UNSET:
            field_dict["timestamp"] = timestamp
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if annotations is not UNSET:
            field_dict["annotations"] = annotations
        if annotators is not UNSET:
            field_dict["annotators"] = annotators

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.data_point_annotations import DataPointAnnotations
        from ..models.data_point_metadata import DataPointMetadata

        d = src_dict.copy()
        identifier = d.pop("identifier", UNSET)

        _timestamp = d.pop("timestamp", UNSET)
        timestamp: Union[Unset, datetime.datetime]
        if isinstance(_timestamp, Unset):
            timestamp = UNSET
        else:
            timestamp = isoparse(_timestamp)

        _metadata = d.pop("metadata", UNSET)
        metadata: Union[Unset, DataPointMetadata]
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = DataPointMetadata.from_dict(_metadata)

        _annotations = d.pop("annotations", UNSET)
        annotations: Union[Unset, DataPointAnnotations]
        if isinstance(_annotations, Unset):
            annotations = UNSET
        else:
            annotations = DataPointAnnotations.from_dict(_annotations)

        annotators = cast(List[str], d.pop("annotators", UNSET))

        data_point = cls(
            identifier=identifier,
            timestamp=timestamp,
            metadata=metadata,
            annotations=annotations,
            annotators=annotators,
        )

        data_point.additional_properties = d
        return data_point

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
