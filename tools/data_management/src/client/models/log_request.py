from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.log_request_extra_filter import LogRequestExtraFilter


T = TypeVar("T", bound="LogRequest")


@attr.s(auto_attribs=True)
class LogRequest:
    """
    Attributes:
        identifiers (Union[Unset, List[str]]):
        extra_filter (Union[Unset, LogRequestExtraFilter]):
    """

    identifiers: Union[Unset, List[str]] = UNSET
    extra_filter: Union[Unset, "LogRequestExtraFilter"] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        identifiers: Union[Unset, List[str]] = UNSET
        if not isinstance(self.identifiers, Unset):
            identifiers = self.identifiers

        extra_filter: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.extra_filter, Unset):
            extra_filter = self.extra_filter.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if identifiers is not UNSET:
            field_dict["identifiers"] = identifiers
        if extra_filter is not UNSET:
            field_dict["extra_filter"] = extra_filter

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.log_request_extra_filter import LogRequestExtraFilter

        d = src_dict.copy()
        identifiers = cast(List[str], d.pop("identifiers", UNSET))

        _extra_filter = d.pop("extra_filter", UNSET)
        extra_filter: Union[Unset, LogRequestExtraFilter]
        if isinstance(_extra_filter, Unset):
            extra_filter = UNSET
        else:
            extra_filter = LogRequestExtraFilter.from_dict(_extra_filter)

        log_request = cls(
            identifiers=identifiers,
            extra_filter=extra_filter,
        )

        log_request.additional_properties = d
        return log_request

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
