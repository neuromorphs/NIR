"""This file exists to overcome the issue that superclasses to dataclasses doesnt't
support default attributes --- before Python 3.10.

C.f. https://stackoverflow.com/q/69711886
Specifically, we wish to assign a version to all nodes without specifically providing it
as an argument. Until Python 3.9 end-of-life in 2026, the below solution should work for
Python < 3.10.
"""
from dataclasses import MISSING, field, make_dataclass


class Dataclass:
    pass


class Declarations:
    pass


class Definitions:
    pass


# Thanks to https://stackoverflow.com/q/69711886
def meta_dataclass(
    cls=None,
    /,
    *,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
):
    def wrap(cls):
        declaration_bases = []
        definition_bases = []

        for base in cls.__bases__:
            if issubclass(base, Dataclass):
                declaration_bases += [
                    c for c in base.__bases__ if issubclass(c, Declarations)
                ]
                definition_bases += [
                    c for c in base.__bases__ if issubclass(c, Definitions)
                ]
            elif len(cls.__bases__) == 1 and base != object:
                raise ValueError(
                    f"meta_dataclasses can only inherit from other meta_dataclasses. "
                    f"{cls} inherits from {base}"
                )

        declaration_bases.append(Declarations)
        definition_bases.append(Definitions)

        fields = []
        if hasattr(cls, "__annotations__"):
            for field_name, field_type in cls.__annotations__.items():
                f = (
                    field(default=cls.__dict__[field_name])
                    if field_name in cls.__dict__
                    else field()
                )
                fields.append((field_name, field_type, f))

        declarations = make_dataclass(
            cls_name=f"{cls.__name__}_Declarations",
            bases=tuple(declaration_bases),
            fields=[f for f in fields if isinstance(f[2].default, type(MISSING))],
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )

        definitions = make_dataclass(
            cls_name=f"{cls.__name__}_Definitions",
            bases=tuple(definition_bases),
            fields=[f for f in fields if not isinstance(f[2].default, type(MISSING))],
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )

        cls_wrapper = make_dataclass(
            cls_name=cls.__name__,
            fields=[],
            bases=(Dataclass, definitions, declarations),
            namespace=cls.__dict__,
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )

        return cls_wrapper

    if cls is None:
        return wrap
    else:
        return wrap(cls)
