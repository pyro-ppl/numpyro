import jax.numpy as jnp
from parsita import TextParsers, reg, repsep, Success, ParseError


def _to_dict(parsed):
    return dict([parsed])


def _convert_proteins(dicts):
    return {k: tuple(list(ll) for ll in zip(*v)) for d in dicts for k, v in d.items()}


AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "G",
    "E",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

DSSP = ["-", "H", "E", "T", "G", "S", "B", "I"]


class ProteinParser(TextParsers, whitespace=r"[ \t]*"):
    name = reg(r"[A-Za-z0-9_]+")
    declaration = "#" >> name

    aminoacid = reg("|".join(AMINO_ACIDS))

    dssp = reg("|".join(DSSP))

    number = reg(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?") > float

    polypeptide = (
        (declaration << "\n")
        & (repsep(aminoacid & dssp & (number | "NT") & (number | "CT"), "\n"))
    ) > _to_dict
    dataset = repsep(polypeptide, "\n") << "\n" > _convert_proteins

    @classmethod
    def parsef(cls, filename):
        with open(filename) as data:
            st = data.read()
            dataset = cls.dataset.parse(st)
            if isinstance(dataset, Success):
                dataset = dataset.value
                return dataset
            else:
                raise ParseError(dataset.message)

    @classmethod
    def parsef_jnp(cls, filename):
        structured_data = cls.parsef(filename)
        inddata = []
        lengths = []
        for as_, ds, phis, psis in structured_data.values():
            inddata.append(
                (
                    list(map(AMINO_ACIDS.index, as_[1:-1])),
                    list(map(DSSP.index, ds[1:-1])),
                    phis[1:-1],
                    psis[1:-1],
                )
            )
            lengths.append(len(as_[1:-1]))
        max_length = max(lengths)
        res = tuple(
            jnp.array([la + [0] * (max_length - len(la)) for la in ll])
            for ll in zip(*inddata)
        )
        return *res, jnp.array(lengths)
