"""Frozen facet metadata used by the production dispatch registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class ContractMetadata:
    """Frozen identity and scored rows for one facet contract.

    Parameters
    ----------
    title : str
        Contract title.
    sha256 : str
        SHA-256 of the authoritative Markdown bytes.
    scored_subterms : tuple[str, ...]
        Score-visible manifest sub-term ids.
    """

    title: str
    sha256: str
    scored_subterms: Tuple[str, ...]


CONTRACTS: Mapping[str, ContractMetadata] = {
    "U01": ContractMetadata(
        "Distance/stress fidelity",
        "0db6143734cebf7f28d5a3f231cc7accb4010b8fed86c78ab470927e93651641",
        ("U01.headline",),
    ),
    "U01b": ContractMetadata(
        "Distance strata/bands",
        "30b3e2ee99109b898bf5d301e7d0b1b1fd96065cccd875ecf423757683399e66",
        ("U01b.local", "U01b.long"),
    ),
    "U02": ContractMetadata(
        "Shepard rank fidelity",
        "d4a6ff0c6a7699832bfe925d38feee236309ca858a371b36ba4fb5169e294235",
        ("U02.headline",),
    ),
    "U03": ContractMetadata(
        "Neighborhood preservation, multi-radius, SOFT",
        "48c17bf5152255db40a9e35db6ad09fedb7aca9c57c6bf33d7901c43288d7741",
        ("U03.r_1", "U03.r_2", "U03.r_4"),
    ),
    "U04a": ContractMetadata(
        "Density-map fidelity (structure)",
        "59a7eeda57c5735b2ced3d050af16b57181e127735012a0deb20ac7fff2d08db",
        ("U04a.2u", "U04a.8u"),
    ),
    "U04b": ContractMetadata(
        "Crowding / whitespace legibility",
        "2005ce3d0597b90fefc5f0cc5197f44b64b2c24911eea27e858afe2f2e3c3dd4",
        ("U04b.part_1", "U04b.part_2"),
    ),
    "U05": ContractMetadata(
        "Graph-geometric shape fidelity",
        "47a4b416c09f564831aaf1d9fb66a80bef138bce6bb8daf9a02c3d3b1fa0cfa3",
        ("U05.headline",),
    ),
    "U06": ContractMetadata(
        "Symmetry display",
        "9f6f3aa20625f493f3193bb2c8e377b79d2801257ef494d71cd433421497dabe",
        ("U06.headline",),
    ),
    "U07": ContractMetadata(
        "NORMATIVE CONTRACT: Crossing slot",
        "63526aa6824dfa5180234ba97086725621e8b9e8aa3c40713c85a7c4a86caf2b",
        ("U7.base", "U7.tail"),
    ),
    "U08": ContractMetadata(
        "Angular resolution at nodes",
        "bcc8037fd0f620ce8e43fa7ba68acd711601e12355dd48a2741469dc997a4ebf",
        ("U08.headline",),
    ),
    "U09": ContractMetadata(
        "Edge-length coherence, stratified/conditional",
        "deda54396b4f2de53842b94593bb0d0cc1fdd2084cc0874cec819c0de15abad7",
        ("U09.headline",),
    ),
    "U10": ContractMetadata(
        "Edge-node occlusion / false attachment",
        "91f7929473c6d2f9fb9afc0b0fded75bdc899fe521ca942794b6042739aaf850",
        ("U10.headline",),
    ),
    "U11": ContractMetadata(
        "Routed-edge quality: NORMATIVE CONTRACT (A1)",
        "411dab9b787a8465d2f2591f3048bb6d11d1e57075071532d37c647e74e71dfd",
        ("U11.i", "U11.ii", "U11.iii", "U11.iv", "U11.v"),
    ),
    "U12": ContractMetadata(
        "Path/edge continuity",
        "0170de3c481a98b8d063ff505c039de969221b1e43c3bef596a64c7e0c083d47",
        ("U12.headline",),
    ),
    "U13": ContractMetadata(
        "Near-parallel edge ambiguity / bundle confusion",
        "359205189c5fa413aae039d4033602af0a665c9e93e583c274b57d1341d1ca43",
        ("U13.i", "U13.ii"),
    ),
    "U14": ContractMetadata(
        "False adjacency (node-pair proximity ambiguity)",
        "20a4536ac81681828042a6d06ad76113d0b5c08305e08887799956799a339d65",
        ("U14.headline",),
    ),
    "U15": ContractMetadata(
        "Multi-edge / self-loop legibility",
        "ba372b7b0425c2a202e8d76de7a16b88f9bb18ebb2a6220df0ecfc4089ad83c7",
        ("U15.i", "U15.ii"),
    ),
    "U16": ContractMetadata(
        "Edge-label placement",
        "10e732d51189a4e9bf607044ce70d96042a3f3474fa172917467b043e49145a7",
        ("U16.i", "U16.ii"),
    ),
    "U17": ContractMetadata(
        "Node-node occlusion / clearance",
        "c4b70fd5614c1c66e1e96cb62e41fbe87803319f5b50f96b4bf8bbc2cc74a322",
        ("U17.1",),
    ),
    "U18": ContractMetadata(
        "Label legibility (node labels)",
        "68e32c4e9edfac024c8d2e651538d5e41e0781134960f75ca06b131cb40b346d",
        ("U18.ll", "U18.ln", "U18.le"),
    ),
    "U19": ContractMetadata(
        "Text legibility feasibility",
        "b7166c90068923a4edb1bcd4dad3371ebb7dd9134d7e8f657afa9b2ae2a7e2da",
        ("U19.headline",),
    ),
    "U20a": ContractMetadata(
        "Resolution-limit degeneracy",
        "65e179ba53966fecc9fb77c4c219e05086bee5dac536540ab46bab785313b530",
        ("U20a.i", "U20a.ii", "U20a.iii"),
    ),
    "U20b": ContractMetadata(
        "Scale-legibility plateau",
        "8037cbb7308319db2434687b3b149b5ee6797a53f78eda9e81727ebe39531947",
        ("U20b.headline",),
    ),
    "U21": ContractMetadata(
        "Frame economy / anti-sprawl",
        "c662b51acf5fa3863ad367627168b280cdc6eb23194f7577203d9b0bd899d4de",
        ("U21.d_sparse_n", "U21.d_overflow"),
    ),
    "U22": ContractMetadata(
        "Aspect ratio / shape",
        "d141e6fef540d3115fe342b275133278b550ee5d064663779b19c58b61b6304e",
        ("U22.headline",),
    ),
    "U23": ContractMetadata(
        "Visual balance",
        "c14a86d9e00099c5db2c6d571e1af057896146222e96737678253a76ffdd5972",
        ("U23.headline",),
    ),
    "U24": ContractMetadata(
        "Total ink economy",
        "5857b951b89bbebff02131437db2b8070d84dc03460746adc7f7e65ac1d03573",
        ("U24.headline",),
    ),
    "U25": ContractMetadata(
        "Cluster cohesion / compactness",
        "1abaa433699c51d9c1d2ed8c7bde4643df3d8eb46fe9133d465b434e40f15299",
        ("U25.headline",),
    ),
    "U26": ContractMetadata(
        "Cluster separation (+ community faithfulness)",
        "bba49ea4d943942ee7510d7e16651470a63a488a990394d5b3ec8e1b09311508",
        ("U26.i", "U26.ii", "U26.iii"),
    ),
    "U27": ContractMetadata(
        "Containment / non-intrusion",
        "fee484e87c819fbe1457b7519fb0586ef1ca5f27f8fed0e7eeaa5fc7ba997f5c",
        ("U27.i", "U27.ii", "U27.iii"),
    ),
    "U28": ContractMetadata(
        "Hierarchy nesting fidelity",
        "5d48376583595a1fad3dee50676d440eb2f0d5bf9bdd6acb03b93535a421e649",
        ("U28.i", "U28.ii", "U28.iii"),
    ),
    "U29": ContractMetadata(
        "Cluster shape coherence",
        "658cf935c25388c5959b5c4af9880840c59b28fe1b6284d0dfe1dfa0d9e7a819",
        ("U29.headline",),
    ),
    "U30": ContractMetadata(
        "Cluster labels",
        "64ce190db377b294922a09b4b0e045224c02d611ff451b33f3fb5d07d1ca69c4",
        ("U30.i", "U30.ii", "U30.iii"),
    ),
    "U31": ContractMetadata(
        "Direction consistency",
        "e8c67a87acbfb2a4882c39f54ff50e6c34238cabb5324fad2ab48b0fc5f99ef2",
        ("U31.headline",),
    ),
    "U32": ContractMetadata(
        "Rank/layer clarity",
        "e1fdb3ffef119b8151d07ba8673778c88bb7bc565d596248220d2e828fe6eba4",
        ("U32.L_iso", "U32.L_crisp", "U32.L_overlap"),
    ),
    "U33": ContractMetadata(
        "Tree quality bundle",
        "b9ea391f1645e98497c216a76ba3c1b0d69901cdf5d54c42872376a7ea4e7feb",
        (
            "U33.layered.1",
            "U33.layered.2",
            "U33.layered.3",
            "U33.layered.4",
            "U33.radial.1",
            "U33.radial.2",
        ),
    ),
    "U34": ContractMetadata(
        "Flow path traceability",
        "30aeff9b4e59ee355dbad183fb26686c084c426eb1430762b8e6f273e2fdb2cc",
        ("U34.L_back", "U34.L_mono", "U34.L_cont"),
    ),
    "U35": ContractMetadata(
        "Weighted distance fidelity",
        "7644f1fef1bcede526f923da3bc70e24005a0c28296369300eec9cc9d4743810",
        ("U35.headline",),
    ),
    "U36": ContractMetadata(
        "Local weight monotonicity",
        "574b04b859567036912ca0815385311a2524dca0cb89ff0bda27acbb4c0ed703",
        ("U36.headline",),
    ),
    "U37": ContractMetadata(
        "Thickness-only weight encoding",
        "ad46f1330b123bd7941b47e9e49cd00b59803c0705b2d2432bda4da81ece53da",
        ("U37.ell_e", "U37.ell_ord"),
    ),
    "U38": ContractMetadata(
        "Multi-component packing",
        "1a0023e6c0515714dbda83f97a6a57e1c4ecbf7e27f15a6650c674ca9f0f68b1",
        ("U38.L_clear", "U38.L_pack", "U38.L_prop"),
    ),
    "U39": ContractMetadata(
        "Port compliance",
        "5de7639e452aa58f5d54c5d3530bda54f74def1ab2e09e8baaa57aa4cd8f3d90",
        ("U39.1", "U39.2", "U39.3", "U39.4"),
    ),
    "U40": ContractMetadata(
        "Temporal / mental-map continuity",
        "b10f18a4380790db1be284956dc2094f3bdcc72334be500b7c08b5e9cc43ce8c",
        ("U40.1", "U40.2", "U40.3"),
    ),
    "U41": ContractMetadata(
        "Planarity and face quality",
        "b26cdb05f3d09cda123b5fd6becbc8d7ebdaca931b89d0f2d2ee1e2d0f0f518a",
        ("U41.L_conv", "U41.L_area"),
    ),
    "U42": ContractMetadata(
        "Encoding fidelity & contrast",
        "7ee672a532e6032cab87ca1ffe35156192c1a19edf384a4db5e33803d389c94e",
        ("U42.i", "U42.ii", "U42.iv"),
    ),
}

SCORED_SUBTERM_COUNT = sum(len(contract.scored_subterms) for contract in CONTRACTS.values())


def contract_doc(facet_id: str) -> str:
    """Return the required function docstring for a contract.

    Parameters
    ----------
    facet_id : str
        Frozen contract id.

    Returns
    -------
    str
        Contract title followed by its frozen SHA-256.
    """

    contract = CONTRACTS[facet_id]
    return f"{contract.title}. Frozen SHA-256: {contract.sha256}."
