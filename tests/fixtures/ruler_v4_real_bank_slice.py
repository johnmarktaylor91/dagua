"""Recorded, non-sealed RULER V4 main-bank loader fixture."""

FIXTURE: dict[str, object] = {
    "source": {
        "bank_path": "p3/bank/main/main-s0001.jsonl",
        "bank_sha256": "36de2befb2f53583cb65c21f7e1ef2f9e41b522535473c9d50b6144a181f9d67",  # noqa: E501  # pragma: allowlist secret
        "family_map_path": "p3/frozen/A15_FAMILY_MAP.json",
        "family_map_sha256": "32852df2a9c737d513469dff4df84ad91aec4893cb39cbc96d9f26e71bd28ebe",  # noqa: E501  # pragma: allowlist secret
        "manifest_path": "p3/stage/maincamp/sessions/main-s0001/session_manifest.jsonl",
        "manifest_sha256": "e79df9779c8ec13268f39f75843e642588cf5c0b31edf4560d5f59a8779b54b7",  # noqa: E501  # pragma: allowlist secret
        "role_hash": "efe09188d2f1680a6ca55857e1db0c5d7a9201c2e66600d4fdbec057514c0b02",  # noqa: E501  # pragma: allowlist secret
    },
    "bank_rows": [
        {
            "abstain": False,
            "base_pair_id": "46642a406027f09265c0ef590f4dc67ae17b0bd8c4d06544eefac3141073b1e7",
            "budget_line": "PRIMARY",
            "confidence": 2,
            "control_type": None,
            "graph_hash": "121c3f99a0da32150dbc132d11239c7ccd6af05f7c2b00c9814f0217f7bd7026",  # noqa: E501  # pragma: allowlist secret
            "instrument_hash": "3e13615e8debe6d7825a6963ea2543b4e9071ed030e0bda20bce61be9c36938c",  # noqa: E501  # pragma: allowlist secret
            "is_replication": False,
            "judge_id": "fable-5/CF@4",
            "malformed": False,
            "presentation_id": "1d4c45965019eca55ba0922c801ab06b290d3d28b34b707844c64ed9ddc635d8",
            "session_accepted": True,
            "session_id": "main-s0001",
            "side_bit": 0,
            "tie": False,
            "verdict": -1,
        },
        {
            "abstain": False,
            "base_pair_id": "0d7fb4be24e62f919ad20224097ad8ac914fc62958727541864d229f7e388785",
            "budget_line": "PRIMARY",
            "confidence": 3,
            "control_type": None,
            "graph_hash": "a3ebd3b10190b759e6c9daf14e344d9414f23e3b9ed27fe77963ad2e638c62ab",  # noqa: E501  # pragma: allowlist secret
            "instrument_hash": "3e13615e8debe6d7825a6963ea2543b4e9071ed030e0bda20bce61be9c36938c",  # noqa: E501  # pragma: allowlist secret
            "is_replication": True,
            "judge_id": "fable-5/CF@4",
            "malformed": False,
            "presentation_id": "492502071f30fa7d630f89a8545c89d0316a95e772a03900c4ba49faf4eac068",
            "session_accepted": True,
            "session_id": "main-s0001",
            "side_bit": 1,
            "tie": False,
            "verdict": 3,
        },
    ],
    "manifest_rows": [
        {
            "base_pair_id": "46642a406027f09265c0ef590f4dc67ae17b0bd8c4d06544eefac3141073b1e7",
            "blind_id_A": "0807550bb8cf162d3a4f45ac2ab35588478fb104e19f3b7229739ae4deb90904",
            "blind_id_B": "0bb04815dbbaa610cb5097e4d8eece1bd8cffcfcac9bc3279fcdb7e216ef5551",
            "control_type": None,
            "graph_hash": "121c3f99a0da32150dbc132d11239c7ccd6af05f7c2b00c9814f0217f7bd7026",  # noqa: E501  # pragma: allowlist secret
            "presentation_id": "1d4c45965019eca55ba0922c801ab06b290d3d28b34b707844c64ed9ddc635d8",
            "profile_opaque_id": "cfg-primary-r210",
            "session_id": "main-s0001",
        },
        {
            "base_pair_id": "0d7fb4be24e62f919ad20224097ad8ac914fc62958727541864d229f7e388785",
            "blind_id_A": "012c6633bfeb8206bfd7934d6c8d54307e08db67caf74e3faaa826f1e0436992",
            "blind_id_B": "015add2d703354887b7d243c63c5ba1280d199873c066bfe20291e508abab060",
            "control_type": None,
            "graph_hash": "a3ebd3b10190b759e6c9daf14e344d9414f23e3b9ed27fe77963ad2e638c62ab",  # noqa: E501  # pragma: allowlist secret
            "presentation_id": "492502071f30fa7d630f89a8545c89d0316a95e772a03900c4ba49faf4eac068",
            "profile_opaque_id": "cfg-primary-r210",
            "session_id": "main-s0001",
        },
    ],
    "frozen_schedule_rows": [
        {
            "base_pair_id": "46642a406027f09265c0ef590f4dc67ae17b0bd8c4d06544eefac3141073b1e7",
            "graph_hash": "121c3f99a0da32150dbc132d11239c7ccd6af05f7c2b00c9814f0217f7bd7026",  # noqa: E501  # pragma: allowlist secret
            "partition": "cross-family-calibration",
            "presentation_id": "bef71a78b496a85d1b4705bfc8c1dfd1fe6f212ad9f2b5d2234dcb9181caa21e",
            "session_id": "s0001",
        },
        {
            "base_pair_id": "0d7fb4be24e62f919ad20224097ad8ac914fc62958727541864d229f7e388785",
            "graph_hash": "a3ebd3b10190b759e6c9daf14e344d9414f23e3b9ed27fe77963ad2e638c62ab",  # noqa: E501  # pragma: allowlist secret
            "partition": "train",
            "presentation_id": "b4252f5f6090e7a2409a86209724726750a9c764c4d1e0f9519039d1f85cec64",
            "session_id": "s0001",
        },
        {
            "base_pair_id": "0d7fb4be24e62f919ad20224097ad8ac914fc62958727541864d229f7e388785",
            "graph_hash": "a3ebd3b10190b759e6c9daf14e344d9414f23e3b9ed27fe77963ad2e638c62ab",  # noqa: E501  # pragma: allowlist secret
            "partition": "train",
            "presentation_id": "ceac38bcb08cd46aa2227573d9484a90a7e017999b1ad501641a4050c50e6baa",
            "session_id": "s0297",
        },
    ],
    "graphs": {
        "121c3f99a0da32150dbc132d11239c7ccd6af05f7c2b00c9814f0217f7bd7026": {  # noqa: E501  # pragma: allowlist secret
            "authoring_suite": "fairfield-authored",
            "family_role": "family-calibration",
            "generator_family": "authored::kitchen_sink_hybrid_net",
            "graph_name": "kitchen_sink_hybrid_net",
            "primary_class": "clustered-compound",
            "role": "cross-family-calibration",
            "rule_id": "F-99",
            "size_band": "le30",
            "sweep_unit": "authored::kitchen_sink_hybrid_net::kitchen_sink_hybrid_net",
        },
        "a3ebd3b10190b759e6c9daf14e344d9414f23e3b9ed27fe77963ad2e638c62ab": {  # noqa: E501  # pragma: allowlist secret
            "authoring_suite": "classic-generators",
            "family_role": "fit-pool",
            "generator_family": "gen-rgg",
            "graph_name": "rgg_500",
            "primary_class": "dense-sparse-random",
            "role": "train",
            "rule_id": "F-22",
            "size_band": "301-1000",
            "sweep_unit": "gen-rgg::rgg-ladder",
        },
    },
}
