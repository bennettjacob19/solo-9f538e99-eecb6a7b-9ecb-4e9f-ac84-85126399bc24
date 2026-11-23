"""Member-identification tests.

These cover the repair mix-up story: the two data disks can be put back under
each other's filenames, but the array must keep addressing each physical member
by its original logical slot.  They also cover the startup guard rails:
foreign equal-length images and duplicate member identities refuse to start
without touching any image or the journal.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import BLOCK_SIZE
from app.errors import VolumeCorrupt

from conftest import block, make_settings, read, read_bytes, write


def _seed(client: TestClient, stripes: int = 4) -> None:
    for index in range(stripes):
        assert write(client, index, "data0", block(0x10 + index)).status_code == 200
        assert write(client, index, "data1", block(0xA0 + index)).status_code == 200


def _snapshot(client: TestClient, stripes: int):
    return {
        (disk, index): read_bytes(client, index, disk)
        for index in range(stripes)
        for disk in ("data0", "data1", "parity")
    }


def _swap_data_images(data_dir: Path) -> None:
    os.rename(data_dir / "data0.img", data_dir / "swap.tmp")
    os.rename(data_dir / "data1.img", data_dir / "data0.img")
    os.rename(data_dir / "swap.tmp", data_dir / "data1.img")


def test_legacy_volume_registers_original_slots_without_changing_images(tmp_path):
    # Simulate a pre-membership volume: three raw images, no members.json.
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        before = {name: (tmp_path / name).read_bytes() for name in
                  ("data0.img", "data1.img", "parity.img")}
    registry_file = tmp_path / "members.json"
    assert registry_file.exists()
    registry = json.loads(registry_file.read_text())
    assert set(registry["members"]) == {"data0", "data1", "parity"}
    for slot, profile in registry["members"].items():
        assert profile["original_image"] == f"{slot}.img"
        assert profile["image"] == f"{slot}.img"
        assert len(profile["member_id"]) >= 16
        assert len(profile["block_sha256"]) == 4
    # Registration never rewrites the raw images.
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
        assert (tmp_path / name).stat().st_size == 4 * BLOCK_SIZE


def test_status_lists_image_behind_every_logical_slot(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        members = c.get("/api/status").json()["members"]
    by_slot = {member["slot"]: member for member in members}
    assert set(by_slot) == {"data0", "data1", "parity"}
    for slot in ("data0", "data1", "parity"):
        member = by_slot[slot]
        assert member["image"] == f"{slot}.img"
        assert member["image_path"].endswith(f"{slot}.img")
        assert member["registered_image"] == f"{slot}.img"
        assert member["state"] == "online"
        assert member["swapped"] is False
        assert member["block_fingerprints"] == 4
        assert member["member_id"]


def test_swapped_data_disks_keep_their_original_slots(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        confirmed = _snapshot(c, 4)
        member_ids_before = {
            slot: profile["member_id"]
            for slot, profile in json.loads((tmp_path / "members.json").read_text())[
                "members"
            ].items()
        }

    _swap_data_images(tmp_path)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        by_slot = {member["slot"]: member for member in status["members"]}
        # The logical slot now points at the other filename but flags the swap.
        assert by_slot["data0"]["image"] == "data1.img"
        assert by_slot["data1"]["image"] == "data0.img"
        assert by_slot["data0"]["swapped"] is True
        assert by_slot["data1"]["swapped"] is True
        assert by_slot["parity"]["swapped"] is False
        # Block I/O still follows the original slots.
        assert _snapshot(c, 4) == confirmed
        assert write(c, 2, "data0", block(0x7E)).status_code == 200
        assert read_bytes(c, 2, "data0") == block(0x7E)
        assert c.get("/api/verify").json()["consistent"] is True
        member_ids_after = {
            slot: profile["member_id"]
            for slot, profile in json.loads((tmp_path / "members.json").read_text())[
                "members"
            ].items()
        }
        assert member_ids_after == member_ids_before


def test_swap_survives_a_crash_and_recovery_uses_the_right_member(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        confirmed = _snapshot(c, 4)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        crashed = write(c, 1, "data0", block(0x5E), power_loss="data")
        assert crashed.status_code == 503

    _swap_data_images(tmp_path)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["journal"]["state"] == "clear"
        assert status["last_recovery"]["recovered"] is True
        # The journal payload is applied to the members by identity, not name.
        assert read_bytes(c, 1, "data0") == block(0x5E)
        assert read_bytes(c, 1, "data1") == confirmed[("data1", 1)]
        assert c.get("/api/verify").json()["consistent"] is True
        # Other stripes are untouched.
        for index in (0, 2, 3):
            for disk in ("data0", "data1", "parity"):
                assert read_bytes(c, index, disk) == confirmed[(disk, index)]


def test_pending_journal_remains_slot_correct_under_swap_in_degraded_mode(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 1, "data0", block(0x6B), power_loss="data").status_code == 503

    _swap_data_images(tmp_path)

    # data0 is offline; its (renamed) physical file must never be opened.
    offline_view = None
    for _ in range(2):
        with TestClient(create_app(make_settings(tmp_path, offline="data0"))) as c:
            status = c.get("/api/status").json()
            assert status["mode"] == "degraded"
            assert status["journal"]["state"] == "prepared"
            assert read(c, 1, "data0")["source"] == "reconstructed"
            view = _snapshot(c, 4)
            assert view[("data0", 1)] == block(0x6B)
            if offline_view is not None:
                assert view == offline_view
            offline_view = view
            assert write(c, 0, "data1", block(1)).status_code == 409

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/api/verify").json()["consistent"] is True
        assert read_bytes(c, 1, "data0") == block(0x6B)
        assert _snapshot(c, 4) == offline_view


def test_foreign_equal_length_image_refuses_start_and_changes_nothing(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        confirmed = _snapshot(c, 4)
        journal_before = (tmp_path / "journal.img").read_bytes()
        images_before = {
            name: (tmp_path / name).read_bytes()
            for name in ("data0.img", "data1.img", "parity.img")
        }

    # Replace data0's content with same-length garbage carrying no identity.
    with open(tmp_path / "data0.img", "wb") as handle:
        handle.write(bytes([0xAB]) * (4 * BLOCK_SIZE))

    with pytest.raises(VolumeCorrupt) as excinfo:
        with TestClient(create_app(make_settings(tmp_path))) as c:
            c.get("/api/status")
    assert excinfo.value.extra["reason"] == "foreign_image"
    assert "data0.img" in excinfo.value.extra["foreign_images"]

    # Nothing was modified: images, journal and registry keep their bytes.
    assert (tmp_path / "journal.img").read_bytes() == journal_before
    for name, content in images_before.items():
        if name == "data0.img":
            continue
        assert (tmp_path / name).read_bytes() == content

    # Reinstall the correct member (healthy replacement rebuilds it) and the
    # original volume boots again.
    (tmp_path / "data0.img").unlink()
    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["mode"] == "healthy"
        assert status["rebuilt_disks"] == ["data0"]
        assert c.get("/api/verify").json()["consistent"] is True
        assert _snapshot(c, 4) == confirmed


def test_duplicate_member_identity_refuses_start(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        journal_before = (tmp_path / "journal.img").read_bytes()
        data0_before = (tmp_path / "data0.img").read_bytes()

    # Two files now hold the data1 member; the data0 member is effectively gone.
    backup = tmp_path / "data0.img.keep"
    os.replace(tmp_path / "data0.img", backup)
    with open(tmp_path / "data0.img", "wb") as handle:
        handle.write((tmp_path / "data1.img").read_bytes())

    with pytest.raises(VolumeCorrupt) as excinfo:
        with TestClient(create_app(make_settings(tmp_path))) as c:
            c.get("/api/status")
    assert excinfo.value.extra["reason"] == "duplicate_member"
    assert set(excinfo.value.extra["duplicate_images"]) == {"data0.img", "data1.img"}
    # Journal and the surviving images were not touched.
    assert (tmp_path / "journal.img").read_bytes() == journal_before
    assert (tmp_path / "data1.img").stat().st_size == 4 * BLOCK_SIZE

    # Reinstall the genuine data0 member and the volume restarts.
    os.replace(backup, tmp_path / "data0.img")
    assert (tmp_path / "data0.img").read_bytes() == data0_before
    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["mode"] == "healthy"
        assert c.get("/api/verify").json()["consistent"] is True


def test_rebuilt_member_keeps_a_stable_identity_for_the_others(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        confirmed = _snapshot(c, 4)
        ids_before = {
            slot: profile["member_id"]
            for slot, profile in json.loads((tmp_path / "members.json").read_text())[
                "members"
            ].items()
        }

    (tmp_path / "data1.img").unlink()
    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["rebuilt_disks"] == ["data1"]
        assert _snapshot(c, 4) == confirmed
        ids_after = {
            slot: profile["member_id"]
            for slot, profile in json.loads((tmp_path / "members.json").read_text())[
                "members"
            ].items()
        }
    # Survivors keep their identity; only the replacement gets a fresh entry.
    assert ids_after["data0"] == ids_before["data0"]
    assert ids_after["parity"] == ids_before["parity"]
    assert ids_after["data1"] == ids_before["data1"]


def test_offline_slot_reads_follow_actual_slot_positions(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        confirmed = _snapshot(c, 4)

    _swap_data_images(tmp_path)
    with TestClient(create_app(make_settings(tmp_path, offline="data1"))) as c:
        status = c.get("/api/status").json()
        assert status["mode"] == "degraded"
        for index in range(4):
            result = read(c, index, "data1")
            assert result["source"] == "reconstructed"
            assert base64.b64decode(result["data_b64"]) == confirmed[("data1", index)]
        # The surviving data0 member still reads natively with its own content.
        assert read(c, 0, "data0")["source"] == "native"
        assert read_bytes(c, 0, "data0") == confirmed[("data0", 0)]
        assert c.get("/api/verify").json()["checked"] is False


def test_registry_is_not_corrupted_by_a_wrong_size_image(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
    registry_before = (tmp_path / "members.json").read_text()
    with open(tmp_path / "parity.img", "r+b") as handle:
        handle.truncate(4 * BLOCK_SIZE - 7)
    with pytest.raises(VolumeCorrupt) as excinfo:
        with TestClient(create_app(make_settings(tmp_path))) as c:
            c.get("/api/status")
    assert excinfo.value.extra["reason"] == "foreign_image"
    assert (tmp_path / "members.json").read_text() == registry_before


def test_extra_foreign_image_is_refused_in_healthy_and_degraded_modes(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        journal_before = (tmp_path / "journal.img").read_bytes()
    with open(tmp_path / "intruder.img", "wb") as handle:
        handle.write(bytes([0x99]) * 4 * BLOCK_SIZE)
    for settings in (
        make_settings(tmp_path),
        make_settings(tmp_path, offline="data0"),
        make_settings(tmp_path, offline="data1"),
    ):
        with pytest.raises(VolumeCorrupt) as excinfo:
            with TestClient(create_app(settings)) as c:
                c.get("/api/status")
        assert excinfo.value.extra["reason"] == "foreign_image"
        assert excinfo.value.extra["foreign_images"] == ["intruder.img"]
    # Refusal is read-only: journal and the three member images are untouched.
    assert (tmp_path / "journal.img").read_bytes() == journal_before
    os.unlink(tmp_path / "intruder.img")
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/api/verify").json()["consistent"] is True


def test_extra_duplicate_member_is_refused_in_every_mode(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        data1_before = (tmp_path / "data1.img").read_bytes()
    with open(tmp_path / "intruder.img", "wb") as handle:
        handle.write(data1_before)
    for settings in (
        make_settings(tmp_path),
        make_settings(tmp_path, offline="data0"),
        make_settings(tmp_path, offline="data1"),
    ):
        with pytest.raises(VolumeCorrupt) as excinfo:
            with TestClient(create_app(settings)) as c:
                c.get("/api/status")
        assert excinfo.value.extra["reason"] == "duplicate_member"
        assert set(excinfo.value.extra["duplicate_images"]) == {
            "data1.img",
            "intruder.img",
        }
    os.unlink(tmp_path / "intruder.img")
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/api/verify").json()["consistent"] is True


def test_swap_back_to_registered_filenames_boots_the_original_volume(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        confirmed = _snapshot(c, 4)
    _swap_data_images(tmp_path)
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/api/verify").json()["consistent"] is True
    # Repair finished correctly: filenames restored to the registered slots.
    _swap_data_images(tmp_path)
    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["mode"] == "healthy"
        assert all(not member["swapped"] for member in status["members"])
        assert _snapshot(c, 4) == confirmed
        assert c.get("/api/verify").json()["consistent"] is True


def test_degraded_swap_then_healthy_recovery_converges(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 1, "data0", block(0x6B), power_loss="data").status_code == 503
    _swap_data_images(tmp_path)
    with TestClient(create_app(make_settings(tmp_path, offline="data1"))) as c:
        assert c.get("/api/status").json()["mode"] == "degraded"
        assert read_bytes(c, 1, "data0") == block(0x6B)
    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["journal"]["state"] == "clear"
        assert c.get("/api/verify").json()["consistent"] is True
        assert read_bytes(c, 1, "data0") == block(0x6B)
