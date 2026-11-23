"""Member identification for the three-disk XOR array.

The drill's original failure mode is a repair mix-up: the two data disks are
put back into the wrong slots.  A plain XOR check cannot notice that, because
``A XOR B XOR P`` is symmetric in ``A`` and ``B``.  This module gives every
array member a durable identity and maps each *logical slot* (``data0``,
``data1``, ``parity``) to the physical image file that currently backs it,
independently of the file's name.

Design constraints from the drill:

* Legacy volumes predate this feature.  Their images are flat files of
  ``stripes * BLOCK_SIZE`` bytes and must keep their exact raw format and
  contents.  Membership metadata therefore lives in a separate JSON registry
  (``members.json``) next to the images; the images themselves are never
  reformatted, relabeled or rewritten during registration.
* A registry can be stale relative to the images: a confirmed write updates
  the registry transactionally, but a crash may leave a ``prepared`` journal
  with some blocks already landed.  Recognition therefore accepts both the
  last-registered content and the rolled-forward content described by the
  pending journal.
* Recognition is strictly read-only.  A foreign (unknown) equal-length image
  or two images claiming the same member identity aborts startup with a
  :class:`~app.errors.VolumeCorrupt` before any image, registry or journal
  byte is modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any

from .config import BLOCK_SIZE, DISK_NAMES
from .errors import VolumeCorrupt
from .journal import JournalRecord

REGISTRY_NAME = "members.json"
REGISTRY_VERSION = 1


@dataclass(frozen=True)
class MemberProfile:
    """Identity and last-known content fingerprint of one array member."""

    slot: str
    member_id: str
    image: str
    size_bytes: int
    block_sha256: tuple[str, ...] = field(default_factory=tuple)
    registered_ns: int = 0
    # The filename under which this member was first registered.  Kept for the
    # whole lifetime of the volume so a later repair mix-up stays visible even
    # after ``image`` has been updated to the current backing filename.
    original_image: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["block_sha256"] = list(self.block_sha256)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "MemberProfile":
        return cls(
            slot=data["slot"],
            member_id=data["member_id"],
            image=data["image"],
            size_bytes=int(data["size_bytes"]),
            block_sha256=tuple(data.get("block_sha256", ())),
            registered_ns=int(data.get("registered_ns", 0)),
            original_image=data.get("original_image", data["image"]),
        )


@dataclass(frozen=True)
class MemberRegistry:
    profiles: dict[str, MemberProfile]
    version: int = REGISTRY_VERSION
    registered_ns: int = 0

    def profile(self, slot: str) -> MemberProfile:
        return self.profiles[slot]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "registered_ns": self.registered_ns,
            "members": {slot: profile.to_json() for slot, profile in self.profiles.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "MemberRegistry":
        members = data["members"]
        profiles = {}
        for slot in DISK_NAMES:
            if slot not in members:
                raise VolumeCorrupt(
                    "member registry is incomplete",
                    detail={"missing_slot": slot, "registry": REGISTRY_NAME},
                )
            profile = MemberProfile.from_json(members[slot])
            if profile.slot != slot:
                raise VolumeCorrupt(
                    "member registry slot does not match its entry",
                    detail={"slot": slot, "recorded_slot": profile.slot},
                )
            profiles[slot] = profile
        return cls(
            profiles=profiles,
            version=int(data.get("version", REGISTRY_VERSION)),
            registered_ns=int(data.get("registered_ns", 0)),
        )


def new_member_id() -> str:
    return uuid.uuid4().hex


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_block_hashes(path: Path, stripes: int) -> tuple[str, ...]:
    """Return the per-stripe SHA-256 fingerprint of an image (read-only)."""

    hashes: list[str] = []
    fd = os.open(path, os.O_RDONLY)
    try:
        for index in range(stripes):
            block = os.pread(fd, BLOCK_SIZE, index * BLOCK_SIZE)
            if len(block) != BLOCK_SIZE:
                raise VolumeCorrupt(
                    f"short read from image {path.name}",
                    detail={"image": path.name, "index": index, "bytes": len(block)},
                )
            hashes.append(sha256_hex(block))
    finally:
        os.close(fd)
    return tuple(hashes)


def load_registry(path: Path) -> MemberRegistry | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VolumeCorrupt(
            f"cannot read member registry {path.name}: {exc}",
            detail={"registry": path.name},
        ) from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise VolumeCorrupt(
            f"member registry {path.name} is not valid JSON",
            detail={"registry": path.name},
        ) from exc
    registry = MemberRegistry.from_json(data)
    if registry.version != REGISTRY_VERSION:
        raise VolumeCorrupt(
            "member registry has an unsupported version",
            detail={"version": registry.version, "supported": REGISTRY_VERSION},
        )
    return registry


def save_registry(path: Path, registry: MemberRegistry, *, fsync: bool = True) -> None:
    """Atomically persist the registry (temp file + rename + directory fsync)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(registry.to_json(), indent=2, sort_keys=True).encode("utf-8")
    fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    if fsync:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def build_registry(
    images: dict[str, Path],
    *,
    stripes: int,
    offline: str | None = None,
    fsync: bool = True,
) -> MemberRegistry:
    """Register a volume for the first time using its current slot filenames.

    ``images`` maps every logical slot to its backing file.  Online images are
    fingerprinted from their actual bytes; the offline member (whose file must
    not be opened) gets a filename-only placeholder that is enriched on the
    next healthy start.
    """

    now = time.time_ns()
    profiles: dict[str, MemberProfile] = {}
    for slot in DISK_NAMES:
        path = images[slot]
        if slot == offline or not path.exists():
            profiles[slot] = MemberProfile(
                slot=slot,
                member_id=new_member_id(),
                image=path.name,
                size_bytes=stripes * BLOCK_SIZE,
                block_sha256=(),
                registered_ns=now,
                original_image=path.name,
            )
            continue
        hashes = read_block_hashes(path, stripes)
        profiles[slot] = MemberProfile(
            slot=slot,
            member_id=new_member_id(),
            image=path.name,
            size_bytes=path.stat().st_size,
            block_sha256=hashes,
            registered_ns=now,
            original_image=path.name,
        )
    return MemberRegistry(profiles=profiles, registered_ns=now)


def _rolled_forward_hashes(
    profile: MemberProfile, record: JournalRecord
) -> tuple[str, ...] | None:
    """Expected fingerprint after the pending journal commits on this slot."""

    if not profile.block_sha256 or not 0 <= record.stripe < len(profile.block_sha256):
        return None
    rolled = list(profile.block_sha256)
    if profile.slot == "data0":
        rolled[record.stripe] = sha256_hex(record.data0)
    elif profile.slot == "data1":
        rolled[record.stripe] = sha256_hex(record.data1)
    else:
        rolled[record.stripe] = sha256_hex(record.parity)
    return tuple(rolled)


def _slot_models(
    registry: MemberRegistry, record: JournalRecord | None
) -> dict[str, list[tuple[str, ...]]]:
    """Acceptable content fingerprints per slot (baseline + rolled forward)."""

    models: dict[str, list[tuple[str, ...]]] = {}
    for slot, profile in registry.profiles.items():
        acceptable: list[tuple[str, ...]] = []
        if profile.block_sha256:
            acceptable.append(profile.block_sha256)
            if record is not None:
                rolled = _rolled_forward_hashes(profile, record)
                if rolled is not None and rolled not in acceptable:
                    acceptable.append(rolled)
        models[slot] = acceptable
    return models


def slot_edges(
    registry: MemberRegistry,
    fingerprints: dict[str, tuple[str, ...]],
    record: JournalRecord | None,
) -> dict[str, set[str]]:
    """Map each candidate image filename to the slots it may legitimately be."""

    models = _slot_models(registry, record)
    edges: dict[str, set[str]] = {}
    for image, hashes in fingerprints.items():
        matched: set[str] = set()
        for slot, acceptable in models.items():
            profile = registry.profile(slot)
            # A profile without fingerprints (offline placeholder) can only be
            # claimed by its originally registered filename.
            if not acceptable:
                registered_name = profile.original_image or profile.image
                if registered_name == image:
                    matched.add(slot)
                continue
            if hashes in acceptable:
                matched.add(slot)
        edges[image] = matched
    return edges


def _fail_on_bad_membership(
    edges: dict[str, set[str]],
    registry: MemberRegistry,
    needed_slots: set[str],
) -> None:
    foreign = sorted(image for image, slots in edges.items() if not slots)
    if foreign:
        raise VolumeCorrupt(
            "refusing to start: unknown disk image(s) do not match any registered member",
            detail={
                "reason": "foreign_image",
                "foreign_images": foreign,
                "registered": {
                    slot: {"member_id": p.member_id, "image": p.image}
                    for slot, p in registry.profiles.items()
                },
            },
        )
    # Every image is known but no complete assignment exists.  A slot with
    # several claimants means two images carry the same member identity while
    # another member is absent; report that explicitly rather than opaquely.
    covered: dict[str, list[str]] = {}
    for image, slots in edges.items():
        for slot in slots:
            covered.setdefault(slot, []).append(image)
    for slot in sorted(covered, key=lambda value: (-len(covered[value]), value)):
        claimants = sorted(covered[slot])
        if len(claimants) > 1:
            raise VolumeCorrupt(
                "refusing to start: two disk images claim the same member identity",
                detail={
                    "reason": "duplicate_member",
                    "slot": slot,
                    "member_id": registry.profile(slot).member_id,
                    "duplicate_images": claimants,
                },
            )
    # Fallback (no slot has two direct claimants, yet the matching is partial).
    raise VolumeCorrupt(
        "refusing to start: disk images cannot be assigned to the registered members",
        detail={"reason": "unassignable", "edges": {k: sorted(v) for k, v in edges.items()}},
    )


def recognize(
    *,
    registry: MemberRegistry,
    data_dir: Path,
    stripes: int,
    candidates: list[str],
    needed_slots: set[str],
    record: JournalRecord | None = None,
) -> dict[str, str]:
    """Assign candidate image filenames to logical slots.

    ``candidates`` are the online images eligible for content inspection; the
    offline member's image is deliberately excluded and never opened.
    Returns a ``{slot: image}`` mapping covering every candidate.  Raises
    :class:`VolumeCorrupt` for a foreign image or a duplicated identity.
    """

    candidates = sorted(candidates)
    fingerprints = {
        image: read_block_hashes(data_dir / image, stripes) for image in candidates
    }
    edges = slot_edges(registry, fingerprints, record)

    if any(not slots for slots in edges.values()):
        _fail_on_bad_membership(edges, registry, needed_slots)

    # Enumerate every injective assignment of the candidate images to slots
    # they may legitimately occupy.  When fewer images than slots exist the
    # assignment covers only the present members; the caller is responsible for
    # rebuilding the missing one.  Prefer each member's last-registered
    # filename.  The search is tiny (at most three images), so brute force is
    # clear and deterministic.
    slots = sorted(needed_slots)
    images = candidates
    best: tuple[int, list[tuple[str, str]], dict[str, str]] | None = None
    for chosen_slots in permutations(slots, len(images)):
        assignment = {slot: image for slot, image in zip(chosen_slots, images)}
        if all(slot in edges[image] for slot, image in assignment.items()):
            filename_hits = sum(
                1
                for slot, image in assignment.items()
                if (registry.profile(slot).original_image or registry.profile(slot).image)
                == image
            )
            sortable = sorted(assignment.items())
            if best is None or (filename_hits, sortable) > (best[0], best[1]):
                best = (filename_hits, sortable, assignment)
    if best is None:
        _fail_on_bad_membership(edges, registry, needed_slots)
    return dict(best[2])
