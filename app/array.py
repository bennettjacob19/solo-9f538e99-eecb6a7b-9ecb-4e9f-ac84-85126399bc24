"""Three-disk XOR array (A, B, parity) over persistent 4 KiB block files.

Layout: every disk is a flat file of ``stripes * 4096`` bytes.  Stripe *i*
occupies offset ``i * 4096`` on all three disks and satisfies::

    parity[i] = data0[i] XOR data1[i]

A stripe write is a whole-stripe operation guarded by the redo journal in
``journal.py``.  Because both new data blocks are journaled, recovery can
rebuild the entire stripe deterministically; the parity block never needs to
be journaled as it is a pure XOR of the two data blocks.
"""

from __future__ import annotations

import base64
import hashlib
import os
import threading
from pathlib import Path
from typing import Any

from .config import (
    BLOCK_SIZE,
    DATA_DISKS,
    DISK_NAMES,
    Settings,
    normalize_disk,
    normalize_power_loss,
)
from .errors import (
    BlockOutOfRange,
    DegradedReadOnly,
    InvalidBlockPayload,
    InvalidConfiguration,
    ServiceCrashed,
    SimulatedPowerLoss,
    VolumeCorrupt,
)
from .journal import Journal, JournalRecord, new_record
from .membership import (
    MemberProfile,
    MemberRegistry,
    build_registry,
    load_registry,
    read_block_hashes,
    recognize,
    save_registry,
    slot_edges,
)

ZERO_BLOCK = bytes(BLOCK_SIZE)
POWER_LOSS_SCENARIOS = ("none", "data", "data0", "data1", "parity", "neither", "both")


def xor_blocks(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class XorArray:
    """A tiny XOR array with a single-slot, crash-safe redo journal."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.offline = settings.offline_disk
        self.journal = Journal(settings.journal_path, fsync=settings.fsync)
        self._lock = threading.RLock()
        self.crashed = False
        self.last_recovery: dict[str, Any] | None = None
        self.rebuilt_disks: list[str] = []
        # Logical slot -> physical backing image filename.  Membership
        # recognition makes this independent of the on-disk filenames, so two
        # data disks swapped during a repair are still addressed by their
        # original slots.
        self.slot_images: dict[str, str] = {
            name: settings.disk_path(name).name for name in DISK_NAMES
        }
        self.registry: MemberRegistry | None = None
        # Validate the journal *before* touching any disk image: a corrupt
        # intent must abort startup rather than trigger a rebuild/roll-forward.
        self._pending: JournalRecord | None = self.journal.load_pending()
        if self._pending is not None and not 0 <= self._pending.stripe < settings.stripes:
            raise VolumeCorrupt(
                "journal references a stripe outside the array",
                detail={
                    "stripe": self._pending.stripe,
                    "stripes": settings.stripes,
                },
            )
        self._ensure_disks()
        self._seq = 0
        self._recover()

    # ------------------------------------------------------------------ setup

    @property
    def path(self) -> Path:
        return Path(self.settings.data_dir)

    def _image_path(self, name: str) -> Path:
        return self.path / self.slot_images[name]

    def _ensure_disks(self) -> None:
        """Create, identify, validate or rebuild the backing files.

        A disk named on the command line as offline is never created or
        touched, so a genuinely absent disk can be simulated.  When the array
        starts healthy and exactly one image is missing, that image is rebuilt
        from the other two (``missing = other XOR parity``), which is the normal
        path after a replacement disk is inserted.

        When a member registry already exists, the physical images are first
        matched to logical slots by their content fingerprints (see
        :mod:`app.membership`): swapped filenames are followed back to the
        original slots, while a foreign equal-length image or two images with
        the same member identity abort startup read-only.
        """

        self.path.mkdir(parents=True, exist_ok=True)
        expected = self.settings.capacity_bytes
        paths = {name: self.settings.disk_path(name) for name in DISK_NAMES}
        existing = {name: path.exists() for name, path in paths.items()}

        if not any(existing.values()):
            for name in DISK_NAMES:
                if name == self.offline:
                    continue
                self._create_disk(name, expected)
            # A brand new volume: register the freshly created members under
            # their canonical slot filenames.
            self.registry = build_registry(
                {name: self.settings.disk_path(name) for name in DISK_NAMES},
                stripes=self.settings.stripes,
                offline=self.offline,
                fsync=self.settings.fsync,
            )
            save_registry(
                self.settings.members_path, self.registry, fsync=self.settings.fsync
            )
            return

        registry = load_registry(self.settings.members_path)
        if registry is None:
            # Legacy volume (pre-membership): keep its raw images untouched and
            # register the current slot filenames as the original positions.
            self._ensure_legacy(registry_paths=paths, existing=existing, expected=expected)
            return

        self.registry = registry
        self._ensure_registered(expected=expected)

    def _ensure_legacy(
        self,
        *,
        registry_paths: dict[str, Path],
        existing: dict[str, bool],
        expected: int,
    ) -> None:
        """First-time registration of a volume created before membership existed."""

        for name, present in existing.items():
            if not present or name == self.offline:
                continue
            size = registry_paths[name].stat().st_size
            if size != expected:
                raise VolumeCorrupt(
                    f"disk image {registry_paths[name].name} has an unexpected size",
                    detail={
                        "disk": name,
                        "expected_bytes": expected,
                        "actual_bytes": size,
                    },
                )

        missing = [name for name, present in existing.items() if not present]
        if self.offline is not None:
            unexpected = [name for name in missing if name != self.offline]
            if unexpected:
                raise VolumeCorrupt(
                    "cannot start: disk images are missing while another disk is offline",
                    detail={"missing": missing, "offline_disk": self.offline},
                )
        elif len(missing) > 1:
            raise VolumeCorrupt(
                "cannot start: more than one disk image is missing",
                detail={"missing": missing, "data_dir": str(self.path)},
            )
        elif len(missing) == 1:
            # Healthy replacement on a never-registered volume: rebuild first,
            # then register the whole array from its canonical filenames.
            target = missing[0]
            if target == "parity":
                source_a, source_b = DATA_DISKS
            elif target == "data0":
                source_a, source_b = "data1", "parity"
            else:
                source_a, source_b = "data0", "parity"
            self._create_disk(target, expected)
            self._rebuild_disk(target, source_a, source_b)
            self.rebuilt_disks.append(target)

        self.registry = build_registry(
            registry_paths,
            stripes=self.settings.stripes,
            offline=self.offline,
            fsync=self.settings.fsync,
        )
        save_registry(
            self.settings.members_path, self.registry, fsync=self.settings.fsync
        )

    def _ensure_registered(self, *, expected: int) -> None:
        """Identify registered images and route each one to its original slot."""

        assert self.registry is not None
        registry = self.registry
        slots = set(DISK_NAMES)
        online_slots = {slot for slot in slots if slot != self.offline}

        # Enumerate every *.img file except the journal.  A present-but-wrong-
        # size image is a foreign/corrupt member, not a missing one: rebuilding
        # over it would destroy evidence, so startup is refused read-only.
        all_images = sorted(
            entry
            for entry in self.path.iterdir()
            if entry.is_file()
            and entry.suffix == ".img"
            and entry.name != self.settings.journal_path.name
        )
        wrong_sized = sorted(entry.name for entry in all_images if entry.stat().st_size != expected)
        if wrong_sized:
            raise VolumeCorrupt(
                "refusing to start: disk image has an unexpected size and cannot "
                "belong to this array",
                detail={
                    "reason": "foreign_image",
                    "foreign_images": wrong_sized,
                    "expected_bytes": expected,
                    "actual_bytes": {
                        entry.name: entry.stat().st_size
                        for entry in all_images
                        if entry.name in wrong_sized
                    },
                },
            )
        candidates = [entry.name for entry in all_images]

        assignment: dict[str, str]
        if self.offline is None and len(candidates) == len(slots) - 1:
            # Healthy replacement of exactly one registered member: identify the
            # two survivors against their slots, then rebuild the missing one.
            assignment = recognize(
                registry=registry,
                data_dir=self.path,
                stripes=self.settings.stripes,
                candidates=candidates,
                needed_slots=set(slots),
                record=self._pending,
            )
            target = (slots - set(assignment)).pop()
            self._rebuild_registered_member(target, assignment, expected)
            assignment[target] = self.settings.disk_path(target).name
        elif self.offline is None and len(candidates) < len(slots) - 1:
            raise VolumeCorrupt(
                "cannot start: more than one registered disk image is missing",
                detail={"present": candidates, "expected_members": sorted(slots)},
            )
        elif self.offline is not None:
            assignment = self._recognize_offline(candidates, online_slots)
            missing_online = sorted(online_slots - set(assignment))
            if missing_online:
                raise VolumeCorrupt(
                    "cannot start: registered disk images are missing while another "
                    "disk is offline",
                    detail={
                        "missing_slots": missing_online,
                        "offline_disk": self.offline,
                        "present": candidates,
                    },
                )
        else:
            assignment = recognize(
                registry=registry,
                data_dir=self.path,
                stripes=self.settings.stripes,
                candidates=candidates,
                needed_slots=slots,
                record=self._pending,
            )

        self.slot_images.update(assignment)
        if self.offline is not None:
            # Keep the logical slot pointed at its known (untouched) filename
            # for the status report; its file is never opened in this mode.
            offline_profile = registry.profile(self.offline)
            self.slot_images[self.offline] = (
                offline_profile.original_image or offline_profile.image
            )
        else:
            # Refresh fingerprints (also picks up a rebuilt replacement) while
            # preserving every stable member identity.
            self._sync_registry()

    def _recognize_offline(
        self, candidates: list[str], online_slots: set[str]
    ) -> dict[str, str]:
        """Recognize online members while one slot is configured offline.

        The offline member's file must never be opened for reconstruction, but
        every present image still has to be accounted for so that a foreign
        image or a duplicated online member cannot be hidden behind the
        offline slot.

        The content edges (image -> set of slots it may legitimately be) drive
        the decision:

        1. An image matching *no* slot is foreign.  The single exception is the
           offline member's own registered file, which may be corrupted
           (``OFFLINE_DISK`` explicitly tolerates an unreadable offline disk).
        2. Each remaining file is tried as the offline member.  Excluding it
           must leave the other files uniquely assignable to the online slots;
           a duplicated online identity makes that impossible for every
           exclusion, so startup is refused with the offending images named.
        3. Among valid exclusions the offline member's own content/filename
           match wins, keeping zero-filled volumes deterministic.
        """

        assert self.registry is not None
        registry = self.registry
        fingerprints = {
            image: read_block_hashes(self.path / image, self.settings.stripes)
            for image in candidates
        }
        edges = slot_edges(registry, fingerprints, self._pending)
        offline_slot = self.offline
        offline_profile = registry.profile(offline_slot)
        offline_registered = offline_profile.original_image or offline_profile.image

        # 1. Foreign images: no recognizable slot at all.
        foreign = sorted(
            image
            for image in candidates
            if not edges.get(image) and image != offline_registered
        )
        if foreign:
            raise VolumeCorrupt(
                "refusing to start: unknown disk image(s) do not match any registered member",
                detail={
                    "reason": "foreign_image",
                    "foreign_images": foreign,
                    "offline_disk": offline_slot,
                },
            )

        def try_recognize(excluded: str | None) -> dict[str, str]:
            rest = [name for name in candidates if name != excluded]
            return recognize(
                registry=registry,
                data_dir=self.path,
                stripes=self.settings.stripes,
                candidates=rest,
                needed_slots=online_slots,
                record=self._pending,
            )

        # No offline member file present: the online files must map directly.
        try:
            return try_recognize(None)
        except VolumeCorrupt:
            pass

        # 2./3. Enumerate defensible exclusions.
        options: list[tuple[int, int, str, dict[str, str]]] = []
        duplicate_errors: list[VolumeCorrupt] = []
        for excluded in candidates:
            matched = edges.get(excluded, set())
            try:
                assignment = try_recognize(excluded)
            except VolumeCorrupt as exc:
                if exc.extra.get("reason") == "duplicate_member":
                    duplicate_errors.append(exc)
                continue
            options.append(
                (
                    1 if offline_slot in matched else 0,
                    1 if excluded == offline_registered else 0,
                    excluded,
                    assignment,
                )
            )
        if not options:
            if duplicate_errors:
                # Every possible offline designation still left a duplicated
                # online member: this is a real identity conflict, not ambiguity.
                raise duplicate_errors[0]
            # Re-run to produce the canonical membership error.
            return try_recognize(None)
        options.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return options[0][3]

    def _rebuild_registered_member(
        self,
        target: str,
        assignment: dict[str, str],
        expected: int,
    ) -> None:
        if target == "parity":
            source_a, source_b = DATA_DISKS
        elif target == "data0":
            source_a, source_b = "data1", "parity"
        else:
            source_a, source_b = "data0", "parity"
        if source_a not in assignment or source_b not in assignment:
            raise VolumeCorrupt(
                "cannot start: not enough member images remain to rebuild the missing one",
                detail={"missing": target, "present": sorted(assignment)},
            )
        self._create_disk(target, expected)
        self._rebuild_disk(target, source_a, source_b)
        self.rebuilt_disks.append(target)

    def _create_disk(self, name: str, expected: int) -> None:
        target = self.settings.disk_path(name)
        fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.ftruncate(fd, expected)
            if self.settings.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        if self.settings.fsync:
            dir_fd = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def _rebuild_disk(self, target: str, source_a: str, source_b: str) -> None:
        for index in range(self.settings.stripes):
            rebuilt = xor_blocks(self._read_raw(source_a, index), self._read_raw(source_b, index))
            self._write_raw(target, index, rebuilt)

    def _recover(self) -> None:
        """Replay a pending journal record, tolerating a missing data disk.

        When a data disk is configured offline we still commit the stripe to
        every online disk (including parity).  The journal is intentionally
        *not* cleared in that case, so a later restart without the offline flag
        can finish the job.  Repeated restarts with the same offline disk are
        idempotent: they rewrite identical bytes and leave the journal as is.
        """

        header, _state = self.journal.read_header()
        self._seq = header.seq if header is not None else 0
        record = self._pending
        self._pending = None
        if record is None:
            # Make sure a registry created while a member was offline gets its
            # content fingerprint as soon as the full volume is healthy again.
            if self.offline is None and self.registry is not None:
                self._sync_registry()
            return

        applied: list[str] = []
        if self.offline != "data0":
            self._write_raw("data0", record.stripe, record.data0)
            applied.append("data0")
        if self.offline != "data1":
            self._write_raw("data1", record.stripe, record.data1)
            applied.append("data1")
        self._write_raw("parity", record.stripe, record.parity)
        applied.append("parity")

        cleared = self.offline is None
        if cleared:
            self.journal.clear(record.seq)
            # The roll-forward is complete and durable: refresh the member
            # fingerprints so the registry describes the new content.
            self._sync_registry()
        self._seq = record.seq
        self.last_recovery = {
            "recovered": True,
            "stripe": record.stripe,
            "sequence": record.seq,
            "applied_to": applied,
            "journal_cleared": cleared,
            "pending_offline_disk": self.offline,
            "data0_sha256": sha256_hex(record.data0),
            "data1_sha256": sha256_hex(record.data1),
            "parity_sha256": sha256_hex(record.parity),
        }

    def _sync_registry(self) -> None:
        """Refresh member fingerprints from the current online image bytes.

        The registry is rewritten atomically; member identities and the
        originally registered filenames are preserved.  The offline member is
        carried over untouched because its image must not be opened.
        """

        if self.registry is None:
            return
        profiles: dict[str, MemberProfile] = {}
        for slot, profile in self.registry.profiles.items():
            if slot == self.offline:
                profiles[slot] = profile
                continue
            image = self.slot_images[slot]
            hashes = read_block_hashes(self.path / image, self.settings.stripes)
            profiles[slot] = MemberProfile(
                slot=slot,
                member_id=profile.member_id,
                image=image,
                size_bytes=self.settings.capacity_bytes,
                block_sha256=hashes,
                registered_ns=profile.registered_ns,
                original_image=profile.original_image,
            )
        self.registry = MemberRegistry(profiles=profiles, registered_ns=self.registry.registered_ns)
        save_registry(
            self.settings.members_path, self.registry, fsync=self.settings.fsync
        )

    # ----------------------------------------------------------------- state

    @property
    def mode(self) -> str:
        if self.crashed:
            return "crashed"
        return "degraded" if self.offline else "healthy"

    @property
    def writable(self) -> bool:
        return self.mode == "healthy"

    def _guard_operational(self) -> None:
        if self.crashed:
            raise ServiceCrashed(
                "the service simulated a power loss; restart it to recover",
                detail={"journal_stripe": self.journal.inspect().get("stripe")},
            )

    def _validate_index(self, index: int) -> int:
        try:
            value = int(index)
        except (TypeError, ValueError):
            raise BlockOutOfRange(f"block index {index!r} is not an integer")
        if not 0 <= value < self.settings.stripes:
            raise BlockOutOfRange(
                f"block index {value} is outside 0..{self.settings.stripes - 1}",
                detail={"index": value, "stripes": self.settings.stripes},
            )
        return value

    def _validate_disk(self, disk: str, *, allow_parity: bool = True) -> str:
        try:
            name = normalize_disk(disk, allow_parity=allow_parity)
        except ValueError as exc:
            raise InvalidConfiguration(str(exc)) from exc
        if name is None:
            raise InvalidConfiguration("a disk must be specified")
        return name

    def validate_disk_name(self, disk: str, *, allow_parity: bool = True) -> str:
        """Public wrapper used by the API for validation without I/O."""

        return self._validate_disk(disk, allow_parity=allow_parity)

    # --------------------------------------------------------------- raw I/O

    def _read_raw(self, name: str, index: int) -> bytes:
        target = self._image_path(name)
        try:
            fd = os.open(target, os.O_RDONLY)
        except OSError as exc:
            raise VolumeCorrupt(f"cannot open disk image {target.name}: {exc}") from exc
        try:
            data = os.pread(fd, BLOCK_SIZE, index * BLOCK_SIZE)
        finally:
            os.close(fd)
        if len(data) != BLOCK_SIZE:
            raise VolumeCorrupt(
                f"short read from {name} block {index}",
                detail={"disk": name, "image": target.name, "index": index, "bytes": len(data)},
            )
        return data

    def _write_raw(self, name: str, index: int, data: bytes) -> None:
        target = self._image_path(name)
        try:
            fd = os.open(target, os.O_RDWR)
        except OSError as exc:
            raise VolumeCorrupt(f"cannot open disk image {target.name}: {exc}") from exc
        try:
            os.pwrite(fd, data, index * BLOCK_SIZE)
            if self.settings.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)

    # --------------------------------------------------------------- commands

    def read_block(self, index: int, disk: str) -> dict[str, Any]:
        self._guard_operational()
        idx = self._validate_index(index)
        name = self._validate_disk(disk)

        with self._lock:
            if name == self.offline:
                other = "data1" if name == "data0" else "data0"
                other_data = self._read_raw(other, idx)
                parity = self._read_raw("parity", idx)
                data = xor_blocks(other_data, parity)
                source = "reconstructed"
            else:
                data = self._read_raw(name, idx)
                source = "native"

            return {
                "index": idx,
                "disk": name,
                "size": len(data),
                "data_b64": base64.b64encode(data).decode("ascii"),
                "sha256": sha256_hex(data),
                "source": source,
                "mode": self.mode,
                "offline_disk": self.offline,
                "consistent": None if self.offline else self._stripe_consistent(idx),
            }

    def write_block(
        self,
        index: int,
        disk: str,
        data: bytes,
        *,
        power_loss: str | None = None,
    ) -> dict[str, Any]:
        self._guard_operational()
        idx = self._validate_index(index)
        name = self._validate_disk(disk)
        if name == "parity":
            raise InvalidConfiguration(
                "the parity block is derived; write data0 or data1 instead"
            )
        if self.offline is not None:
            raise DegradedReadOnly(
                "the array is in read-only degraded mode because a data disk is offline",
                detail={"offline_disk": self.offline},
            )
        if len(data) != BLOCK_SIZE:
            raise InvalidBlockPayload(
                f"a block must be exactly {BLOCK_SIZE} bytes",
                detail={"received_bytes": len(data), "expected_bytes": BLOCK_SIZE},
            )
        try:
            scenario = normalize_power_loss(power_loss)
        except ValueError as exc:
            raise InvalidConfiguration(str(exc)) from exc

        with self._lock:
            current0 = self._read_raw("data0", idx)
            current1 = self._read_raw("data1", idx)
            new0 = data if name == "data0" else current0
            new1 = data if name == "data1" else current1
            seq = self._seq + 1

            record = new_record(stripe=idx, seq=seq, data0=new0, data1=new1)
            try:
                self.journal.write_prepared(record)
            except OSError as exc:
                # No disk block has been touched yet.  A torn header is treated
                # as "no intent" by recovery, so the striped data stays as-is.
                raise VolumeCorrupt(
                    f"could not persist the recovery journal: {exc}",
                    detail={"stripe": idx, "sequence": seq, "journal": "write_failed"},
                ) from exc
            self._seq = seq

            landed: list[str] = []
            try:
                if scenario in ("data", "data0", "data1"):
                    targets = ["data0", "data1"] if scenario == "data" else [scenario]
                    for disk_name in targets:
                        self._write_raw(disk_name, idx, new0 if disk_name == "data0" else new1)
                        landed.append(disk_name)
                if scenario in ("parity", "both"):
                    self._write_raw("parity", idx, record.parity)
                    landed.append("parity")
                if scenario == "none":
                    self._write_raw("data0", idx, new0)
                    self._write_raw("data1", idx, new1)
                    self._write_raw("parity", idx, record.parity)
                    landed.extend(("data0", "data1", "parity"))
                    self.journal.clear(seq)
                    # Durable write completed: make the member fingerprints
                    # describe the new content before acknowledging the client.
                    self._sync_registry()
                    return {
                        "index": idx,
                        "disk": name,
                        "size": len(data),
                        "data_b64": base64.b64encode(data).decode("ascii"),
                        "sha256": sha256_hex(data),
                        "stripe_sha256": {
                            "data0": sha256_hex(new0),
                            "data1": sha256_hex(new1),
                            "parity": sha256_hex(record.parity),
                        },
                        "sequence": seq,
                        "journal_cleared": True,
                        "power_loss": "none",
                        "mode": self.mode,
                    }
            except (OSError, VolumeCorrupt) as exc:
                self.crashed = True
                message = exc.message if isinstance(exc, VolumeCorrupt) else str(exc)
                raise VolumeCorrupt(
                    f"disk write failed and the array was stopped: {message}",
                    detail={
                        "stripe": idx,
                        "sequence": seq,
                        "landed": landed,
                        "journal": "prepared",
                        "restart_required": True,
                    },
                ) from exc

            self.crashed = True
            raise SimulatedPowerLoss(
                f"simulated power loss after {scenario} blocks landed; restart the "
                "service to recover",
                detail={
                    "stripe": idx,
                    "sequence": seq,
                    "scenario": scenario,
                    "landed": landed,
                    "journal": "prepared",
                    "restart_required": True,
                },
            )

    # ------------------------------------------------------------ inspection

    def _stripe_consistent(self, index: int) -> bool:
        d0 = self._read_raw("data0", index)
        d1 = self._read_raw("data1", index)
        p = self._read_raw("parity", index)
        return xor_blocks(xor_blocks(d0, d1), p) == ZERO_BLOCK

    def verify(self) -> dict[str, Any]:
        self._guard_operational()
        with self._lock:
            if self.offline is not None:
                return {
                    "mode": self.mode,
                    "checked": False,
                    "consistent": None,
                    "reason": "a data disk is offline; on-disk XOR cannot be verified",
                    "stripes": [],
                }
            stripes = []
            for idx in range(self.settings.stripes):
                ok = self._stripe_consistent(idx)
                stripes.append({"index": idx, "consistent": ok})
            return {
                "mode": self.mode,
                "checked": True,
                "consistent": all(item["consistent"] for item in stripes),
                "stripes": stripes,
            }

    def list_blocks(self, disk: str, *, include_data: bool = False) -> dict[str, Any]:
        """List one disk's blocks, reconstructing the offline disk if needed."""

        self._guard_operational()
        name = self._validate_disk(disk)
        with self._lock:
            blocks = []
            for idx in range(self.settings.stripes):
                if name == self.offline:
                    other = "data1" if name == "data0" else "data0"
                    data = xor_blocks(
                        self._read_raw(other, idx), self._read_raw("parity", idx)
                    )
                    source = "reconstructed"
                else:
                    data = self._read_raw(name, idx)
                    source = "native"
                entry: dict[str, Any] = {
                    "index": idx,
                    "sha256": sha256_hex(data),
                    "size": len(data),
                    "source": source,
                    "consistent": None if self.offline else self._stripe_consistent(idx),
                }
                if include_data:
                    entry["data_b64"] = base64.b64encode(data).decode("ascii")
                blocks.append(entry)
            return {
                "mode": self.mode,
                "disk": name,
                "blocks": blocks,
                "include_data": include_data,
            }

    def member_status(self) -> list[dict[str, Any]]:
        """Logical-slot -> physical image mapping shown by ``GET /api/status``."""

        members: list[dict[str, Any]] = []
        for slot in DISK_NAMES:
            image = self.slot_images.get(slot, self.settings.disk_path(slot).name)
            profile = self.registry.profile(slot) if self.registry else None
            if slot == self.offline:
                state = "offline"
            elif self.rebuilt_disks and slot in self.rebuilt_disks:
                state = "rebuilt"
            else:
                state = "online"
            members.append(
                {
                    "slot": slot,
                    "state": state,
                    "image": image,
                    "image_path": str(self.path / image),
                    "member_id": profile.member_id if profile else None,
                    "registered_image": (
                        profile.original_image or profile.image
                    )
                    if profile
                    else None,
                    "swapped": bool(
                        profile and (profile.original_image or profile.image) != image
                    ),
                    "size_bytes": profile.size_bytes
                    if profile
                    else self.settings.capacity_bytes,
                    "block_fingerprints": len(profile.block_sha256)
                    if profile
                    else 0,
                }
            )
        return members

    def status(self) -> dict[str, Any]:
        with self._lock:
            journal_info = self.journal.inspect()
            return {
                "mode": self.mode,
                "writable": self.writable,
                "offline_disk": self.offline,
                "stripes": self.settings.stripes,
                "block_size": BLOCK_SIZE,
                "capacity_bytes": self.settings.capacity_bytes,
                "data_dir": str(self.settings.data_dir),
                "disks": list(DISK_NAMES),
                "members": self.member_status(),
                "sequence": self._seq,
                "journal": journal_info,
                "last_recovery": self.last_recovery,
                "rebuilt_disks": list(self.rebuilt_disks),
            }

    def journal_info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "sequence": self._seq,
                **self.journal.inspect(),
            }
