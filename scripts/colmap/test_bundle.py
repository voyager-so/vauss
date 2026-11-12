#!/usr/bin/env python3
"""Focused fail-closed policy and deterministic-packaging checks (no build/network)."""

import argparse
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import bundle


class DependencyAuditTests(unittest.TestCase):
    def test_parses_full_closure_including_loader(self):
        closure = bundle.parse_ldd("""
            linux-vdso.so.1 (0x00007ffd11af1000)
            libceres.so.4 => /build/prefix/lib/libceres.so.4 (0x00007f6941343000)
            libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f6941000000)
            /lib64/ld-linux-x86-64.so.2 (0x00007f69426a8000)
        """)
        self.assertEqual(closure["libceres.so.4"], Path("/build/prefix/lib/libceres.so.4"))
        self.assertEqual(closure["ld-linux-x86-64.so.2"], Path("/lib64/ld-linux-x86-64.so.2"))

    def test_rejects_missing_and_ambiguous_dependencies(self):
        for output in (
            "libceres.so.4 => not found",
            "statically linked",
            "libceres.so.4 => /one/libceres.so.4 (0x1)\nlibceres.so.4 => /two/libceres.so.4 (0x2)",
        ):
            with self.subTest(output=output), self.assertRaises(bundle.AuditError):
                bundle.parse_ldd(output)

    def test_unknown_and_forbidden_sonames_fail_closed(self):
        for name in ("libcholmod.so.3", "libgmp.so.10", "libjbig.so.0", "libcuda.so.1",
                     "libpython3.10.so.1.0", "libmystery.so.1", "libboost_graph.so.1.83.0"):
            with self.subTest(name=name), self.assertRaises(bundle.AuditError):
                bundle.library_rule(name)
        self.assertIsNone(bundle.library_rule("libc.so.6"))
        self.assertEqual(bundle.library_rule("libstdc++.so.6").license, bundle.GCC_EXCEPTION)
        self.assertTrue(bundle.library_rule("libquadmath.so.0").source_required)

    def test_rejects_absolute_elf_needed_paths_and_wrong_architecture(self):
        headers = "Class: ELF64\nMachine: Advanced Micro Devices X86-64\n"
        with patch.object(bundle, "command", side_effect=[headers, "(NEEDED) Shared library: [/build/libbad.so]"]):
            with self.assertRaisesRegex(bundle.AuditError, "nonrelocatable"):
                bundle.dynamic_info(Path("colmap"))
        with patch.object(bundle, "command", return_value="Class: ELF64\nMachine: AArch64\n"):
            with self.assertRaisesRegex(bundle.AuditError, "x86-64"):
                bundle.dynamic_info(Path("colmap"))

    def test_runtime_floor_only_uses_needed_versions(self):
        output = """
Version definition section '.gnu.version_d' contains 1 entry:
  Name: GLIBC_9.99
Version needs section '.gnu.version_r' contains 2 entries:
  Name: GLIBC_2.9
  Name: GLIBC_2.35
  Name: GLIBCXX_3.4.30
"""
        self.assertEqual(bundle.glibc_requirement(output), (2, 35))
        with patch.object(bundle, "command", return_value=output.replace("GLIBC_2.35", "GLIBC_2.36")):
            with self.assertRaisesRegex(bundle.AuditError, "target floor is 2.35"):
                bundle.check_glibc(Path("colmap"))

    def test_cuda_is_blocked_before_touching_inputs(self):
        with self.assertRaisesRegex(bundle.AuditError, "SiftGPU"):
            bundle.bundle(argparse.Namespace(variant="cuda"))


class SourcePackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "gcc-12_12.3.orig.tar.xz"
        self.source.write_bytes(b"complete source package fixture")
        self.dsc = self.root / "gcc-12_12.3.dsc"
        self.dsc.write_text(
            "-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA512\n\n"
            "Format: 3.0 (quilt)\nSource: gcc-12\nVersion: 12.3-1ubuntu1\n"
            f"Checksums-Sha256:\n {bundle.digest(self.source)} {self.source.stat().st_size} {self.source.name}\n"
            "-----BEGIN PGP SIGNATURE-----\nnot verified by this unit test\n"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_source_set_and_exact_source_version(self):
        files = bundle.verify_source_package(self.root, "gcc-12", "12.3-1ubuntu1")
        self.assertEqual(set(files), {self.dsc, self.source})
        with self.assertRaisesRegex(bundle.AuditError, "LGPL source package required"):
            bundle.verify_source_package(self.root, "gcc-12", "12.3-1ubuntu2")

    def test_rejects_missing_or_changed_source_archives(self):
        self.source.write_bytes(b"corruption")
        with self.assertRaisesRegex(bundle.AuditError, "checksum mismatch"):
            bundle.verify_source_package(self.root, "gcc-12", "12.3-1ubuntu1")
        self.source.unlink()
        with self.assertRaisesRegex(bundle.AuditError, "missing file"):
            bundle.verify_source_package(self.root, "gcc-12", "12.3-1ubuntu1")

    def test_rejects_source_path_escape(self):
        self.dsc.write_text(self.dsc.read_text().replace(self.source.name, "../outside.tar.xz"))
        with self.assertRaisesRegex(bundle.AuditError, "unsafe source package filename"):
            bundle.verify_source_package(self.root, "gcc-12", "12.3-1ubuntu1")


class ArchiveTests(unittest.TestCase):
    def test_archive_has_flat_paths_fixed_ownership_and_reproducible_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            (root / "bin").mkdir(parents=True)
            binary = root / "bin/colmap"
            binary.write_bytes(b"ELF fixture")
            binary.chmod(0o755)
            (root / "share").mkdir()
            (root / "share/notice").write_text("notice\n")
            first, second = Path(temporary) / "one.tar.gz", Path(temporary) / "two.tar.gz"
            bundle.deterministic_archive(root, first, 0)
            os.utime(binary, (100, 100))
            bundle.deterministic_archive(root, second, 0)
            self.assertEqual(bundle.digest(first), bundle.digest(second))
            with tarfile.open(first, "r:gz") as archive:
                self.assertIn("bin/colmap", archive.getnames())
                for member in archive.getmembers():
                    self.assertEqual((member.uid, member.gid, member.mtime), (0, 0, 0))
                    self.assertNotIn("..", Path(member.name).parts)
                self.assertEqual(archive.getmember("bin/colmap").mode, 0o755)

    def test_metadata_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source").mkdir()
            (root / "outside").write_text("secret")
            (root / "source/notice").symlink_to(root / "outside")
            with self.assertRaisesRegex(bundle.AuditError, "symlink"):
                bundle.copy_regular_tree(root / "source", root / "dest")


if __name__ == "__main__":
    unittest.main()
