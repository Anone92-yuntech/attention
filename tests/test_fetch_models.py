"""Download integrity and cleanup checks using tiny in-memory responses."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import fetch_models


class InterruptedResponse(io.BytesIO):
    """Return some bytes before a real-looking interrupted transport."""

    def __init__(self, payload):
        super().__init__(payload)
        self.read_count = 0

    def read(self, size=-1):
        self.read_count += 1
        if self.read_count > 1:
            raise URLError("connection interrupted")
        return super().read(min(size, 3) if size >= 0 else 3)


class FetchModelsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.payload = b"a tiny model used only by tests\n"
        self.asset = fetch_models.ModelAsset(
            relative_path="models/tiny-model.bin",
            size=len(self.payload),
            sha256=hashlib.sha256(self.payload).hexdigest(),
        )
        self.destination = self.root / self.asset.relative_path

    def write_existing(self, content):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.write_bytes(content)

    def assert_no_part_files(self):
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_valid_existing_file_is_verified_without_network(self):
        self.write_existing(self.payload)
        with patch("fetch_models.urllib.request.urlopen") as network:
            result = fetch_models.ensure_model(self.asset, root=self.root)
        self.assertEqual(result, "verified")
        network.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assert_no_part_files()

    def test_existing_corrupt_files_are_preserved_and_never_downloaded_over(self):
        corrupt_values = (b"wrong length", b"x" * len(self.payload))
        for content in corrupt_values:
            with self.subTest(content=content):
                self.write_existing(content)
                with patch("fetch_models.urllib.request.urlopen") as network:
                    with self.assertRaises(fetch_models.ModelIntegrityError):
                        fetch_models.ensure_model(self.asset, root=self.root)
                network.assert_not_called()
                self.assertEqual(self.destination.read_bytes(), content)
                self.assert_no_part_files()

    def test_successful_download_uses_release_basename_and_project_relative_path(self):
        with patch("fetch_models.urllib.request.urlopen",
                   return_value=io.BytesIO(self.payload)) as network:
            result = fetch_models.ensure_model(self.asset, root=self.root)
        self.assertEqual(result, "downloaded")
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assert_no_part_files()
        network.assert_called_once()
        request = network.call_args.args[0]
        self.assertEqual(request.full_url,
                         fetch_models.RELEASE_BASE + "/tiny-model.bin")
        self.assertEqual(network.call_args.kwargs.get("timeout"), 60)

    def test_same_length_wrong_sha_is_never_committed(self):
        content = b"x" * len(self.payload)
        with patch("fetch_models.urllib.request.urlopen",
                   return_value=io.BytesIO(content)):
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertFalse(self.destination.exists())
        self.assert_no_part_files()

    def test_wrong_size_is_never_committed_even_when_sha_matches_response(self):
        content = self.payload[:-1]
        asset = fetch_models.ModelAsset(
            relative_path=self.asset.relative_path,
            size=len(self.payload),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        with patch("fetch_models.urllib.request.urlopen",
                   return_value=io.BytesIO(content)):
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(asset, root=self.root)
        self.assertFalse(self.destination.exists())
        self.assert_no_part_files()

    def test_verify_only_missing_file_never_uses_network(self):
        with patch("fetch_models.urllib.request.urlopen") as network:
            with self.assertRaises(FileNotFoundError):
                fetch_models.ensure_model(self.asset, root=self.root, verify_only=True)
        network.assert_not_called()
        self.assertFalse(self.destination.exists())
        self.assert_no_part_files()

    def test_verify_only_checks_hash_of_existing_file(self):
        self.write_existing(b"x" * len(self.payload))
        with patch("fetch_models.urllib.request.urlopen") as network:
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(self.asset, root=self.root, verify_only=True)
        network.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"x" * len(self.payload))

    def test_interrupted_download_cleans_only_its_own_temporary_file(self):
        self.destination.parent.mkdir(parents=True)
        unrelated = self.destination.parent / "another-process.part"
        unrelated.write_bytes(b"leave this alone")
        with patch("fetch_models.urllib.request.urlopen",
                   return_value=InterruptedResponse(self.payload)):
            with self.assertRaises((URLError, fetch_models.ModelIntegrityError)):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertFalse(self.destination.exists())
        self.assertEqual(unrelated.read_bytes(), b"leave this alone")
        self.assertEqual(list(self.root.rglob("*.part")), [unrelated])

    def test_destination_created_during_download_is_not_overwritten(self):
        concurrent_content = b"another process owns this file"

        def response(*_args, **_kwargs):
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            self.destination.write_bytes(concurrent_content)
            return io.BytesIO(self.payload)

        with patch("fetch_models.urllib.request.urlopen", side_effect=response):
            with self.assertRaises(fetch_models.ModelIntegrityError):
                fetch_models.ensure_model(self.asset, root=self.root)
        self.assertEqual(self.destination.read_bytes(), concurrent_content)
        self.assert_no_part_files()

    def test_cli_verify_only_checks_each_manifest_entry(self):
        with patch.object(fetch_models, "ensure_model", return_value="verified") as ensure:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = fetch_models.main(["--verify-only"])
        self.assertEqual(code, 0)
        self.assertEqual(ensure.call_count, len(fetch_models.ASSETS))
        self.assertEqual([call.args[0] for call in ensure.call_args_list],
                         list(fetch_models.ASSETS))
        self.assertTrue(all(call.kwargs.get("verify_only")
                            for call in ensure.call_args_list))

    def test_cli_returns_nonzero_when_a_required_file_is_missing(self):
        with patch.object(fetch_models, "ensure_model", side_effect=FileNotFoundError("missing")):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = fetch_models.main(["--verify-only"])
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
