"""Synthetic regressions: python -B -m unittest -v test_llm. No live Ollama calls."""
import json
import socket
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz
import httpx
import numpy as np
import ollama

import rag
from chunking import recursive_chunker
from embeddings import Embedder
from llm import LLM, recent_history
from models import ChunkMetadata
from streamlit.testing.v1 import AppTest
from vector_store import VectorStore


def pdf_bytes(text):
    with fitz.open() as pdf:
        pdf.new_page()  # Blank original page must not shift the next page's number.
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
        return pdf.tobytes()


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Any attempted network connection during BGE load/encoding fails this suite.
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")):
            cls.embedder = Embedder()
            cls.embedder.embed_query("Offline smoke check")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.upload_patch = patch.object(rag, "UPLOAD_FOLDER", self.root / "uploads")
        self.upload_patch.start()
        self.addCleanup(self.upload_patch.stop)
        self.store = VectorStore(self.root / "kb")

    def ingest(self, text, name="same.pdf"):
        data = pdf_bytes(text)
        document_id = rag.get_document_id(data)
        metadata = []
        for page in rag.extract_pages(data):
            for chunk in recursive_chunker(page["text"]):
                metadata.append(ChunkMetadata(document_id, len(metadata), name, page["page_number"], chunk))
        vectors = self.embedder.embed_documents([m.text for m in metadata])
        rag.save_uploaded_file(data, document_id)
        self.store.add(vectors, metadata)
        return document_id, data, vectors, metadata

    def test_ingestion_retrieval_duplicate_and_deletion(self):
        a, data, vectors, metadata = self.ingest("Aurora uses copper shielding.")
        self.assertEqual(metadata[0].page_number, 2)
        self.assertEqual(a, rag.get_document_id(data))
        with self.assertRaises(ValueError):
            self.store.add(vectors, metadata)
        b, _, _, _ = self.ingest("Zephyr uses ceramic insulation.")
        self.assertNotEqual(a, b)
        self.assertEqual(len(self.store.get_documents()), 2)
        query = self.embedder.embed_query("What shielding does Aurora use?")
        self.assertEqual(self.store.search(query, 50)[0].metadata.document_id, a)
        self.assertEqual(VectorStore(self.root / "kb").count(), self.store.count())
        self.store.delete_document(a)
        rag.delete_uploaded_file(a, "same.pdf")
        self.assertFalse((rag.UPLOAD_FOLDER / f"{a}.pdf").exists())
        self.assertTrue((rag.UPLOAD_FOLDER / f"{b}.pdf").exists())
        self.assertTrue(all(r.metadata.document_id == b for r in self.store.search(query)))
        self.ingest("Nimbus uses steel panels.")
        ids = [(m.document_id, m.chunk_index) for m in self.store.metadata]
        self.assertEqual(len(ids), len(set(ids)))
        for document_id in list(self.store.get_documents()):
            self.store.delete_document(document_id)
        self.assertEqual(VectorStore(self.root / "kb").count(), 0)
        self.assertEqual(self.store.search(query), [])

    def test_invalid_inputs_and_legacy(self):
        for data in (b"", b"invalid pdf", pdf_bytes(""), b"x" * (rag.MAX_PDF_BYTES + 1)):
            with self.assertRaises(ValueError):
                rag.extract_pages(data)
        with fitz.open() as pdf:
            pixels = fitz.Pixmap(fitz.csRGB, fitz.Rect(0, 0, 8, 8), False)
            pixels.clear_with(255)
            pdf.new_page().insert_image(fitz.Rect(0, 0, 80, 80), stream=pixels.tobytes("png"))
            scanned = pdf.tobytes()
        with self.assertRaisesRegex(ValueError, "No extractable text"):
            rag.extract_pages(scanned)
        with fitz.open(stream=pdf_bytes("secret"), filetype="pdf") as pdf:
            encrypted = pdf.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret")
        with self.assertRaisesRegex(ValueError, "Password"):
            rag.extract_pages(encrypted)
        with self.assertRaises(ValueError):
            self.store.add(np.zeros((1, 384)), [])
        for vectors in (np.zeros(384), np.zeros((0, 384)), np.zeros((1, 2)), np.full((1, 384), np.nan)):
            with self.assertRaises(ValueError):
                self.store.add(vectors, [ChunkMetadata("a" * 64, 0, "a.pdf", 1, "text")])
        legacy = self.root / "legacy"
        legacy.mkdir()
        (legacy / "metadata.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "Legacy"):
            VectorStore(legacy)
        self.assertTrue((legacy / "metadata.json").exists())

    def test_atomic_failure_and_corruption(self):
        a, _, _, _ = self.ingest("Aurora uses copper shielding.")
        before = self.store.path.read_bytes()
        with patch("vector_store.os.replace", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                self.store.delete_document(a)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertTrue(self.store.document_exists(a))
        self.assertEqual(list(self.store.path.parent.glob("*.tmp")), [])
        with zipfile.ZipFile(self.store.path) as archive:
            raw = archive.read("index.faiss")
            manifest = json.loads(archive.read("manifest.json"))
        for change in ("configuration", "checksum", "count"):
            changed = json.loads(json.dumps(manifest))
            if change == "configuration":
                changed["embedding"]["model"] = "wrong model"
            elif change == "checksum":
                changed["index_sha256"] = "wrong checksum"
            else:
                changed["metadata"] = []
            with zipfile.ZipFile(self.store.path, "w") as archive:
                archive.writestr("index.faiss", raw)
                archive.writestr("manifest.json", json.dumps(changed))
            with self.assertRaises(ValueError):
                VectorStore(self.root / "kb")
        self.store.path.write_bytes(b"broken archive")
        with self.assertRaises(ValueError):
            VectorStore(self.root / "kb")

    def test_app_upload_and_reruns(self):
        from streamlit.elements.widgets.chat import ChatInputValue
        data = pdf_bytes("Aurora uses copper shielding.")
        upload = SimpleNamespace(name="same.pdf", size=len(data), getvalue=lambda: data)
        with patch("vector_store.VectorStore", side_effect=lambda: VectorStore(self.root / "kb")), \
                patch("embeddings.Embedder", return_value=self.embedder), \
                patch("rag.extract_pages", wraps=rag.extract_pages) as extract, \
                patch.object(LLM, "model_status", return_value="Ready locally"), \
                patch.object(LLM, "stream_answer", side_effect=lambda *args: iter(["Copper ", "[S1]"])) as generate:
            app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=60).run()
            self.assertFalse(app.exception)
            def submit(text="", files=()):
                # AppTest only serializes text. Supply the installed widget's actual value type.
                value = ChatInputValue(text=text, files=list(files), _include_files=True)
                with patch("streamlit.chat_input", side_effect=[value, None]):
                    app.run()
                self.assertFalse(app.exception)
            submit(files=[upload])
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(generate.call_count, 0)
            self.assertEqual(VectorStore(self.root / "kb").count(), 1)
            self.assertIn("Document added", app.session_state["chats"][1][-1]["content"])
            submit("What shielding?", [upload])  # Duplicate plus question uses existing vectors.
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(generate.call_args.args[2], [])
            with patch.object(LLM, "resolve_question", return_value="How is copper shielding useful?") as rewrite:
                submit("How is it useful?")
                self.assertEqual(rewrite.call_args.args[0], "How is it useful?")
                self.assertTrue(rewrite.call_args.args[1])
                self.assertEqual(app.session_state["chats"][1][-1]["retrieval_query"], "How is copper shielding useful?")
            displayed = "\n".join(text.value for text in app.text)
            metadata = VectorStore(self.root / "kb").metadata[0]
            self.assertIn(f"Chunk ID: {metadata.document_id}:0", displayed)
            self.assertTrue(any("S1 · same.pdf · Page 2" == e.label for e in app.expander))
            app.slider[0].set_value(3).run()
            app.run()
            self.assertEqual(generate.call_count, 2)
            next(b for b in app.button if b.label == "New Chat").click().run()
            self.assertEqual(app.session_state["active_chat"], 2)
            self.assertTrue(app.session_state["chats"][1])
            # Exercise real native text deserialization with file support enabled.
            app.chat_input[0].set_value("Second question").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["chats"][2][-1]["sources"])
            self.assertEqual(generate.call_args.args[2], [])
            app.button(key="chat-1").click().run()
            self.assertEqual(app.session_state["active_chat"], 1)
            app.button(key="remove-2").click().run()
            self.assertNotIn(2, app.session_state["chats"])
            self.assertEqual(VectorStore(self.root / "kb").count(), 1)
            before = generate.call_count
            invalid = SimpleNamespace(name="invalid.pdf", size=3, getvalue=lambda: b"bad")
            submit("Do not answer this", [invalid])
            self.assertEqual(generate.call_count, before)
            self.assertIn("not added", app.session_state["chats"][1][-1]["content"])
            with patch.object(LLM, "stream_answer", side_effect=ValueError("stream interrupted")):
                submit("Retry")
            self.assertIn("not completed", app.session_state["chats"][1][-1]["content"])
            next(b for b in app.button if b.label == "New Chat").click().run()
            submit("Unrelated chat", [invalid])
            unrelated = app.session_state["active_chat"]
            app.button(key="chat-1").click().run()
            next(b for b in app.button if b.label == "Confirm deletion").click().run()
            self.assertNotIn(1, app.session_state["chats"])
            self.assertTrue(app.session_state["chats"][unrelated])
            self.assertEqual(VectorStore(self.root / "kb").count(), 0)
            self.assertEqual(list(rag.UPLOAD_FOLDER.iterdir()), [])
            with patch("vector_store.os.replace", side_effect=OSError("simulated save failure")):
                submit(files=[upload])
            self.assertEqual(VectorStore(self.root / "kb").count(), 0)
            self.assertEqual(list(rag.UPLOAD_FOLDER.iterdir()), [])
            submit("Copper?", [upload])  # Successful file-and-question submission.
            self.assertEqual(VectorStore(self.root / "kb").count(), 1)
            self.assertTrue(app.session_state["chats"][app.session_state["active_chat"]][-1]["sources"])

    def test_conversational_retrieval(self):
        llm = LLM()
        _, _, _, metadata = self.ingest("LLMs summarize text. Their limitations include hallucinations.")
        source = SimpleNamespace(metadata=metadata[0])
        messages = [{"role": "user", "content": "What is an LLM?"},
                    {"role": "assistant", "content": "A Large Language Model.", "sources": [source]}]
        history = recent_history(messages)
        self.assertEqual(len(history), 2)
        contextual = messages + [{"role": "user", "content": "How is it useful?"},
            {"role": "assistant", "content": "LLMs summarize text.", "sources": [source],
             "retrieval_query": "How are Large Language Models useful?"}]
        self.assertEqual(recent_history(contextual)[-2]["content"], "How are Large Language Models useful?")
        self.assertEqual(recent_history(messages + [{"role": "user", "content": "Failed turn"},
                         {"role": "assistant", "content": "Answer not completed"}]), history)
        self.assertEqual(len(recent_history(messages * 20)), 6)
        self.assertLessEqual(sum(len(m["content"]) for m in recent_history(messages * 20)), 3900)
        with patch.object(llm.client, "chat") as chat:
            self.assertEqual(llm.resolve_question("What is solar energy?", history), "What is solar energy?")
            self.assertEqual(llm.resolve_question("What is FAISS and how is it useful?", history),
                             "What is FAISS and how is it useful?")
            self.assertEqual(llm.resolve_question("What is an LLM and why is it important?", []),
                             "What is an LLM and why is it important?")
            self.assertIsNone(llm.resolve_question("How is it useful?", []))
            self.assertIsNone(llm.resolve_question("How is it useful?", [
                {"role": "user", "content": "Compare LLMs and retrieval."},
                {"role": "assistant", "content": "Both are useful."}]))
            chat.assert_not_called()
        for original, resolved in [("How is it useful?", "How are Large Language Models useful?"),
                                   ("What are its limitations?", "What are the limitations of Large Language Models?"),
                                   ("Can you explain that in simpler terms?", "What is an LLM?")]:
            with patch.object(llm.client, "chat", return_value=SimpleNamespace(
                    message=SimpleNamespace(content=resolved))):
                query = llm.resolve_question(original, history)
                self.assertEqual(query, resolved)
                result = self.store.search(self.embedder.embed_query(query), 1)[0]
                self.assertEqual(result.metadata.document_id, metadata[0].document_id)
        for malformed in ["CLARIFY", "Here is an answer", "How is it useful?", "question?\nextra"]:
            with patch.object(llm.client, "chat", return_value=SimpleNamespace(
                    message=SimpleNamespace(content=malformed))):
                self.assertIsNone(llm.resolve_question("How is it useful?", history))
        with patch.object(llm.client, "chat", side_effect=ConnectionError("unavailable")):
            with self.assertRaisesRegex(ValueError, "standalone question"):
                llm.resolve_question("How is it useful?", history)
        with patch.object(llm.client, "chat", side_effect=lambda **kwargs: (item for item in ())) as chat:
            with self.assertRaises(ValueError):
                llm.generate_answer("How is it useful?", "[S1] LLMs summarize text.", history)
            sent = chat.call_args.kwargs["messages"]
            self.assertEqual(sent[1:3], history)
            self.assertIn("never use history as factual evidence", sent[0]["content"].lower())

    def test_ollama_errors(self):
        llm = LLM()
        for error, message in (
            (ConnectionError("offline"), "Cannot connect"),
            (ollama.ResponseError("missing", 404), "model missing"),
            (httpx.ReadTimeout("slow"), "timed out"),
            (ollama.ResponseError("failed", 500), "generation failed"),
        ):
            with patch.object(llm.client, "chat", side_effect=error):
                with self.assertRaisesRegex(ValueError, message):
                    llm.generate_answer("What shielding?", "[S1] Copper.")
        with self.assertRaises(ValueError):
            llm.generate_answer("x" * 2001, "context")
        def chunks(done=True):
            yield SimpleNamespace(message=SimpleNamespace(content="Copper "), done=False)
            yield SimpleNamespace(message=SimpleNamespace(content="[S1]"), done=done)
        with patch.object(llm.client, "chat", side_effect=lambda **kwargs: chunks()):
            self.assertEqual(llm.generate_answer("Shielding?", "Copper."), "Copper [S1]")
        with patch.object(llm.client, "chat", side_effect=lambda **kwargs: chunks(False)):
            with self.assertRaisesRegex(ValueError, "stopped before completing"):
                list(llm.stream_answer("Shielding?", "Copper."))
        def interrupted():
            yield SimpleNamespace(message=SimpleNamespace(content="Partial"), done=False)
            raise httpx.ReadTimeout("interrupted")
        with patch.object(llm.client, "chat", side_effect=lambda **kwargs: interrupted()):
            with self.assertRaisesRegex(ValueError, "timed out"):
                list(llm.stream_answer("Shielding?", "Copper."))


if __name__ == "__main__":
    unittest.main()
