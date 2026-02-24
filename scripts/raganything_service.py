#!/usr/bin/env python3
"""
RAGanything HTTP service for n8n integration.
Full RAG capabilities: document processing + semantic query.

v3.3 - Async job processing + Smart features:
- Async job queue with webhook callbacks
- Background processing for long-running PDFs
- Job status polling endpoint
- Circuit breaker and rate limiting
- PDF hash deduplication (skip already processed)
- MinerU parser with docling fallback
"""

import os
import sys
import json
import asyncio
import threading
import time
import uuid
import hashlib
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from collections import deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Add raganything to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "raganything"))

# Configuration
HOST = os.getenv("RAG_HOST", "0.0.0.0")
PORT = int(os.getenv("RAG_PORT", "8767"))
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = os.getenv("RAG_OUTPUT_BASE", str(_SERVICE_ROOT / "data" / "extracted"))
RAG_STORAGE = os.getenv("RAG_STORAGE_DIR", str(_SERVICE_ROOT / "data" / "rag_knowledge_base"))
PROCESS_TIMEOUT = 14400  # 4 hours (MinerU on CPU is very slow)
MAX_CONCURRENT_JOBS = 2  # Max parallel MinerU processes
MAX_QUEUE_DEPTH = 10  # Max jobs waiting in queue (total capacity = 2 + 10 = 12)
MAX_JOB_HISTORY = 100  # Keep last N completed jobs

# API Keys (optional - fallback to local if not available)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Embedding model config (local)
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))

# LLM config
OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "qwen3:8b")

# Vision processing (expensive - uses GPT-4o)
# Set to False to skip image analysis and save API costs
ENABLE_VISION = os.getenv("RAG_ENABLE_VISION", "false").lower() == "true"

# Rerank config (local model - free, improves query quality)
ENABLE_RERANK = os.getenv("RAG_ENABLE_RERANK", "true").lower() == "true"
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Parser config (MinerU primary, docling fallback)
PARSER_PAGE_THRESHOLD = int(os.getenv("RAG_PARSER_THRESHOLD", "15"))  # Pages
DEFAULT_PARSER = os.getenv("RAG_DEFAULT_PARSER", "mineru")

# Path mapping: host ↔ container translation (for Docker/container deployments)
# Set RAG_HOST_PATH_PREFIX and RAG_CONTAINER_PATH_PREFIX to enable path translation.
# Set RAG_PATH_MAPPINGS for multiple mappings: "container_prefix:host_prefix,..."
_HOST_PATH_PREFIX = os.getenv("RAG_HOST_PATH_PREFIX", "")
_CONTAINER_PATH_PREFIX = os.getenv("RAG_CONTAINER_PATH_PREFIX", "/workspace/")
_PATH_MAPPINGS = os.getenv("RAG_PATH_MAPPINGS", "")  # e.g. "/workspace/data/:/srv/data/,/workspace/alt/:/mnt/alt/"
_ALLOWED_PDF_ROOTS_RAW = os.getenv("RAG_ALLOWED_PDF_ROOTS", "")
ALLOW_UNSAFE_PDF_PATHS = os.getenv("RAG_ALLOW_UNSAFE_PDF_PATHS", "false").lower() == "true"
MAX_REQUEST_BODY_BYTES = int(os.getenv("RAG_MAX_REQUEST_BODY_BYTES", "1048576"))  # 1 MiB
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RAG_RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RAG_RATE_LIMIT_MAX_REQUESTS", "120"))
TRUST_PROXY_HEADERS = os.getenv("RAG_TRUST_PROXY_HEADERS", "false").lower() == "true"

# PDF hash deduplication storage
PDF_HASH_DB = os.path.join(RAG_STORAGE, "processed_pdfs.json")


def _build_allowed_pdf_roots() -> tuple[Path, ...]:
    """Build canonical allowlist for incoming PDF paths."""
    configured_roots: list[str] = []

    if _ALLOWED_PDF_ROOTS_RAW.strip():
        configured_roots.extend(p.strip() for p in _ALLOWED_PDF_ROOTS_RAW.split(",") if p.strip())
    else:
        if _HOST_PATH_PREFIX:
            configured_roots.append(_HOST_PATH_PREFIX)
        for mapping in _PATH_MAPPINGS.split(","):
            if ":" not in mapping:
                continue
            _, host_pfx = mapping.split(":", 1)
            host_pfx = host_pfx.strip()
            if host_pfx:
                configured_roots.append(host_pfx)
        configured_roots.append(str(_SERVICE_ROOT / "data"))

    roots: list[Path] = []
    seen: set[str] = set()
    for raw_root in configured_roots:
        try:
            root = Path(raw_root).expanduser().resolve(strict=False)
        except Exception:
            continue
        root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)
        roots.append(root)
    return tuple(roots)


ALLOWED_PDF_ROOTS = _build_allowed_pdf_roots()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def translate_container_to_host_path(pdf_path: str) -> str:
    """Translate container paths to host paths using configured mappings."""
    mapped_path = pdf_path

    if mapped_path.startswith(_CONTAINER_PATH_PREFIX) and _PATH_MAPPINGS:
        for mapping in _PATH_MAPPINGS.split(","):
            if ":" not in mapping:
                continue
            container_pfx, host_pfx = mapping.split(":", 1)
            container_pfx = container_pfx.strip()
            host_pfx = host_pfx.strip()
            if container_pfx and mapped_path.startswith(container_pfx):
                return mapped_path.replace(container_pfx, host_pfx, 1)

    if mapped_path.startswith(_CONTAINER_PATH_PREFIX) and _HOST_PATH_PREFIX:
        return mapped_path.replace(_CONTAINER_PATH_PREFIX, _HOST_PATH_PREFIX, 1)

    return mapped_path


def sanitize_pdf_path(pdf_path: str) -> str:
    """Validate and canonicalize pdf_path from API input."""
    if not isinstance(pdf_path, str):
        raise ValueError("pdf_path must be a string")

    raw_path = pdf_path.strip()
    if not raw_path:
        raise ValueError("pdf_path is empty")
    if "\x00" in raw_path:
        raise ValueError("pdf_path contains invalid characters")

    translated_path = translate_container_to_host_path(raw_path)
    candidate = Path(translated_path).expanduser()
    if not candidate.is_absolute():
        raise ValueError("pdf_path must be an absolute path")

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError("PDF not found")
    except OSError as exc:
        raise ValueError("Invalid pdf_path") from exc

    if not resolved.is_file():
        raise FileNotFoundError("PDF not found")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError("pdf_path must point to a .pdf file")
    if not os.access(resolved, os.R_OK):
        raise PermissionError("PDF is not readable")

    if not ALLOW_UNSAFE_PDF_PATHS and ALLOWED_PDF_ROOTS:
        allowed = any(_is_relative_to(resolved, root) for root in ALLOWED_PDF_ROOTS)
        if not allowed:
            raise PermissionError("pdf_path is outside allowed directories")

    return str(resolved)


# =============================================================================
# PDF HASH DEDUPLICATION
# =============================================================================
class PDFHashStore:
    """Persistent storage for processed PDF hashes."""

    def __init__(self, db_path: str = PDF_HASH_DB):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        """Load hash database from disk."""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, "r") as f:
                    self.hashes = json.load(f)
            else:
                self.hashes = {}
        except Exception as e:
            print(f"[HashStore] Failed to load: {e}, starting fresh")
            self.hashes = {}

    def _save(self):
        """Save hash database to disk."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, "w") as f:
                json.dump(self.hashes, f, indent=2)
        except Exception as e:
            print(f"[HashStore] Failed to save: {e}")

    def get_pdf_hash(self, pdf_path: str) -> str:
        """Calculate SHA256 hash of PDF file."""
        sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def is_processed(self, pdf_hash: str) -> bool:
        """Check if PDF hash has been processed."""
        with self.lock:
            return pdf_hash in self.hashes

    def get_existing(self, pdf_hash: str) -> Optional[dict]:
        """Get info about already processed PDF."""
        with self.lock:
            return self.hashes.get(pdf_hash)

    def mark_processed(self, pdf_hash: str, paper_id: str, output_dir: str, parser_used: str):
        """Mark PDF as processed with metadata."""
        with self.lock:
            self.hashes[pdf_hash] = {
                "paper_id": paper_id,
                "output_dir": output_dir,
                "parser": parser_used,
                "processed_at": datetime.now().isoformat(),
            }
            self._save()

    def get_stats(self) -> dict:
        """Get hash store statistics."""
        with self.lock:
            return {
                "total_processed": len(self.hashes),
                "by_parser": {
                    "mineru": sum(1 for v in self.hashes.values() if v.get("parser") == "mineru"),
                    "docling": sum(1 for v in self.hashes.values() if v.get("parser") == "docling"),
                },
            }


# =============================================================================
# PARSER ROUTER (MinerU primary, docling fallback)
# =============================================================================
def get_pdf_page_count(pdf_path: str) -> int:
    """Get number of pages in PDF."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"[ParserRouter] Failed to count pages: {e}")
        return -1  # Unknown


def select_parser(pdf_path: str, page_threshold: int = PARSER_PAGE_THRESHOLD) -> str:
    """Select parser based on PDF page count.

    Currently uses MinerU for all PDFs because:
    - Docling is only ~6% faster (3.1 vs 3.3 sec/page) - not worth extra 500MB deps
    - MinerU is already installed and working

    The page counting is preserved for logging/metrics purposes.
    """
    num_pages = get_pdf_page_count(pdf_path)

    if num_pages < 0:
        print(f"[ParserRouter] Cannot read page count, using default: {DEFAULT_PARSER}")
        return DEFAULT_PARSER

    print(f"[ParserRouter] {num_pages} pages → mineru")
    return "mineru"


# Global hash store instance
pdf_hash_store = None


def get_hash_store() -> PDFHashStore:
    """Get or create singleton hash store."""
    global pdf_hash_store
    if pdf_hash_store is None:
        pdf_hash_store = PDFHashStore()
    return pdf_hash_store


if OPENAI_API_KEY:
    print(f"[Config] OpenAI available - primary LLM: {OPENAI_MODEL}")
else:
    print(f"[Config] OpenAI not available - using Ollama: {OLLAMA_MODEL}")
print(f"[Config] Embeddings: {EMBEDDING_MODEL} (local, dim={EMBEDDING_DIM})")
print(f"[Config] Vision: {'ENABLED (GPT-4o)' if ENABLE_VISION else 'DISABLED'}")
print(f"[Config] Rerank: {'ENABLED (' + RERANK_MODEL + ')' if ENABLE_RERANK else 'DISABLED'}")


# =============================================================================
# SHARED EVENT LOOP - Fix for async object binding across threads
# =============================================================================
_shared_loop = None
_loop_thread = None
_loop_lock = threading.Lock()


def get_shared_loop():
    """Get or create a shared event loop running in a background thread."""
    global _shared_loop, _loop_thread

    with _loop_lock:
        if _shared_loop is None or not _shared_loop.is_running():
            # Create new event loop
            _shared_loop = asyncio.new_event_loop()

            def run_loop():
                asyncio.set_event_loop(_shared_loop)
                _shared_loop.run_forever()

            _loop_thread = threading.Thread(target=run_loop, daemon=True)
            _loop_thread.start()
            print("[EventLoop] Started shared event loop in background thread")

        return _shared_loop


def run_in_shared_loop(coro):
    """Run a coroutine in the shared event loop and wait for result."""
    loop = get_shared_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()  # Block until complete


class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    paper_id: str
    pdf_path: str
    status: JobStatus = JobStatus.QUEUED
    webhook_url: Optional[str] = None
    force_parser: Optional[str] = None  # Force specific parser (mineru/docling)
    force_reprocess: bool = False  # Skip hash dedup check
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: int = 0  # 0-100

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "paper_id": self.paper_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
        }


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(self, failure_threshold=3, recovery_timeout=300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = deque(maxlen=10)
        self.state = "closed"
        self.last_failure_time = None
        self.lock = threading.Lock()

    def record_failure(self):
        with self.lock:
            now = datetime.now()
            self.failures.append(now)
            self.last_failure_time = now
            recent = [f for f in self.failures if now - f < timedelta(minutes=5)]
            if len(recent) >= self.failure_threshold:
                self.state = "open"
                print(f"[CircuitBreaker] OPEN - {len(recent)} failures in 5 min")

    def record_success(self):
        with self.lock:
            self.state = "closed"

    def can_proceed(self) -> bool:
        with self.lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self.state = "half-open"
                        print("[CircuitBreaker] Half-open - allowing test request")
                        return True
                return False
            return True  # half-open allows one request

    def reset(self):
        """Manually reset circuit breaker."""
        with self.lock:
            self.failures.clear()
            self.state = "closed"
            self.last_failure_time = None
            print("[CircuitBreaker] Manually reset to CLOSED")

    def get_status(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "recent_failures": len([f for f in self.failures if datetime.now() - f < timedelta(minutes=5)]),
            }


class IpRateLimiter:
    """Simple sliding-window per-IP rate limiter."""

    def __init__(self, max_requests: int = RATE_LIMIT_MAX_REQUESTS, window_sec: int = RATE_LIMIT_WINDOW_SEC):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._requests = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client_ip: str) -> tuple[bool, int]:
        if self.max_requests <= 0 or self.window_sec <= 0:
            return True, 0

        now = time.monotonic()
        with self._lock:
            bucket = self._requests[client_ip]
            while bucket and (now - bucket[0]) >= self.window_sec:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_sec - (now - bucket[0])))
                return False, retry_after

            bucket.append(now)
            return True, 0


class AsyncJobQueue:
    """Async job queue with background processing."""

    def __init__(self, max_workers=MAX_CONCURRENT_JOBS):
        self.jobs: dict[str, Job] = {}
        self.job_history: deque = deque(maxlen=MAX_JOB_HISTORY)
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        self._active_count = 0

    def submit_job(
        self,
        paper_id: str,
        pdf_path: str,
        webhook_url: Optional[str] = None,
        force_parser: Optional[str] = None,
        force_reprocess: bool = False,
    ) -> Job:
        """Submit a new job and return immediately."""
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            job_id=job_id,
            paper_id=paper_id,
            pdf_path=pdf_path,
            webhook_url=webhook_url,
            force_parser=force_parser,
            force_reprocess=force_reprocess,
        )

        with self.lock:
            self.jobs[job_id] = job

        # Submit to thread pool
        self.executor.submit(self._process_job, job)
        print(f"[JobQueue] Submitted job {job_id} for {paper_id}")

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self.lock:
            return self.jobs.get(job_id)

    def get_active_count(self) -> int:
        """Get number of active jobs."""
        with self.lock:
            return sum(1 for j in self.jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING))

    def can_accept(self) -> bool:
        """Check if queue can accept more jobs."""
        return self.get_active_count() < (self.max_workers + MAX_QUEUE_DEPTH)

    def _process_job(self, job: Job):
        """Process job in background thread."""
        try:
            with self.lock:
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now().isoformat()

            print(f"[JobQueue] Processing {job.job_id}: {job.paper_id}")

            # Run async processing in shared event loop
            result = run_in_shared_loop(self._do_process(job))

            with self.lock:
                job.completed_at = datetime.now().isoformat()
                if result.get("success"):
                    job.status = JobStatus.COMPLETED
                    job.result = result
                    circuit_breaker.record_success()
                else:
                    job.status = JobStatus.FAILED
                    job.error = result.get("error", "Unknown error")
                    circuit_breaker.record_failure()

            print(f"[JobQueue] Completed {job.job_id}: {job.status.value}")

            # Call webhook if configured
            if job.webhook_url:
                self._call_webhook(job)

            # Move to history
            with self.lock:
                if job.job_id in self.jobs:
                    self.job_history.append(self.jobs.pop(job.job_id))

        except Exception as e:
            print(f"[JobQueue] Error processing {job.job_id}: {e}")
            with self.lock:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now().isoformat()
            circuit_breaker.record_failure()

            if job.webhook_url:
                self._call_webhook(job)

    async def _do_process(self, job: Job) -> dict:
        """Actual document processing with smart parser selection."""
        try:
            # Calculate PDF hash for deduplication
            hash_store = get_hash_store()
            pdf_hash = hash_store.get_pdf_hash(job.pdf_path)

            # Check if already processed (by hash) - skip if force_reprocess
            if not job.force_reprocess:
                existing = hash_store.get_existing(pdf_hash)
                if existing:
                    print(f"[JobQueue] SKIP (hash match): {job.paper_id} - same as {existing['paper_id']}")
                    output_dir_container = existing["output_dir"]
                    if _HOST_PATH_PREFIX and not output_dir_container.startswith(_CONTAINER_PATH_PREFIX):
                        output_dir_container = existing["output_dir"].replace(_HOST_PATH_PREFIX, _CONTAINER_PATH_PREFIX)
                    return {
                        "success": True,
                        "indexed": True,
                        "output_dir": output_dir_container,
                        "skipped": True,
                        "reason": f"Already processed as {existing['paper_id']}",
                        "parser": existing.get("parser", "unknown"),
                    }

            # Select parser (forced or auto-detected)
            if job.force_parser:
                selected_parser = job.force_parser
                print(f"[JobQueue] Forced parser: {selected_parser} for {job.paper_id}")
            else:
                selected_parser = select_parser(job.pdf_path)

            rag = get_rag_instance()

            # Create output directory
            safe_id = job.paper_id.replace("arxiv:", "").replace("/", "_").replace(":", "_")
            output_dir = os.path.join(OUTPUT_BASE, safe_id)
            os.makedirs(output_dir, exist_ok=True)

            # Download and save LaTeX macros for arXiv papers
            if job.paper_id.startswith("arxiv:"):
                try:
                    from raganything.latex_macros import extract_and_save_macros
                    arxiv_id = job.paper_id.replace("arxiv:", "")
                    macros = extract_and_save_macros(arxiv_id, output_dir)
                    if macros:
                        print(f"[JobQueue] Extracted {len(macros)} LaTeX macros for {arxiv_id}")
                except Exception as e:
                    print(f"[JobQueue] LaTeX macro extraction failed (non-fatal): {e}")

            # Try primary parser, fallback to docling if fails
            parsers_to_try = [selected_parser]
            if selected_parser != "docling":
                parsers_to_try.append("docling")  # Fallback

            last_error = None
            for parser in parsers_to_try:
                try:
                    rag.config.parser = parser
                    print(f"[JobQueue] Trying parser: {parser} for {job.paper_id}")

                    await asyncio.wait_for(
                        rag.process_document_complete(
                            file_path=job.pdf_path,
                            output_dir=output_dir,
                            parse_method="auto",
                        ),
                        timeout=PROCESS_TIMEOUT,
                    )
                    selected_parser = parser  # Update to actual parser used
                    break  # Success, exit loop
                except Exception as e:
                    last_error = e
                    if parser != parsers_to_try[-1]:
                        print(f"[JobQueue] {parser} failed: {e}, trying fallback...")
                    continue
            else:
                # All parsers failed
                raise last_error

            # Find generated markdown
            markdown_length = 0
            for f in Path(output_dir).rglob("*.md"):
                markdown_length += f.stat().st_size
                break

            # Convert path for container
            if _HOST_PATH_PREFIX:
                output_dir_container = output_dir.replace(_HOST_PATH_PREFIX, _CONTAINER_PATH_PREFIX)
            else:
                output_dir_container = output_dir

            # Mark as processed with hash
            hash_store.mark_processed(pdf_hash, job.paper_id, output_dir, selected_parser)
            print(f"[JobQueue] Marked processed: {job.paper_id} (hash: {pdf_hash[:12]}...)")

            return {
                "success": True,
                "indexed": True,
                "output_dir": output_dir_container,
                "markdown_length": markdown_length,
                "entities_extracted": True,
                "parser": selected_parser,
                "pdf_hash": pdf_hash[:12],
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Processing timeout ({PROCESS_TIMEOUT}s)",
            }
        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _call_webhook(self, job: Job):
        """Call webhook URL with job result."""
        if not job.webhook_url:
            return

        try:
            payload = {
                "job_id": job.job_id,
                "paper_id": job.paper_id,
                "status": job.status.value,
                "result": job.result,
                "error": job.error,
            }

            response = requests.post(
                job.webhook_url,
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"},
            )
            print(f"[Webhook] Called {job.webhook_url}: {response.status_code}")
        except Exception as e:
            print(f"[Webhook] Failed to call {job.webhook_url}: {e}")

    def get_status(self) -> dict:
        with self.lock:
            active = [j.to_dict() for j in self.jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)]
            return {
                "active_jobs": len(active),
                "max_workers": self.max_workers,
                "jobs": active,
                "completed_in_history": len(self.job_history),
            }


# Global instances
circuit_breaker = CircuitBreaker()
ip_rate_limiter = IpRateLimiter()
job_queue = AsyncJobQueue()
rag_instance = None
rag_lock = threading.Lock()


def get_rag_instance():
    """Get or create singleton RAGAnything instance."""
    global rag_instance

    with rag_lock:
        if rag_instance is not None:
            return rag_instance

        print("[RAG] Initializing RAGAnything with hybrid local/cloud...")

        try:
            from raganything import RAGAnything, RAGAnythingConfig
            from lightrag.utils import EmbeddingFunc
            from sentence_transformers import SentenceTransformer
            import numpy as np

            # Ensure storage directory exists
            os.makedirs(RAG_STORAGE, exist_ok=True)

            # RAGAnything configuration
            config = RAGAnythingConfig(
                working_dir=RAG_STORAGE,
                parser="mineru",
                parse_method="auto",
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )

            # ===== LOCAL EMBEDDINGS (sentence-transformers) =====
            print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}...")
            embed_model = SentenceTransformer(EMBEDDING_MODEL)

            async def local_embed(texts):
                """Generate embeddings using local sentence-transformers."""
                if isinstance(texts, str):
                    texts = [texts]
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, lambda: embed_model.encode(texts, normalize_embeddings=True)
                )
                return np.array(embeddings).tolist()

            embedding_func = EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=8192,
                func=local_embed,
            )
            print(f"[RAG] Embeddings ready: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")

            # ===== HYBRID LLM (OpenAI → Ollama fallback) =====
            def call_ollama(prompt, system_prompt=None, **kwargs):
                """Call Ollama API for local LLM."""
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = requests.post(
                    f"{OLLAMA_HOST}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["message"]["content"]

            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                """Hybrid LLM: try OpenAI first, fallback to Ollama."""
                if OPENAI_API_KEY:
                    try:
                        from lightrag.llm.openai import openai_complete_if_cache

                        return openai_complete_if_cache(
                            OPENAI_MODEL,
                            prompt,
                            system_prompt=system_prompt,
                            history_messages=history_messages,
                            api_key=OPENAI_API_KEY,
                            **kwargs,
                        )
                    except Exception as e:
                        print(f"[LLM] OpenAI failed, falling back to Ollama: {e}")

                return call_ollama(prompt, system_prompt, **kwargs)

            # ===== VISION MODEL (OpenAI only, optional) =====
            def vision_model_func(
                prompt,
                system_prompt=None,
                history_messages=[],
                image_data=None,
                messages=None,
                **kwargs,
            ):
                """Vision model - requires OpenAI (no local fallback for vision)."""
                if not ENABLE_VISION:
                    return "[Vision processing disabled - set RAG_ENABLE_VISION=true to enable]"
                if not OPENAI_API_KEY:
                    return "[Vision processing requires OpenAI API key]"

                try:
                    from lightrag.llm.openai import openai_complete_if_cache

                    if messages:
                        return openai_complete_if_cache(
                            "gpt-4o",
                            "",
                            messages=messages,
                            api_key=OPENAI_API_KEY,
                            **kwargs,
                        )
                    elif image_data:
                        return openai_complete_if_cache(
                            "gpt-4o",
                            "",
                            messages=[
                                {"role": "system", "content": system_prompt} if system_prompt else None,
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                        },
                                    ],
                                },
                            ],
                            api_key=OPENAI_API_KEY,
                            **kwargs,
                        )
                    else:
                        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
                except Exception as e:
                    print(f"[Vision] Failed: {e}")
                    return f"[Vision processing failed: {e}]"

            # ===== RERANKER (local, optional) =====
            rerank_func = None
            if ENABLE_RERANK:
                try:
                    from FlagEmbedding import FlagReranker

                    print(f"[RAG] Loading reranker: {RERANK_MODEL}...")
                    reranker = FlagReranker(RERANK_MODEL, use_fp16=True)

                    async def rerank_func(query: str, documents: list[str], top_n: int = 5):
                        """Rerank documents using local model.

                        Returns index-based results as expected by LightRAG:
                        [{"index": i, "relevance_score": score}, ...]
                        """
                        if not documents:
                            return []
                        pairs = [[query, d] for d in documents]
                        scores = reranker.compute_score(pairs)
                        if isinstance(scores, float):
                            scores = [scores]
                        indexed = sorted(
                            enumerate(scores), key=lambda x: x[1], reverse=True
                        )
                        return [
                            {"index": i, "relevance_score": s}
                            for i, s in indexed[:top_n]
                        ]

                    print(f"[RAG] Reranker ready: {RERANK_MODEL}")
                except Exception as e:
                    print(f"[RAG] Reranker failed to load: {e}")
                    rerank_func = None

            # Initialize RAGAnything with reranker
            lightrag_kwargs = {}
            if rerank_func:
                lightrag_kwargs["rerank_model_func"] = rerank_func

            rag_instance = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
                lightrag_kwargs=lightrag_kwargs,
            )

            print(f"[RAG] Initialized successfully. Storage: {RAG_STORAGE}")
            print(f"[RAG] LLM: {'OpenAI ' + OPENAI_MODEL + ' → ' if OPENAI_API_KEY else ''}Ollama {OLLAMA_MODEL}")
            return rag_instance

        except Exception as e:
            print(f"[RAG] Failed to initialize: {e}")
            import traceback

            traceback.print_exc()
            raise


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a new thread."""

    daemon_threads = True


class RAGAnythingHandler(BaseHTTPRequestHandler):
    """Handler for RAG service requests."""

    def send_json(self, status_code: int, data: dict):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_internal_error(self, _exc: Exception):
        """Log full error server-side, return generic error to clients."""
        import traceback

        traceback.print_exc()
        self.send_json(500, {"success": False, "error": "Internal server error"})

    def _get_client_ip(self) -> str:
        """Return client IP; proxy headers are opt-in."""
        if TRUST_PROXY_HEADERS:
            forwarded_for = self.headers.get("X-Forwarded-For", "")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()
        return self.client_address[0] if self.client_address else "unknown"

    def _should_rate_limit(self, path: str) -> bool:
        """Keep health/status endpoints accessible for monitoring."""
        return path not in ("/health", "/status")

    def _enforce_rate_limit(self) -> bool:
        if not self._should_rate_limit(self.path):
            return True

        client_ip = self._get_client_ip()
        allowed, retry_after = ip_rate_limiter.allow(client_ip)
        if allowed:
            return True

        self.send_json(
            429,
            {
                "success": False,
                "error": "Rate limit exceeded",
                "retry_after": retry_after,
            },
        )
        return False

    def read_json_body(self) -> dict:
        """Read JSON from request body."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc

        if content_length < 0:
            raise ValueError("Invalid Content-Length header")
        if content_length > MAX_REQUEST_BODY_BYTES:
            raise ValueError(f"Request body too large (max {MAX_REQUEST_BODY_BYTES} bytes)")

        body = self.rfile.read(content_length).decode("utf-8")
        if not body:
            return {}

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON body") from exc

    def do_POST(self):
        """Handle POST requests."""
        if not self._enforce_rate_limit():
            return

        if self.path == "/process":
            self.handle_process()
        elif self.path == "/process/sync":
            self.handle_process_sync()  # Legacy sync endpoint
        elif self.path == "/query":
            self.handle_query()
        else:
            self.send_error(404, "Not found. Use /process, /process/sync, or /query")

    def handle_process(self):
        """Submit document for async processing."""
        try:
            # Circuit breaker check
            if not circuit_breaker.can_proceed():
                self.send_json(
                    503,
                    {
                        "success": False,
                        "error": "Service temporarily unavailable (circuit breaker open)",
                        "retry_after": circuit_breaker.recovery_timeout,
                    },
                )
                return

            # Capacity check
            if not job_queue.can_accept():
                self.send_json(
                    429,
                    {
                        "success": False,
                        "error": f"Too many jobs queued (max {MAX_CONCURRENT_JOBS + MAX_QUEUE_DEPTH})",
                        "retry_after": 300,
                    },
                )
                return

            data = self.read_json_body()
            pdf_path = data.get("pdf_path", "")
            paper_id = data.get("paper_id", "")
            webhook_url = data.get("webhook_url")  # Optional callback URL
            force_parser = data.get("force_parser")  # Optional: force specific parser (mineru/docling)

            if not pdf_path or not paper_id:
                self.send_json(400, {"success": False, "error": "Missing pdf_path or paper_id"})
                return

            # Check for force_reprocess flag
            force_reprocess = data.get("force_reprocess", False)

            # DEDUPLICATION CHECK (before PDF check - cached don't need PDF)
            safe_id = paper_id.replace("arxiv:", "").replace("/", "_").replace(":", "_")
            output_dir = os.path.join(OUTPUT_BASE, safe_id)

            if not force_reprocess and os.path.exists(output_dir):
                # Check if actually has processed content (markdown files)
                md_files = list(Path(output_dir).rglob("*.md"))
                if md_files:
                    # Already processed - return cached result
                    markdown_length = sum(f.stat().st_size for f in md_files)
                    if _HOST_PATH_PREFIX:
                        output_dir_container = output_dir.replace(_HOST_PATH_PREFIX, _CONTAINER_PATH_PREFIX)
                    else:
                        output_dir_container = output_dir

                    print(f"[RAG] SKIP (cached): {paper_id} - already processed")

                    # Call webhook with cached result if configured
                    if webhook_url:
                        try:
                            requests.post(
                                webhook_url,
                                json={
                                    "job_id": "cached",
                                    "paper_id": paper_id,
                                    "status": "completed",
                                    "result": {
                                        "success": True,
                                        "cached": True,
                                        "output_dir": output_dir_container,
                                        "markdown_length": markdown_length,
                                    },
                                    "error": None,
                                },
                                timeout=30,
                            )
                        except Exception as e:
                            print(f"[Webhook] Failed for cached result: {e}")

                    self.send_json(
                        200,  # OK (not 202 Accepted)
                        {
                            "success": True,
                            "cached": True,
                            "message": "Already processed - returning cached result",
                            "paper_id": paper_id,
                            "output_dir": output_dir_container,
                            "markdown_length": markdown_length,
                            "hint": "Use force_reprocess=true to reprocess",
                        },
                    )
                    return

            pdf_path = sanitize_pdf_path(pdf_path)

            # Submit job (returns immediately)
            job = job_queue.submit_job(paper_id, pdf_path, webhook_url, force_parser, force_reprocess)

            self.send_json(
                202,  # Accepted
                {
                    "success": True,
                    "message": "Job submitted for processing",
                    "job_id": job.job_id,
                    "paper_id": paper_id,
                    "status": job.status.value,
                    "poll_url": f"/jobs/{job.job_id}",
                    "webhook_configured": webhook_url is not None,
                },
            )

        except ValueError as e:
            self.send_json(400, {"success": False, "error": str(e)})
        except PermissionError as e:
            self.send_json(403, {"success": False, "error": str(e)})
        except FileNotFoundError as e:
            self.send_json(404, {"success": False, "error": str(e)})
        except Exception as e:
            self._send_internal_error(e)

    def handle_process_sync(self):
        """Legacy synchronous processing (for small files only)."""
        # Redirect to async with warning
        self.send_json(
            400,
            {
                "success": False,
                "error": "Sync processing deprecated. Use POST /process for async processing with webhook callback.",
                "migration": {
                    "new_endpoint": "POST /process",
                    "new_params": {
                        "pdf_path": "path to PDF",
                        "paper_id": "document ID",
                        "webhook_url": "optional callback URL when processing completes",
                    },
                    "poll_endpoint": "GET /jobs/{job_id}",
                },
            },
        )

    def handle_query(self):
        """Query the knowledge graph."""
        try:
            data = self.read_json_body()
            query = data.get("query", "")
            mode = data.get("mode", "hybrid")  # hybrid, local, global
            context_only = data.get("context_only", False)

            if not query:
                self.send_json(400, {"success": False, "error": "Missing query"})
                return

            # Query RAGAnything (sync)
            result = self._query_sync(query, mode, context_only=context_only)
            self.send_json(200, result)

        except ValueError as e:
            self.send_json(400, {"success": False, "error": str(e)})
        except Exception as e:
            self._send_internal_error(e)

    def _query_sync(self, query: str, mode: str, context_only: bool = False) -> dict:
        """Query the knowledge graph (sync version)."""
        try:
            rag = get_rag_instance()

            print(f"[RAG] Query: {query[:50]}... (mode={mode}, context_only={context_only})")

            # Use sync query since our LLM functions are synchronous
            # Disable VLM enhanced to avoid async issues with sync vision func
            result = rag.query(
                query, mode=mode, vlm_enhanced=False,
                only_need_context=context_only,
            )

            response = {
                "success": True,
                "query": query,
                "mode": mode,
                "context_only": context_only,
            }
            if context_only:
                response["context"] = result
            else:
                response["answer"] = result
            return response

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"success": False, "error": "Internal query error"}

    def do_GET(self):
        """Handle GET requests."""
        if not self._enforce_rate_limit():
            return

        if self.path == "/health":
            rag_ready = rag_instance is not None
            hash_store = get_hash_store()
            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "RAGanything",
                    "version": "3.3-smart",
                    "port": PORT,
                    "rag_initialized": rag_ready,
                    "storage": RAG_STORAGE,
                    "circuit_breaker": circuit_breaker.get_status(),
                    "jobs": job_queue.get_status(),
                    "hash_store": hash_store.get_stats(),
                    "parser_router": {
                        "threshold_pages": PARSER_PAGE_THRESHOLD,
                        "default_parser": DEFAULT_PARSER,
                    },
                    "features": [
                        "async_processing",
                        "webhook_callbacks",
                        "full_rag",
                        "knowledge_graph",
                        "semantic_query",
                        "multimodal",
                        "equation_processing",
                        "pdf_hash_dedup",
                        "smart_parser_router",
                    ],
                },
            )
        elif self.path == "/status":
            self.send_json(
                200,
                {
                    "circuit_breaker": circuit_breaker.get_status(),
                    "jobs": job_queue.get_status(),
                    "rag_initialized": rag_instance is not None,
                },
            )
        elif self.path.startswith("/jobs/"):
            job_id = self.path.split("/jobs/")[1]
            job = job_queue.get_job(job_id)
            if job:
                self.send_json(200, job.to_dict())
            else:
                # Check history
                for hist_job in job_queue.job_history:
                    if hist_job.job_id == job_id:
                        self.send_json(200, hist_job.to_dict())
                        return
                self.send_json(404, {"error": f"Job {job_id} not found"})
        elif self.path == "/jobs":
            self.send_json(200, job_queue.get_status())
        elif self.path == "/reset-circuit-breaker":
            circuit_breaker.reset()
            self.send_json(200, {"success": True, "message": "Circuit breaker reset"})
        else:
            self.send_error(404, "Not found")

    def log_message(self, format, *args):
        print(f"[RAGService] {args[0]}")


class ReusableThreadingServer(ThreadingHTTPServer):
    """HTTP server with SO_REUSEADDR to avoid 'Address already in use' errors."""

    allow_reuse_address = True


def initialize_rag_at_startup():
    """Pre-initialize RAG instance at startup to load existing knowledge base."""
    import asyncio

    print("[RAG] Pre-initializing RAGAnything at startup...")
    try:
        rag = get_rag_instance()

        # Run async initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(rag._ensure_lightrag_initialized())
            if result.get("success"):
                print("[RAG] LightRAG initialized successfully from existing storage")
            else:
                print(f"[RAG] LightRAG initialization warning: {result.get('error', 'unknown')}")
        finally:
            loop.close()

        return True
    except Exception as e:
        print(f"[RAG] Pre-initialization failed (will retry on first request): {e}")
        return False


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(RAG_STORAGE, exist_ok=True)

    server = ReusableThreadingServer((HOST, PORT), RAGAnythingHandler)

    print(f"RAGanything service v3.3-smart starting on http://{HOST}:{PORT}")
    print(f"Knowledge base storage: {RAG_STORAGE}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Timeout: {PROCESS_TIMEOUT}s | Max concurrent: {MAX_CONCURRENT_JOBS}")
    print("\nEndpoints:")
    print("  POST /process      - Submit async job (returns job_id)")
    print("  GET  /jobs/{id}    - Poll job status")
    print("  GET  /jobs         - List all active jobs")
    print("  POST /query        - Query knowledge graph")
    print("  GET  /health       - Health check")
    print("  GET  /reset-circuit-breaker - Reset circuit breaker")

    # Pre-initialize RAG to load existing knowledge base
    print("\nPre-initializing RAG...")
    initialize_rag_at_startup()
    print("\nReady to serve requests!")

    server.serve_forever()
