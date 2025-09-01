#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


def debug(msg: str) -> None:
    print(f"[pdf2audiobook] {msg}")


# ------------------------------ PDF Extraction ------------------------------
def extract_text_from_pdf(path: str, start_page: Optional[int], end_page: Optional[int]) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        debug("Missing dependency: pypdf. Install with `pip install pypdf`.\n" + str(e))
        raise SystemExit(2)

    reader = PdfReader(path)

    total_pages = len(reader.pages)
    if start_page is None:
        start_page = 1
    if end_page is None:
        end_page = total_pages

    if start_page < 1 or end_page < start_page or end_page > total_pages:
        raise SystemExit(f"Invalid page range: {start_page}-{end_page} for PDF with {total_pages} pages")

    chunks: List[str] = []
    for i in range(start_page - 1, end_page):
        try:
            page = reader.pages[i]
            text = page.extract_text() or ""
        except Exception:
            text = ""
        # Normalize whitespace per page
        text = normalize_text(text)
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def normalize_text(text: str) -> str:
    # Replace multiple spaces/newlines with single spaces/newlines, keep paragraphs
    # First, normalize Windows newlines
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse more than 2 newlines to exactly 2 (paragraph break)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Collapse internal whitespace runs
    t = re.sub(r"[ \t\f\v]+", " ", t)
    # Trim lines
    t = "\n".join(s.strip() for s in t.splitlines())
    # Remove empty leading/trailing lines
    t = t.strip()
    return t


# ------------------------------- Text Chunking ------------------------------
def split_into_chunks(text: str, target_chars: int = 4000, hard_limit: int = 6500) -> List[str]:
    """
    Split text into chunks that are roughly target_chars long, not exceeding hard_limit.
    Prefer splitting at sentence or paragraph boundaries.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush_current():
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current).strip())
            current = []
            current_len = 0

    for para in paragraphs:
        # If paragraph is longer than hard_limit, split by sentences.
        if len(para) > hard_limit:
            sentences = split_sentences(para)
            for s in sentences:
                if current_len + len(s) + 1 > hard_limit:
                    flush_current()
                if current_len + len(s) + 1 > target_chars and current:
                    flush_current()
                current.append(s)
                current_len += len(s) + 1
        else:
            if current_len + len(para) + 2 > hard_limit:
                flush_current()
            if current_len + len(para) + 2 > target_chars and current:
                flush_current()
            current.append(para)
            current_len += len(para) + 2

    flush_current()
    return chunks


def split_sentences(text: str) -> List[str]:
    # Naive sentence splitter when punkt isn't available.
    # Splits on .!? followed by space and a capital letter.
    # Keeps the delimiter with the sentence.
    parts = re.split(r"([.!?])", text)
    sents: List[str] = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            sents.append((parts[i] + parts[i + 1]).strip())
        else:
            if parts[i].strip():
                sents.append(parts[i].strip())
    return sents


# ------------------------------- TTS Backends -------------------------------
class Backend:
    MAC_SAY = "mac_say"
    ESPEAK = "espeak"
    PYTTSX3 = "pyttsx3"


def detect_backend(preferred: str = "auto") -> Tuple[str, Optional[str]]:
    if preferred != "auto":
        return preferred, None
    if sys.platform == "darwin" and shutil.which("say"):
        return Backend.MAC_SAY, None
    if shutil.which("espeak-ng"):
        return Backend.ESPEAK, "espeak-ng"
    if shutil.which("espeak"):
        return Backend.ESPEAK, "espeak"
    try:
        import importlib
        importlib.import_module("pyttsx3")
        return Backend.PYTTSX3, None
    except Exception:
        pass
    raise SystemExit("No TTS backend found. Install pyttsx3 or have `say` (macOS) or `espeak(-ng)` available.")


@dataclass
class TTSOptions:
    voice: Optional[str] = None
    rate: Optional[int] = None  # words per minute


def synthesize_chunks(
    chunks: List[str],
    output_dir: str,
    backend: str,
    espeak_bin: Optional[str],
    opts: TTSOptions,
) -> Tuple[List[str], str]:
    os.makedirs(output_dir, exist_ok=True)
    audio_files: List[str] = []
    if backend == Backend.MAC_SAY:
        ext = ".aiff"
        for idx, text in enumerate(chunks, 1):
            a_path = os.path.join(output_dir, f"chunk_{idx:05d}{ext}")
            t_path = os.path.join(output_dir, f"chunk_{idx:05d}.txt")
            with open(t_path, "w", encoding="utf-8") as f:
                f.write(text)
            cmd = ["say", "-o", a_path, "-f", t_path]
            if opts.voice:
                cmd.extend(["-v", opts.voice])
            if opts.rate:
                cmd.extend(["-r", str(opts.rate)])
            run(cmd)
            audio_files.append(a_path)
        return audio_files, ext

    if backend == Backend.ESPEAK:
        bin_name = espeak_bin or "espeak-ng"
        if not shutil.which(bin_name):
            raise SystemExit("Could not find espeak/espeak-ng in PATH")
        ext = ".wav"
        for idx, text in enumerate(chunks, 1):
            a_path = os.path.join(output_dir, f"chunk_{idx:05d}{ext}")
            t_path = os.path.join(output_dir, f"chunk_{idx:05d}.txt")
            with open(t_path, "w", encoding="utf-8") as f:
                f.write(text)
            cmd = [bin_name, "-w", a_path, "-f", t_path]
            if opts.voice:
                cmd.extend(["-v", opts.voice])
            if opts.rate:
                cmd.extend(["-s", str(opts.rate)])
            run(cmd)
            audio_files.append(a_path)
        return audio_files, ext

    if backend == Backend.PYTTSX3:
        try:
            import importlib
            pyttsx3 = importlib.import_module("pyttsx3")
        except Exception as e:
            raise SystemExit("pyttsx3 is not installed. Install with `pip install pyttsx3`.") from e

        engine = pyttsx3.init()
        if opts.rate:
            try:
                engine.setProperty("rate", int(opts.rate))
            except Exception:
                pass
        if opts.voice:
            try:
                voices = engine.getProperty("voices")
                match = choose_voice(voices, opts.voice)
                if match is not None:
                    engine.setProperty("voice", match.id)
            except Exception:
                pass
        ext = ".wav"
        files: List[str] = []
        for idx, text in enumerate(chunks, 1):
            a_path = os.path.join(output_dir, f"chunk_{idx:05d}{ext}")
            engine.save_to_file(text, a_path)
            files.append(a_path)
        engine.runAndWait()
        return files, ext

    raise SystemExit(f"Unknown backend: {backend}")


def choose_voice(voices: Iterable, query: str):
    q = query.lower()
    best = None
    for v in voices:
        name = getattr(v, "name", "") or ""
        id_ = getattr(v, "id", "") or ""
        if q in name.lower() or q in id_.lower():
            best = v
            break
    return best


def run(cmd: List[str]) -> None:
    debug("Running: " + " ".join(shlex_quote(arg) for arg in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:  # pragma: no cover
        raise SystemExit(f"Command failed: {cmd}\n{e}")


def shlex_quote(s: str) -> str:
    # Minimal portable shell quoting
    if re.fullmatch(r"[A-Za-z0-9_./:\-]+", s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


# ----------------------------- Audio Concatenation --------------------------
def merge_audio_files(files: List[str], output_file: str) -> None:
    if not files:
        raise SystemExit("No audio files to merge")
    ext = os.path.splitext(files[0])[1].lower()
    if not all(os.path.splitext(f)[1].lower() == ext for f in files):
        raise SystemExit("All chunk files must have the same extension to merge")

    if ext == ".wav":
        merge_wav(files, output_file)
    elif ext == ".aiff" or ext == ".aif":
        merge_aiff(files, output_file)
    else:
        raise SystemExit(f"Unsupported audio format for merge: {ext}")


def merge_wav(files: List[str], output_file: str) -> None:
    import wave
    with wave.open(files[0], "rb") as wf0:
        params = wf0.getparams()
        frames = [wf0.readframes(wf0.getnframes())]
    for fp in files[1:]:
        with wave.open(fp, "rb") as wfi:
            if wfi.getparams()[:3] != params[:3]:  # nchannels, sampwidth, framerate
                raise SystemExit("WAV parameters differ between chunks; cannot merge.")
            frames.append(wfi.readframes(wfi.getnframes()))
    with wave.open(output_file, "wb") as out:
        out.setparams(params)
        for fr in frames:
            out.writeframes(fr)


def merge_aiff(files: List[str], output_file: str) -> None:
    import aifc
    with aifc.open(files[0], "rb") as af0:
        nchannels = af0.getnchannels()
        sampwidth = af0.getsampwidth()
        framerate = af0.getframerate()
        comptype = af0.getcomptype()
        compname = af0.getcompname()
        frames = [af0.readframes(af0.getnframes())]
    for fp in files[1:]:
        with aifc.open(fp, "rb") as afi:
            if (
                afi.getnchannels() != nchannels
                or afi.getsampwidth() != sampwidth
                or afi.getframerate() != framerate
                or afi.getcomptype() != comptype
            ):
                raise SystemExit("AIFF parameters differ between chunks; cannot merge.")
            frames.append(afi.readframes(afi.getnframes()))
    with aifc.open(output_file, "wb") as out:
        out.setnchannels(nchannels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.setcomptype(comptype, compname)
        for fr in frames:
            out.writeframes(fr)


# -------------------------------- Voice Listing -----------------------------
def list_voices(backend: str, espeak_bin: Optional[str]) -> None:
    if backend == Backend.MAC_SAY:
        # `say -v ?` lists voices
        subprocess.run(["say", "-v", "?"], check=False)
        return
    if backend == Backend.ESPEAK:
        bin_name = espeak_bin or ("espeak-ng" if shutil.which("espeak-ng") else "espeak")
        subprocess.run([bin_name, "-v", "?"])
        return
    if backend == Backend.PYTTSX3:
        try:
            import importlib
            pyttsx3 = importlib.import_module("pyttsx3")
        except Exception as e:
            raise SystemExit("pyttsx3 is not installed. Install with `pip install pyttsx3`.") from e
        engine = pyttsx3.init()
        for v in engine.getProperty("voices"):
            print(f"- id: {getattr(v, 'id', '')}\n  name: {getattr(v, 'name', '')}\n  lang: {getattr(v, 'languages', '')}")
        return
    raise SystemExit(f"Unknown backend: {backend}")


# ----------------------------------- CLI -----------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a PDF into an audiobook (WAV/AIFF)")
    sub = p.add_subparsers(dest="cmd")

    # list-voices
    lp = sub.add_parser("list-voices", help="List available voices for the detected or chosen backend")
    lp.add_argument("--backend", default="auto", choices=["auto", Backend.MAC_SAY, Backend.ESPEAK, Backend.PYTTSX3])

    # convert
    cp = sub.add_parser("convert", help="Convert a PDF to audio")
    cp.add_argument("pdf", help="Path to the input PDF file")
    cp.add_argument("--out", required=True, help="Path to output audio file (.wav or .aiff)")
    cp.add_argument("--pages", help="Page range like 1:10 (inclusive)")
    cp.add_argument("--start-page", type=int)
    cp.add_argument("--end-page", type=int)
    cp.add_argument("--backend", default="auto", choices=["auto", Backend.MAC_SAY, Backend.ESPEAK, Backend.PYTTSX3])
    cp.add_argument("--voice", help="Voice name/id or substring")
    cp.add_argument("--rate", type=int, help="Words per minute (backend dependent)")
    cp.add_argument("--chunk-size", type=int, default=4000, help="Approximate characters per chunk")
    cp.add_argument("--hard-limit", type=int, default=6500, help="Maximum characters per chunk")
    cp.add_argument("--keep-chunks", action="store_true", help="Keep intermediate chunk files for inspection")

    p.set_defaults(cmd="convert")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.cmd == "list-voices":
        backend, espeak_bin = detect_backend(args.backend)
        list_voices(backend, espeak_bin)
        return 0

    if args.cmd == "convert":
        pdf_path = args.pdf
        out_path = args.out
        if not os.path.exists(pdf_path):
            debug(f"Input PDF not found: {pdf_path}")
            return 2

        # Determine pages
        sp = args.start_page
        ep = args.end_page
        if args.pages:
            try:
                sp_s, ep_s = args.pages.split(":", 1)
                sp = int(sp_s) if sp_s else None
                ep = int(ep_s) if ep_s else None
            except Exception:
                return die("Invalid --pages format. Use start:end, e.g. 1:10 or 5:")

        debug("Extracting text from PDF…")
        text = extract_text_from_pdf(pdf_path, sp, ep)
        if not text:
            return die("No extractable text found in the given PDF. If it is scanned, OCR is required (not supported here).")

        debug("Splitting text into chunks…")
        chunks = split_into_chunks(text, target_chars=args.chunk_size, hard_limit=args.hard_limit)
        if not chunks:
            return die("Text splitting produced no chunks.")

        backend, espeak_bin = detect_backend(args.backend)
        debug(f"Using backend: {backend}")

        with tempfile.TemporaryDirectory(prefix="pdf2audio_") as tmpdir:
            out_dir = os.path.join(tmpdir, "chunks")
            debug("Synthesizing chunks to audio files…")
            files, ext = synthesize_chunks(
                chunks=chunks,
                output_dir=out_dir,
                backend=backend,
                espeak_bin=espeak_bin,
                opts=TTSOptions(voice=args.voice, rate=args.rate),
            )

            if args.keep_chunks:
                # Copy chunk files next to the output for inspection
                keep_dir = os.path.splitext(out_path)[0] + "_chunks"
                os.makedirs(keep_dir, exist_ok=True)
                for f in files:
                    shutil.copy2(f, os.path.join(keep_dir, os.path.basename(f)))
                debug(f"Kept chunks at {keep_dir}")

            # Ensure extension matches backend
            desired_ext = os.path.splitext(out_path)[1].lower()
            if not desired_ext:
                out_path = out_path + ext
            elif desired_ext != ext:
                debug(f"Output extension {desired_ext} does not match backend-produced {ext}. Using {ext}.")
                out_path = os.path.splitext(out_path)[0] + ext

            debug("Merging chunk audio into final file…")
            merge_audio_files(files, out_path)
            debug(f"Done: {out_path}")
        return 0

    return 1


def die(msg: str) -> int:
    print(f"Error: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
