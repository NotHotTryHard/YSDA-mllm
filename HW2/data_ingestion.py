from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests


DEFAULT_PDF_URLS = (
    "https://publichealth.hsc.wvu.edu/media/5553/russian-history-part-i.pdf",
    "https://publichealth.hsc.wvu.edu/media/5589/history-of-russia-part-2.pdf",
)


def download_pdfs(
    pdf_urls: tuple[str, ...] | list[str],
    data_dir: Path,
    timeout: int = 60,
) -> list[Path]:
    """Download the PDFs into the local data directory."""
    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths = []
    for pdf_url in pdf_urls:
        filename = Path(urlparse(pdf_url).path).name
        if not filename:
            raise ValueError(f"Error: {pdf_url}")

        target_path = data_dir / filename
        response = requests.get(pdf_url, timeout=timeout)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        downloaded_paths.append(target_path)

    return downloaded_paths


if __name__ == "__main__":
    download_pdfs(DEFAULT_PDF_URLS, Path("."))
