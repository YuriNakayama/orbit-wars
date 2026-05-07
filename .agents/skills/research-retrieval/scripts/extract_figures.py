#!/usr/bin/env python3
"""Extract figures and tables from PDF files as images.

Usage:
    uv run --with pymupdf extract_figures.py <pdf_path> <output_dir> [--min-size 150]

Extracts:
1. Embedded images (photos, diagrams) from the PDF
2. Page regions containing figures/tables rendered as high-res PNG

Output:
    <output_dir>/
        img-001.png    # Embedded images
        img-002.png
        ...
        page-NNN-figure.png  # Full-width page crops (if figure captions detected)
        metadata.json        # Metadata for all extracted images
"""

import json
import sys
from pathlib import Path


def extract_images(pdf_path: str, output_dir: str, min_size: int = 150) -> list[dict]:
    """Extract embedded images and figure regions from a PDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    img_counter = 0

    # Pass 1: Extract embedded images
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = base_image["image"]
            width = base_image["width"]
            height = base_image["height"]
            ext = base_image["ext"]

            # Skip tiny images (icons, bullets, etc.)
            if width < min_size or height < min_size:
                continue

            img_counter += 1
            filename = f"img-{img_counter:03d}.{ext}"
            filepath = output / filename

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            results.append({
                "type": "embedded_image",
                "file": filename,
                "page": page_num + 1,
                "width": width,
                "height": height,
                "format": ext,
            })

    # Pass 2: Find pages with figure/table captions and render them as high-res PNG
    figure_pages: set[int] = set()
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        text_lower = text.lower()

        # Detect figure/table captions
        has_figure = any(
            marker in text_lower
            for marker in [
                "figure ", "fig. ", "fig ", "table ",
                "図", "表", "algorithm ",
            ]
        )
        if has_figure:
            figure_pages.add(page_num)

    for page_num in sorted(figure_pages):
        page = doc[page_num]
        # Render at 2x resolution for clarity
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)

        img_counter += 1
        filename = f"page-{page_num + 1:03d}-figures.png"
        filepath = output / filename
        pix.save(str(filepath))

        results.append({
            "type": "page_render",
            "file": filename,
            "page": page_num + 1,
            "width": pix.width,
            "height": pix.height,
            "format": "png",
            "note": "Full page render containing figure/table captions",
        })

    doc.close()

    # Write metadata
    metadata = {
        "source": pdf_path,
        "total_extracted": len(results),
        "embedded_images": sum(1 for r in results if r["type"] == "embedded_image"),
        "page_renders": sum(1 for r in results if r["type"] == "page_render"),
        "images": results,
    }

    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <pdf_path> <output_dir> [--min-size N]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    min_size = 150

    if "--min-size" in sys.argv:
        idx = sys.argv.index("--min-size")
        min_size = int(sys.argv[idx + 1])

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    results = extract_images(pdf_path, output_dir, min_size)

    print(f"Extracted {len(results)} images from {pdf_path}")
    for r in results:
        print(f"  [{r['type']}] {r['file']} (page {r['page']}, {r['width']}x{r['height']})")


if __name__ == "__main__":
    main()
