# OCR from PDFs with Florence-2

## What This Does
This script pulls text from PDFs using OCR. It converts each page into an image and runs it through Microsoft's Florence-2-large model to extract text.

## What You Need
Install these first:

```bash
pip install pymupdf pillow torch transformers
```

## How to Use

```python
text = perform_ocr_from_pdf("your_file.pdf")
print(text)
```

## How It Works
- Opens the PDF and turns pages into images.
- Loads Florence-2-large (a vision-language model from Microsoft).
- Feeds images into the model with an OCR prompt.
- Returns the extracted text.

## Speed Boost
- Uses GPU if available.
- Runs on `torch.float16` for better performance.

## Extras
- `max_new_tokens=4096` controls output length per page.
- `num_beams=3` balances speed and accuracy.
- `trust_remote_code=True` is needed to load Florence-2.

## Future Upgrades?
- Batch process multiple PDFs.
- Add logging for progress tracking.
- Let users tweak model settings via CLI.

