# Document tamper model

Trains a small image classifier (ResNet-18) that looks at a photo of an ID card and answers:

- is it **genuine**, or was one field **edited**?
- if edited, **which field**: `name`, `dob`, `expiry` or `id_number`

`sample_preview.jpg` shows what the generated training cards look like.

## Read this first: what this model is and is not

- It is trained on **synthetic cards** that this folder draws itself (a made-up country, random names,
  no real person). That is why no privacy-sensitive data is needed.
- On synthetic test images it will probably score close to 100%. **That number says nothing about real IDs.**
  Real cards have other layouts, holograms, fonts and camera conditions, so expect the model to be
  unreliable on them until you add real examples (see "Making it work on real IDs").
- It is a good **prototype and demo**. Do not use it alone to make real decisions about people.

## Steps (local computer)

```bash
cd ml
pip install -r requirements-train.txt
python make_dataset.py --preview          # writes data/preview.jpg - open it and check the cards look sensible
python make_dataset.py                    # 500 images per class to train, 100 to validate, 100 to test
python train.py --workers 0
```

On a CPU this can take 30-60 minutes. To make it faster: `python train.py --img-h 192 --img-w 288 --epochs 6`.

## Steps (free GPU on Google Colab, about 10 minutes)

1. Open colab.research.google.com, make a new notebook, then Runtime > Change runtime type > **T4 GPU**.
2. Zip the `ml` folder, and upload the zip in the Files panel on the left.
3. Run these cells:

```python
!unzip -q ml.zip
%cd ml
!pip install -q pillow numpy
!python make_dataset.py --per-class 1500 --val-per-class 200 --test-per-class 200
!python train.py --epochs 10
```

4. Download `document_tamper.pt` and `metrics.json` from the Files panel.

## Using the trained model in the backend

1. Copy `document_tamper.pt` into `Aibackend/models/` (next to `document_check.py`).
2. Install PyTorch for the backend: `pip install -r requirements-doc.txt`
3. Restart the backend. The log says "Document tamper model loaded", and the API result
   shows `"analysis_mode": "model"` in `document_check`.

If the `.pt` file is missing or PyTorch is not installed, the document check stays in **mock mode**
(random results), so nothing breaks. To switch the model off again, just move the `.pt` file away.

## Reading the training report

- **forgeries caught (recall)**: of the edited cards, how many were flagged.
- **genuine wrongly flagged**: of the untouched cards, how many were flagged by mistake. Keep this low.
- **right field named when caught**: how often `flagged_field` was correct.

## Making it work on real IDs

1. Collect real, **legally usable** cards (your own consented samples, or public research datasets of
   mock/forged IDs such as the MIDV family and SIDTD. Check each licence before using them).
2. Put them in the same folders: untouched cards in `data/train/genuine`, forged ones in the folder of the
   field that was edited. Keep some in `data/val` and `data/test` that the model never trains on.
3. Run `python train.py --epochs 5 --lr 3e-4` again and read the test report on the **real** images.
   That report, not the synthetic one, tells you whether it is good enough.

## Files

| File | What it does |
|---|---|
| `make_dataset.py` | Draws fake ID cards, forges one field in some of them, simulates a phone camera |
| `train.py` | Trains the classifier and writes `document_tamper.pt` and `metrics.json` |
| `requirements-train.txt` | Packages needed for the two scripts above |
