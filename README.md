# Skentics · skin-disease detection

A YOLO11 model that detects and classifies 12 skin conditions. It is the first stage of **Skentics**, which later grew into a real-time vision pipeline (YOLO11 + PaliGemma + OCR + speech on live camera streams).

**Live demo:** https://irfansirkar.github.io/skin-disease-updated/

Built by **Mohammad Irfan Sirkar**.

## What's in this repo

| Path | What it is |
|---|---|
| `project/train_yolov11_skin_diseases.py` | Trains YOLO11 on the skin-disease dataset |
| `project/data.yaml` | Dataset config: 12 classes, from Roboflow (CC BY 4.0) |
| `project/app.py` | Streamlit app: upload a photo and get the prediction, confidence and an explanation |
| `location/location.html` | Location page |
| `docs/` | The live demo site (GitHub Pages) |

Classes: Acne, Chickenpox, Eczema, Monkeypox, Pimple, Psoriasis, Ringworm, basal cell carcinoma, melanoma, tinea-versicolor, vitiligo, warts.

## About the demo

The demo runs in the browser with drawn sample images and pre-computed outputs, so it needs no GPU. It has two tabs: the skin scan, and a simulated live-camera view of the later vision pipeline. **It is not medical advice.**

An earlier CNN version is in [skin-desease](https://github.com/irfansirkar/skin-desease).
