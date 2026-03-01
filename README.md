# ♻️ EcoSort: Edge-Cloud Cascade Waste Classification System

Live Demo: https://students-cs-ecosort-14class-demo.hf.space/

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/Students-CS/ecosort-14class-demo)
[![Model Version](https://img.shields.io/badge/Model-v1.0-blue)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](#)

> **A student developer's journey of building an edge-optimized AI classification system, from data cleaning to serverless deployment.**

## 📌 Links
* **Interactive Web App:** [EcoSort-14Class Demo on Hugging Face](https://huggingface.co/spaces/Students-CS/ecosort-14class-demo)
* **Model Weights (Releases):** [Check out v1.0 checkponts](https://github.com/20070125li-web/ecosort-14class/releases)

---

## 📖 Project Overview

EcoSort is not just another tutorial project using standard datasets. It was built to solve a real-world problem: **How to accurately classify daily waste on edge devices with limited computing power?** Currently running on **Model v1.0**, the system uses an EfficientNet-B3 backbone trained on a highly customized, manually audited 14-class dataset. It features a lightweight local inference engine backed by a manual cloud LLM fallback mechanism.

<img width="2816" height="1536" alt="_20260227214111_32183_14" src="https://github.com/user-attachments/assets/48c5f766-5955-4b29-9053-1de4a0b98f81" />

---

## 🛠️ Development Log: The Problem-Solving Journey

As a student developer, building this system was a process of constant trial, error, and iteration. Here is how EcoSort evolved:

### Phase 1: The Data Trap & The 14-Class Taxonomy
Initially, I used the standard **TrashNet** dataset (6 classes) and achieved a 0.96 accuracy with ResNet. However, real-world tests failed miserably. I realized the high accuracy was a "fake" metric caused by severe class imbalance (the 'trash' class was only 5.4%) and lack of intra-class variance.

I then crawled a 44-class dataset but found many categories completely irrelevant to daily waste. **My solution? I spent over 15 days downloading, manually auditing, cleaning, and re-mapping thousands of images.** I designed a highly practical **14-class taxonomy** based on physical properties and recycling policies. For example:
* `brick` + `ceramic` ➔ `brick_ceramic` (similar texture and disposal method)
* `cigarette`, `diaper`, `mask`, `tissue` ➔ `hygiene_contaminated`
* `battery`, `bulb`, `medicine` ➔ Detailed hazardous categories (`haz_battery`, `haz_device`, `haz_medicine`)

### Phase 2: Edge Computing Constraints
My goal is to eventually deploy this on smart trash cans. Initially, I used ResNet-50, but it was too heavy for edge inference:
* **ResNet-50:** ~25.6M parameters, ~4.1G FLOPs.
* **EfficientNet-B3 (My Choice):** ~12M parameters, ~1.8G FLOPs.

By leveraging Compound Scaling, EfficientNet-B3 cut the computational cost by more than half while maintaining feature extraction capabilities. My customized dataset + EfficientNet achieved a stable validation accuracy of **91.27%** and an F1 score of **0.9259**.

### Phase 3: Handling Image Distortion (Letterbox)
I noticed the model struggled with items of extreme aspect ratios (like long chopsticks or wide cardboard). Standard resizing squashes these features. I implemented a **Letterbox padding** step in the preprocessing pipeline: scaling the longest side and padding the shortest side with solid black (0,0,0) pixels. This significantly improved the model's robustness to diverse image dimensions.

### Phase 4: The Deployment Hustle (Serverless Architecture)
Deploying the backend was the hardest engineering challenge due to lack of funds and hardware:
1.  **Streamlit/Flask Local:** Couldn't expose to the public web stably.
2.  **Render:** Failed because my model checkpoint was 149MB (Render free tier limits files to 100MB, and I couldn't bypass it without a credit card).
3.  **The Serverless Hack:** I utilized **GitHub Releases** as my free large-object storage for the 149MB model weights. Then, I hosted the front-end and computing environment on **Hugging Face Spaces**. The web app automatically fetches the latest model from GitHub upon initialization. Zero cost, zero local server needed.

### Phase 5: Human-in-the-Loop & Fallback Mechanism
Since Model v1.0 isn't perfect, I designed a fail-safe interactive UI:
* **Manual Bounding Box:** Users can draw a box on the web UI to isolate a single item from a cluttered background.
* **AI Fallback (Gemini-2.5-Flash):** If the user is unsatisfied with the local model's prediction, they can manually click a button to trigger the VLM API. The prompt strictly constrains the AI to output within my 14 categories.
* **Data Flywheel:** Users can report misclassifications, turning this webpage into an active data collection tool for training **EcoSort v2.0**.

---

## 🚀 How to Use the Demo

1.  Visit the [Hugging Face Space](https://huggingface.co/spaces/Students-CS/ecosort-14class-demo).
2.  **Upload an image** or take a photo.
3.  (Optional) Draw a bounding box around the target item.
4.  Get the prediction and recycling advice.
5.  If the result seems wrong, click the **AI Fallback** button to request a secondary analysis from Gemini VLM.

---

## 🔮 Future Roadmap
* **Real-time Video Inference:** Implement frame extraction to support live camera feeds (currently paused due to UI layout and inference latency).
* **WebAssembly (WASM):** Quantize the model (INT8/FP16) to run directly inside the browser using ONNX, eliminating server latency completely.
* **Train Model v2.0:** Utilize the edge-cases collected from the web app's reporting system.

---
*Developed by Yuzhen Tong (Students-CS)*
