<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EcoSort Waste Classification Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: #f4f7fb; min-height: 100vh; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 14px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); padding: 30px; border: 1px solid #e7edf5; }
        h1 { color: #0f172a; margin-bottom: 10px; text-align: center; letter-spacing: 0.2px; display: flex; align-items: center; justify-content: center; gap: 10px; flex-wrap: wrap;}
        
        .lab-badge { background: #fef08a; color: #854d0e; font-size: 12px; padding: 4px 10px; border-radius: 999px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;}
        
        .subtitle { text-align: center; color: #475569; margin-bottom: 30px; font-size: 14px; }
        .status { background: #ecfdf3; padding: 12px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; color: #166534; border: 1px solid #bbf7d0; }
        .status.error { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
        .info-card { background: #f8fafc; border: 1px solid #e2e8f0; padding: 14px; border-radius: 10px; margin-bottom: 16px; }
        .info-title { color: #1e293b; font-size: 14px; font-weight: 700; margin-bottom: 8px; }
        .notice-banner { background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a; border-radius: 10px; padding: 10px 12px; font-size: 13px; line-height: 1.6; margin-bottom: 14px; }
        .chips { display: flex; flex-wrap: wrap; gap: 8px; }
        .chip { background: #eaf2ff; color: #1d4ed8; border: 1px solid #cfe0ff; border-radius: 999px; padding: 5px 10px; font-size: 12px; }
        .chip.missing { background: #fff3e8; color: #9a4c16; border-color: #ffd8b9; }
        .scope-grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
        .scope-card { border: 1px solid #e2e8f0; background: #ffffff; border-radius: 8px; padding: 8px 10px; }
        .scope-head { font-size: 13px; color: #1e3a8a; font-weight: 700; margin-bottom: 6px; }
        .scope-sub { display: flex; flex-wrap: wrap; gap: 6px; }
        .tips { color: #475569; font-size: 13px; line-height: 1.6; }
        .upload-area { border: 2px dashed #3b82f6; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.3s; background: #f8fbff; margin-bottom: 20px; }
        .upload-area:hover { border-color: #1d4ed8; background: #eff6ff; }
        .upload-area.dragover { border-color: #1d4ed8; background: #dbeafe; }
        .action-row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin-bottom: 16px; }
        .action-row .btn { min-width: 126px; }
        .upload-icon { font-size: 48px; margin-bottom: 10px; }
        .upload-text { color: #334155; font-size: 16px; }
        #fileInput { display: none; }
        .preview-container { display: none; margin-bottom: 20px; position: relative; }
        .preview-container img { max-width: 100%; max-height: 400px; border-radius: 8px; margin: 0 auto; display: block; }
        #bboxCanvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; display: none; cursor: crosshair; touch-action: none; }
        .manual-panel { display: none; margin-top: 10px; gap: 8px; flex-wrap: wrap; justify-content: center; }
        .manual-confirm { display: none; margin-top: 8px; gap: 8px; flex-wrap: wrap; justify-content: center; }
        .result-container { display: none; margin-top: 20px; }
        .result-card { background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%); color: white; padding: 24px; border-radius: 12px; text-align: center; transition: background 0.3s ease;}
        .result-class { font-size: 32px; font-weight: bold; margin-bottom: 10px; }
        .result-confidence { font-size: 18px; opacity: 0.9; }
        .result-sub { margin-top: 8px; font-size: 14px; opacity: 0.92; }
        .probabilities { margin-top: 20px; }
        .prob-item { display: flex; align-items: center; margin-bottom: 8px; }
        .prob-label { flex: 0 0 150px; font-size: 14px; color: #334155; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .prob-bar { flex: 1; height: 24px; background: #f0f0f0; border-radius: 4px; overflow: hidden; }
        .prob-fill { height: 100%; background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%); transition: width 0.5s; }
        .prob-value { flex: 0 0 60px; text-align: right; font-size: 14px; color: #334155; font-weight: bold; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #1d4ed8; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .btn { background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 16px; cursor: pointer; transition: transform 0.2s; }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .reset-btn { margin-top: 20px; display: none; }
        
        .feedback-panel { margin-top: 20px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; padding-top: 15px; border-top: 1px dashed #cbd5e1; }
        .btn-ai { background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3); }
        .btn-report { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    </style>
</head>
<body>
    <div class="container">
        <h1>EcoSort Demo <span class="lab-badge">🧪 Lab Beta</span></h1>
        <p class="subtitle">Image-based prediction deployed in experimental environment. Help us find edge cases!</p>

        <div id="status" class="status">Connecting to server...</div>

        <div class="info-card">
            <div class="info-title">Supported Label Scope (4 Coarse Classes → Fine Labels)</div>
            <div id="supportedClasses" class="scope-grid"></div>
        </div>

        <div class="info-card">
            <div class="info-title">Capture Guidelines (for better accuracy)</div>
            <div class="tips">
                1) Keep one primary object near the center with good lighting; 2) Use a clean background; 3) Capture at close range and avoid severe occlusion;
                4) Out-of-scope items will be flagged for manual review.
            </div>
        </div>

        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Click or drag an image to upload</div>
            <div style="font-size: 12px; color: #64748b; margin-top: 8px;">Find a tricky object? Test it here (Supported formats: JPG, PNG, WEBP).</div>
        </div>
        <div class="action-row">
            <button class="btn" onclick="fileInput.click()">Upload Image</button>
        </div>
        <div class="notice-banner" id="modeNotice">
            Whole-image prediction is enabled by default. For multiple items, use manual box selection.
        </div>
        <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp">

        <div class="preview-container" id="previewContainer">
            <img id="preview" alt="Preview image">
            <canvas id="bboxCanvas"></canvas>
        </div>
        <div class="manual-panel" id="manualPanel">
            <button class="btn" id="manualModeBtn" onclick="startManualMode()">Manual Box Mode</button>
            <button class="btn" id="manualRunBtn" onclick="runManualClassification()" disabled>Run Box Predictions</button>
            <button class="btn" onclick="clearManualBoxes()">Clear Boxes</button>
        </div>
        <div class="manual-confirm" id="manualConfirmPanel">
            <button class="btn" id="confirmBoxBtn" onclick="confirmPendingBox()">Confirm This Box</button>
            <button class="btn" id="cancelBoxBtn" onclick="cancelPendingBox()">Cancel This Box</button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px;" id="loadingText">Running inference...</p>
        </div>

        <div class="result-container" id="resultContainer">
            <div class="result-card" id="resultCardBg">
                <div class="result-class" id="resultClass">-</div>
                <div class="result-confidence" id="resultConfidence">Confidence: -</div>
                <div class="result-sub" id="resultSub">Fine label: -</div>
            </div>
            <div class="probabilities" id="probabilities"></div>
            
            <div class="feedback-panel" id="feedbackPanel">
                <button class="btn btn-ai" onclick="requestAIReview()">🤖 Incorrect? Request AI Review</button>
                <button class="btn btn-report" onclick="reportEdgeCase()">🐛 Report Edge Case (Log it)</button>
            </div>
        </div>

        <div style="text-align: center;">
            <button class="btn reset-btn" style="margin: 20px auto 0;" id="resetBtn" onclick="reset()">Upload Another Image</button>
        </div>
    </div>

    <script>
        const DEFAULT_API_BASE = '';

        function normalizeApiBase(url) {
            if (!url) return '';
            return String(url).trim().replace(/\/$/, '');
        }

        function resolveApiBase() {
            if (window.location.hostname.endsWith('github.io') && DEFAULT_API_BASE) {
                return normalizeApiBase(DEFAULT_API_BASE);
            }
            return normalizeApiBase(window.location.origin);
        }

        let API_BASE = resolveApiBase();
        
        // Base 14 Intermediate Classes
        let class_names = [
            'brick_ceramic', 'e_waste', 'foam', 'glass', 'haz_battery',
            'haz_device', 'haz_medicine', 'hygiene_contaminated', 'metal',
            'organic_food', 'paper_family', 'plastic', 'small_hard_items', 'textile'
        ];

        const V6_CLASS_ORDER = [
            'brick_ceramic', 'e_waste', 'foam', 'glass', 'haz_battery',
            'haz_device', 'haz_medicine', 'hygiene_contaminated', 'metal',
            'organic_food', 'paper_family', 'plastic', 'small_hard_items', 'textile'
        ];

        const LEGACY_TO_V6_FINE = {
            // v5/v4 legacy aliases
            'haz_chemical': 'plastic',
            'complex_misc': 'small_hard_items',
            'ceramic': 'brick_ceramic',
            'brick': 'brick_ceramic',

            // raw labels that are merged in v6
            'cardboard': 'paper_family',
            'paper': 'paper_family',
            'cup': 'plastic',
            'plastic shaker': 'plastic',
            'plastic_toy': 'plastic',
            'plastic_bag': 'plastic',
            'chemical': 'plastic',
            'paint': 'plastic',
            'cosmetic': 'plastic',
            'clothes': 'textile',
            'shoes': 'textile',
            'towel': 'textile',
            'bag': 'textile',
            'bone': 'organic_food',
            'cake': 'organic_food',
            'egg_shell': 'organic_food',
            'flower': 'organic_food',
            'fruit_peel': 'organic_food',
            'leftover': 'organic_food',
            'tea': 'organic_food',
            'vegetable': 'organic_food',
            'battery': 'haz_battery',
            'bulb': 'haz_device',
            'thermometer': 'haz_device',
            'medicine': 'haz_medicine',
            'lighter': 'haz_medicine',
            'cigarette': 'hygiene_contaminated',
            'mask': 'hygiene_contaminated',
            'diaper': 'hygiene_contaminated',
            'tissue': 'hygiene_contaminated',
            'wet_wipe': 'hygiene_contaminated',
            'chopsticks': 'small_hard_items',
            'pen': 'small_hard_items',
            'toothpick': 'small_hard_items',
            'comb': 'small_hard_items'
        };

        function normalizeFineLabel(name) {
            const raw = String(name || '').trim();
            const n = raw.toLowerCase();
            return LEGACY_TO_V6_FINE[n] || n;
        }

        function normalizeClassList(names) {
            const inList = Array.isArray(names) ? names : [];
            const normalized = inList.map(normalizeFineLabel);
            const uniq = [...new Set(normalized)];

            const ordered = V6_CLASS_ORDER.filter((x) => uniq.includes(x));
            const extras = uniq.filter((x) => !V6_CLASS_ORDER.includes(x));
            return [...ordered, ...extras];
        }

        const coarseLabelEN = {
            recyclable: 'Recyclable',
            hazardous: 'Hazardous',
            kitchen: 'Kitchen Waste',
            other: 'Other Waste'
        };

        let currentImageDataUrl = null;
        let lastPredictionData = null; 
        let manualMode = false;
        let isDrawing = false;
        let drawStart = null;
        let drawCurrent = null;
        let manualBoxes = [];
        let pendingBox = null;

        function coarseLabelToEN(key) {
            return coarseLabelEN[key] || key;
        }

        // ==========================================
        // 🔥 UPDATED MAPPING LOGIC
        // Strictly follows the user's mapping rules
        // ==========================================
        function inferCoarseForFineLabel(name) {
            const n = normalizeFineLabel(name);
            // 1. Recyclable
            // Includes paper, plastic (and its mapped items like chemical/paint/cosmetic containers), glass, metal, textile, e_waste
            if ([
                'paper_family', 'cardboard', 'paper',
                'plastic', 'cup', 'plastic shaker', 'plastic_toy', 'plastic_bag', 'chemical', 'paint', 'cosmetic',
                'glass',
                'metal',
                'textile', 'clothes', 'shoes', 'towel', 'bag',
                'e_waste'
            ].includes(n)) {
                return 'recyclable';
            }
            // 2. Hazardous
            // Includes battery, device (bulb, thermometer), medicine (medicine, lighter)
            if ([
                'haz_battery', 'battery',
                'haz_device', 'bulb', 'thermometer',
                'haz_medicine', 'medicine', 'lighter'
            ].includes(n)) {
                return 'hazardous';
            }
            // 3. Kitchen Waste
            // Includes organic_food and all related food waste items
            if ([
                'organic_food', 'bone', 'cake', 'egg_shell', 'flower', 'fruit_peel', 'leftover', 'tea', 'vegetable'
            ].includes(n)) {
                return 'kitchen';
            }
            // 4. Other Waste
            // Includes hygiene_contaminated, small_hard_items, foam, brick_ceramic
            if ([
                'hygiene_contaminated', 'cigarette', 'mask', 'diaper', 'tissue', 'wet_wipe',
                'small_hard_items', 'chopsticks', 'pen', 'toothpick', 'comb',
                'foam',
                'brick_ceramic', 'brick', 'ceramic'
            ].includes(n)) {
                return 'other';
            }
            return 'other'; // Default fallback
        }
        function renderSupportedClasses(coarseLabels, fineLabels) {
            const container = document.getElementById('supportedClasses');
            container.innerHTML = '';
            const order = Array.isArray(coarseLabels) && coarseLabels.length
                ? coarseLabels
                : ['recyclable', 'hazardous', 'kitchen', 'other'];
            const grouped = {};
            order.forEach((k) => grouped[k] = []);
            normalizeClassList(fineLabels || []).forEach((fine) => {
                const coarse = inferCoarseForFineLabel(fine);
                if (!grouped[coarse]) grouped[coarse] = [];
                grouped[coarse].push(fine);
            });
            order.forEach((coarseKey) => {
                const card = document.createElement('div');
                card.className = 'scope-card';
                const head = document.createElement('div');
                head.className = 'scope-head';
                head.textContent = `${coarseLabelToEN(coarseKey)} (${coarseKey})`;
                card.appendChild(head);
                const sub = document.createElement('div');
                sub.className = 'scope-sub';
                (grouped[coarseKey] || []).forEach((fine) => {
                    const chip = document.createElement('span');
                    chip.className = 'chip';
                    chip.textContent = fine;
                    sub.appendChild(chip);
                });
                if (!(grouped[coarseKey] || []).length) {
                    const empty = document.createElement('span');
                    empty.className = 'chip';
                    empty.textContent = '-';
                    sub.appendChild(empty);
                }
                card.appendChild(sub);
                container.appendChild(card);
            });
        }
        async function init() {
            try {
                const res = await fetch(`${API_BASE}/health`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const contentType = res.headers.get('content-type') || '';
                if (!contentType.includes('application/json')) {
                    throw new Error('API endpoint returned non-JSON response.');
                }
                const data = await res.json();
                
                if (data.status === 'healthy') {
                    if (data.model_loaded) {
                        setStatus('Server connected. Local model is loaded.', false);
                    } else {
                        setStatus('Server connected, but local model is NOT loaded (Check backend).', true);
                    }
                    
                    try {
                        const infoRes = await fetch(`${API_BASE}/model_info`);
                        const info = await infoRes.json();
                        class_names = normalizeClassList(info.class_names || class_names);
                        renderSupportedClasses(info.coarse_labels || [], class_names);
                    } catch(e) {
                        console.warn("Failed to fetch model_info:", e);
                    }
                } else {
                    setStatus('Server reported unhealthy status.', true);
                }
            } catch (e) {
                setStatus('Server connection failed: ' + e.message, true);
            }
        }
        function setStatus(msg, isError = false) {
            const el = document.getElementById('status');
            el.textContent = msg;
            el.className = isError ? 'status error' : 'status';
        }
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewImg = document.getElementById('preview');
        const bboxCanvas = document.getElementById('bboxCanvas');
        const manualPanel = document.getElementById('manualPanel');
        const manualConfirmPanel = document.getElementById('manualConfirmPanel');
        const manualRunBtn = document.getElementById('manualRunBtn');
        bboxCanvas.addEventListener('mousedown', onCanvasMouseDown);
        bboxCanvas.addEventListener('mousemove', onCanvasMouseMove);
        bboxCanvas.addEventListener('mouseup', onCanvasMouseUp);
        bboxCanvas.addEventListener('mouseleave', onCanvasMouseUp);
        bboxCanvas.addEventListener('touchstart', onCanvasTouchStart, { passive: false });
        bboxCanvas.addEventListener('touchmove', onCanvasTouchMove, { passive: false });
        bboxCanvas.addEventListener('touchend', onCanvasTouchEnd, { passive: false });
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => { uploadArea.classList.remove('dragover'); });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        });
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) handleFile(file);
        });
        function handleFile(file) {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImg.style.display = 'block';
                    previewImg.src = e.target.result;
                    currentImageDataUrl = e.target.result;
                    manualPanel.style.display = 'flex';
                    stopManualMode();
                    clearManualBoxes();
                    clearDetectedBoxes();
                    document.getElementById('previewContainer').style.display = 'block';
                    document.getElementById('uploadArea').style.display = 'none';
                    predict(e.target.result, false);
                };
                reader.readAsDataURL(file);
                return;
            }
            setStatus('Image upload only. Please use JPG/PNG/WEBP.', true);
        }
        async function predict(dataUrl, forceVlm = false) {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('loadingText').textContent = forceVlm ? 'Calling Cloud Vision AI for Review...' : 'Running local inference...';
            document.getElementById('resultContainer').style.display = 'none';
            try {
                const endpoint = forceVlm ? `${API_BASE}/predict_vlm` : `${API_BASE}/predict`;
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl, format: 'base64', force_vlm: forceVlm })
                });
                const data = await res.json();
                
                lastPredictionData = data; 
                if (data.error) {
                    setStatus('Inference failed: ' + data.error, true);
                } else {
                    clearDetectedBoxes();
                    displayResult(data, forceVlm);
                    
                    if (data.vlm_fallback_used && data.confidence === 0) {
                        setStatus('AI Review request failed. See note below.', true);
                    } else {
                        setStatus(forceVlm ? 'AI Review Completed. Result updated.' : 'Prediction completed. If incorrect, request AI Review.', false);
                    }
                    
                    const modeNotice = document.getElementById('modeNotice');
                    if (modeNotice) {
                        modeNotice.style.background = '#ecfdf3';
                        modeNotice.style.borderColor = '#bbf7d0';
                        modeNotice.style.color = '#166534';
                        modeNotice.textContent = 'Prediction completed. For multiple items, switch to Manual Box Mode.';
                    }
                }
            } catch (e) {
                setStatus('Request failed: ' + e.message, true);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function requestAIReview() {
            if (!currentImageDataUrl) return;
            predict(currentImageDataUrl, true);
        }

        async function reportEdgeCase() {
            if (!currentImageDataUrl) return;
            const expectedLabel = prompt("The system made a mistake? Please tell us what this item actually is (e.g., used battery, plastic bag):");
            if (!expectedLabel) return;

            try {
                const res = await fetch(`${API_BASE}/report_edge_case`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        image: currentImageDataUrl, 
                        expected_label: expectedLabel,
                        model_prediction: lastPredictionData ? (lastPredictionData.daily_label || lastPredictionData.class_name) : 'unknown'
                    })
                });
                if (res.ok) {
                    alert("✅ Edge Case logged successfully! Thank you for contributing to our research.");
                    setStatus('Edge case logged. Your contribution helps improve the model.', false);
                } else {
                    alert("❌ Failed to submit. Please try again.");
                }
            } catch (e) {
                alert("Network error: " + e.message);
            }
        }

        function displayResult(data, isVlmReview = false) {
            document.getElementById('resultContainer').style.display = 'block';
            const coarse = coarseLabelEN[data.coarse_category] || data.coarse_category || 'Review Required';
            
            const isVlmError = data.vlm_fallback_used && data.uncertain && data.confidence === 0;
            const usedVlm = (Boolean(data.vlm_fallback_used) || isVlmReview) && !isVlmError;
            
            const resultCard = document.getElementById('resultCardBg');
            
            if (isVlmError) {
                resultCard.style.background = 'linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%)';
                document.getElementById('resultClass').textContent = `❌ AI Request Failed`;
            } else if (usedVlm) {
                resultCard.style.background = 'linear-gradient(135deg, #6d28d9 0%, #4c1d95 100%)';
                document.getElementById('resultClass').textContent = `✨ ${coarse} (Cloud AI Verified)`;
            } else {
                resultCard.style.background = 'linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)';
                document.getElementById('resultClass').textContent = data.uncertain ? `⚠️ ${coarse} (Review Recommended)` : coarse;
            }

            document.getElementById('resultConfidence').textContent = `Confidence: ${((data.coarse_confidence || data.confidence) * 100).toFixed(1)}%`;
            document.getElementById('resultSub').textContent = usedVlm
                ? `Cloud AI label: ${data.daily_label || data.class_name}`
                : `Top fine label: ${data.daily_label || data.class_name}`;

            const probsDiv = document.getElementById('probabilities');
            probsDiv.innerHTML = '';
            
            const topPreds = Array.isArray(data.top_predictions) ? data.top_predictions.slice(0, 5) : [];

            topPreds.forEach((pred) => {
                const label = pred.daily_label || pred.class_name;
                const prob = pred.confidence;
                
                const item = document.createElement('div');
                item.className = 'prob-item';
                item.innerHTML = `
                    <div class="prob-label">${label}</div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: ${prob * 100}%"></div>
                    </div>
                    <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                `;
                probsDiv.appendChild(item);
            });

            if (Array.isArray(data.detected_objects) && data.detected_objects.length > 0) {
                const objTitle = document.createElement('div');
                objTitle.style.marginTop = '10px';
                objTitle.style.fontSize = '13px';
                objTitle.style.fontWeight = '600';
                objTitle.style.color = '#334155';
                objTitle.textContent = `${data.detected_objects.length} candidate object(s) detected via manual boxes:`;
                probsDiv.appendChild(objTitle);

                data.detected_objects.forEach((obj, idx) => {
                    const pred = obj.prediction || {};
                    const line = document.createElement('div');
                    line.style.marginTop = '6px';
                    line.style.fontSize = '13px';
                    line.style.color = '#475569';
                    line.textContent = `- Box ${idx + 1}: ${pred.daily_label || pred.class_name || 'unknown'} (${((pred.confidence || 0) * 100).toFixed(1)}%)`;
                    probsDiv.appendChild(line);
                });

                drawDetectedBoxes(data.detected_objects);
            } else {
                clearDetectedBoxes();
            }

            if (data.advice) {
                const advice = document.createElement('div');
                advice.style.marginTop = '8px';
                advice.style.fontSize = '13px';
                advice.style.color = isVlmError ? '#b91c1c' : '#9a4c16';
                advice.style.fontWeight = isVlmError ? 'bold' : 'normal';
                advice.textContent = `Note: ${data.advice}`;
                probsDiv.appendChild(advice);
            }

            document.getElementById('resetBtn').style.display = 'block';
        }

        function reset() {
            document.getElementById('previewContainer').style.display = 'none';
            document.getElementById('resultContainer').style.display = 'none';
            document.getElementById('resetBtn').style.display = 'none';
            document.getElementById('uploadArea').style.display = 'block';
            fileInput.value = '';
            previewImg.src = '';
            currentImageDataUrl = null;
            lastPredictionData = null;
            manualPanel.style.display = 'none';
            stopManualMode();
            clearManualBoxes();
            clearDetectedBoxes();
        }

        function clearDetectedBoxes() {
            const ctx = bboxCanvas.getContext('2d');
            ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
            bboxCanvas.style.display = 'none';
        }

        function startManualMode() {
            if (!currentImageDataUrl) {
                setStatus('Please upload an image before drawing boxes.', true);
                return;
            }
            manualMode = true;
            bboxCanvas.style.pointerEvents = 'auto';
            bboxCanvas.style.display = 'block';
            const btn = document.getElementById('manualModeBtn');
            if (btn) btn.textContent = 'Drawing...';
            setStatus('Manual box mode enabled: drag to draw one or more boxes.', false);
        }

        function stopManualMode() {
            manualMode = false;
            isDrawing = false;
            drawStart = null;
            drawCurrent = null;
            pendingBox = null;
            bboxCanvas.style.pointerEvents = 'none';
            const btn = document.getElementById('manualModeBtn');
            if (btn) btn.textContent = 'Manual Box Mode';
            manualConfirmPanel.style.display = 'none';
        }

        function clearManualBoxes() {
            manualBoxes = [];
            pendingBox = null;
            manualRunBtn.disabled = true;
            manualConfirmPanel.style.display = 'none';
            if (manualMode) drawManualBoxes();
        }

        function confirmPendingBox() {
            if (!pendingBox) return;
            manualBoxes.push({
                bbox_norm: pendingBox.bbox_norm,
                label: `Box ${manualBoxes.length + 1}`,
            });
            pendingBox = null;
            manualRunBtn.disabled = manualBoxes.length === 0;
            manualConfirmPanel.style.display = 'none';
            drawManualBoxes();
            setStatus('Box confirmed. You can keep drawing or run box predictions.', false);
        }

        function cancelPendingBox() {
            pendingBox = null;
            manualConfirmPanel.style.display = 'none';
            drawManualBoxes();
            setStatus('Pending box canceled. You can draw again.', false);
        }

        function onCanvasMouseDown(e) {
            if (!manualMode) return;
            const p = getCanvasPoint(e);
            isDrawing = true;
            drawStart = p;
            drawCurrent = p;
        }

        function onCanvasMouseMove(e) {
            if (!manualMode || !isDrawing) return;
            drawCurrent = getCanvasPoint(e);
            drawManualBoxes();
        }

        function onCanvasMouseUp(e) {
            if (!manualMode || !isDrawing) return;
            const p = getCanvasPoint(e);
            finalizeManualBox(p);
        }

        function onCanvasTouchStart(e) {
            if (!manualMode) return;
            e.preventDefault();
            const touch = e.touches[0];
            if (!touch) return;
            const p = getCanvasPoint(touch);
            isDrawing = true;
            drawStart = p;
            drawCurrent = p;
        }

        function onCanvasTouchMove(e) {
            if (!manualMode || !isDrawing) return;
            e.preventDefault();
            const touch = e.touches[0];
            if (!touch) return;
            drawCurrent = getCanvasPoint(touch);
            drawManualBoxes();
        }

        function onCanvasTouchEnd(e) {
            if (!manualMode || !isDrawing) return;
            e.preventDefault();
            const touch = e.changedTouches[0];
            const p = touch ? getCanvasPoint(touch) : drawCurrent || drawStart;
            finalizeManualBox(p);
        }

        function finalizeManualBox(p) {
            if (!p || !drawStart) return;
            isDrawing = false;
            const x1 = Math.min(drawStart.x, p.x);
            const y1 = Math.min(drawStart.y, p.y);
            const x2 = Math.max(drawStart.x, p.x);
            const y2 = Math.max(drawStart.y, p.y);
            const w = bboxCanvas.width || 1;
            const h = bboxCanvas.height || 1;

            if ((x2 - x1) >= 20 && (y2 - y1) >= 20) {
                pendingBox = {
                    bbox_norm: [x1 / w, y1 / h, x2 / w, y2 / h],
                };
                manualConfirmPanel.style.display = 'flex';
                setStatus('Please confirm whether to add this box.', false);
            }
            drawCurrent = null;
            drawManualBoxes();
        }

        function getCanvasPoint(e) {
            const rect = bboxCanvas.getBoundingClientRect();
            const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
            const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
            return { x, y };
        }

        function drawManualBoxes() {
            if (!syncCanvasToPreview()) {
                setTimeout(drawManualBoxes, 80);
                return;
            }
            const ctx = bboxCanvas.getContext('2d');
            ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
            bboxCanvas.style.display = 'block';
            ctx.lineWidth = 2;
            ctx.font = '12px sans-serif';

            manualBoxes.forEach((obj, idx) => {
                const bn = obj.bbox_norm;
                const x1 = bn[0] * bboxCanvas.width;
                const y1 = bn[1] * bboxCanvas.height;
                const x2 = bn[2] * bboxCanvas.width;
                const y2 = bn[3] * bboxCanvas.height;
                const color = ['#ff4d4f', '#1890ff', '#52c41a', '#fa8c16', '#722ed1'][idx % 5];
                ctx.strokeStyle = color;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                const text = obj.label || `Box ${idx + 1}`;
                const textW = ctx.measureText(text).width;
                const tx = Math.max(0, x1);
                const ty = Math.max(14, y1);
                ctx.fillStyle = color;
                ctx.fillRect(tx, ty - 14, textW + 8, 14);
                ctx.fillStyle = '#fff';
                ctx.fillText(text, tx + 4, ty - 3);
            });

            if (pendingBox) {
                const bn = pendingBox.bbox_norm;
                const x1 = bn[0] * bboxCanvas.width;
                const y1 = bn[1] * bboxCanvas.height;
                const x2 = bn[2] * bboxCanvas.width;
                const y2 = bn[3] * bboxCanvas.height;
                ctx.strokeStyle = '#fa8c16';
                ctx.setLineDash([6, 3]);
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.setLineDash([]);
                const text = 'Pending';
                const textW = ctx.measureText(text).width;
                const tx = Math.max(0, x1);
                const ty = Math.max(14, y1);
                ctx.fillStyle = '#fa8c16';
                ctx.fillRect(tx, ty - 14, textW + 8, 14);
                ctx.fillStyle = '#fff';
                ctx.fillText(text, tx + 4, ty - 3);
            }

            if (isDrawing && drawStart && drawCurrent) {
                const x1 = Math.min(drawStart.x, drawCurrent.x);
                const y1 = Math.min(drawStart.y, drawCurrent.y);
                const x2 = Math.max(drawStart.x, drawCurrent.x);
                const y2 = Math.max(drawStart.y, drawCurrent.y);
                ctx.strokeStyle = '#00b96b';
                ctx.setLineDash([6, 3]);
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.setLineDash([]);
            }
        }

        async function runManualClassification() {
            if (!currentImageDataUrl || manualBoxes.length === 0) {
                setStatus('Please draw at least one box before running prediction.', true);
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultContainer').style.display = 'none';
            try {
                const image = await loadImageFromDataUrl(currentImageDataUrl);
                const crops = manualBoxes.map((box) => cropDataUrlByNormBox(image, box.bbox_norm));
                const preds = [];
                for (const crop of crops) {
                    const res = await fetch(`${API_BASE}/predict`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: crop, format: 'base64' })
                    });
                    const data = await res.json();
                    preds.push(data.error ? { error: data.error } : data);
                }

                const detected = manualBoxes.map((box, idx) => ({
                    ...box,
                    prediction: preds[idx],
                })).filter(x => x.prediction && !x.prediction.error);

                if (!detected.length) {
                    setStatus('Manual box prediction failed. Please try again.', true);
                    return;
                }

                const merged = buildMergedFromObjects(detected);
                displayResult(merged);
                setStatus(`Manual box prediction completed: ${detected.length} object(s).`, false);
                stopManualMode();
            } catch (e) {
                setStatus('Manual prediction failed: ' + e.message, true);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function buildMergedFromObjects(detected) {
            const countDaily = {};
            const countCoarse = {};
            const aggTopPredsMap = {}; 
            let confSum = 0;

            detected.forEach((obj) => {
                const p = obj.prediction || {};
                const daily = p.daily_label || p.class_name || 'unknown';
                const coarse = p.coarse_category || 'other';
                countDaily[daily] = (countDaily[daily] || 0) + 1;
                countCoarse[coarse] = (countCoarse[coarse] || 0) + 1;
                confSum += (p.confidence || 0);
                
                const topPreds = p.top_predictions || [];
                topPreds.forEach(pred => {
                    const label = pred.daily_label || pred.class_name;
                    if(!aggTopPredsMap[label]) {
                        aggTopPredsMap[label] = { daily_label: label, class_name: label, confidence: 0 };
                    }
                    aggTopPredsMap[label].confidence += pred.confidence;
                });
            });

            const n = detected.length;
            
            const aggregatedTopPreds = Object.values(aggTopPredsMap).map(pred => {
                pred.confidence = pred.confidence / n;
                return pred;
            }).sort((a, b) => b.confidence - a.confidence);

            const dailyTop = Object.entries(countDaily).sort((a, b) => b[1] - a[1])[0]?.[0] || 'unknown';
            const coarseTop = Object.entries(countCoarse).sort((a, b) => b[1] - a[1])[0]?.[0] || 'other';

            return {
                is_multi_object: true,
                daily_label: dailyTop,
                coarse_category: coarseTop,
                confidence: confSum / n,
                uncertain: false,
                top_predictions: aggregatedTopPreds,
                detected_objects: detected,
                object_count: detected.length,
            };
        }

        function loadImageFromDataUrl(dataUrl) {
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => reject(new Error('Failed to load image'));
                img.src = dataUrl;
            });
        }

        function cropDataUrlByNormBox(image, bboxNorm) {
            const [x1n, y1n, x2n, y2n] = bboxNorm;
            const sx = Math.max(0, Math.floor(x1n * image.naturalWidth));
            const sy = Math.max(0, Math.floor(y1n * image.naturalHeight));
            const ex = Math.min(image.naturalWidth, Math.ceil(x2n * image.naturalWidth));
            const ey = Math.min(image.naturalHeight, Math.ceil(y2n * image.naturalHeight));
            const sw = Math.max(1, ex - sx);
            const sh = Math.max(1, ey - sy);

            const canvas = document.createElement('canvas');
            canvas.width = sw;
            canvas.height = sh;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
            return canvas.toDataURL('image/jpeg', 0.9);
        }

        function syncCanvasToPreview() {
            const imgRect = previewImg.getBoundingClientRect();
            if (!imgRect.width || !imgRect.height) return false;
            bboxCanvas.width = Math.floor(imgRect.width);
            bboxCanvas.height = Math.floor(imgRect.height);
            bboxCanvas.style.width = `${imgRect.width}px`;
            bboxCanvas.style.height = `${imgRect.height}px`;
            bboxCanvas.style.left = `${previewImg.offsetLeft}px`;
            bboxCanvas.style.top = `${previewImg.offsetTop}px`;
            return true;
        }

        function drawDetectedBoxes(objects) {
            if (!previewImg || previewImg.style.display === 'none') {
                clearDetectedBoxes();
                return;
            }

            if (!syncCanvasToPreview()) {
                setTimeout(() => drawDetectedBoxes(objects), 80);
                return;
            }
            bboxCanvas.style.display = 'block';

            const ctx = bboxCanvas.getContext('2d');
            ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
            ctx.lineWidth = 2;
            ctx.font = '12px sans-serif';

            objects.forEach((obj, index) => {
                const bn = obj.bbox_norm || [];
                if (bn.length !== 4) return;
                const x1 = bn[0] * bboxCanvas.width;
                const y1 = bn[1] * bboxCanvas.height;
                const x2 = bn[2] * bboxCanvas.width;
                const y2 = bn[3] * bboxCanvas.height;

                const color = ['#ff4d4f', '#1890ff', '#52c41a', '#fa8c16', '#722ed1'][index % 5];
                ctx.strokeStyle = color;
                ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));

                const pred = obj.prediction || {};
                const label = obj.label || pred.daily_label || pred.class_name || `Object ${index + 1}`;
                const conf = ((pred.confidence || obj.confidence || 0) * 100).toFixed(0);
                const text = `${label} ${conf}%`;

                const textW = ctx.measureText(text).width;
                const boxX = Math.max(0, x1);
                const boxY = Math.max(14, y1);
                ctx.fillStyle = color;
                ctx.fillRect(boxX, boxY - 14, textW + 8, 14);
                ctx.fillStyle = '#fff';
                ctx.fillText(text, boxX + 4, boxY - 3);
            });
        }

        init();
    </script>
</body>
</html>
