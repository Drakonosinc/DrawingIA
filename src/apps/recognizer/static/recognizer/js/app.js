/**
 * DrawingIA Studio — Frontend Controller
 * Handles smooth HTML5 canvas sketching, undo history, live neural inference,
 * 28x28 Explainable AI (XAI) tensor & heatmap rendering, Active Learning, and QuickDraw Challenge.
 */

(() => {
    const canvas = document.getElementById("drawing-canvas");
    const ctx = canvas.getContext("2d");

    const normCanvas = document.getElementById("canvas-normalized");
    const normCtx = normCanvas.getContext("2d");
    const heatCanvas = document.getElementById("canvas-heatmap");
    const heatCtx = heatCanvas.getContext("2d");

    const brushSlider = document.getElementById("brush-size");
    const brushVal = document.getElementById("brush-size-val");
    const chkAutoPredict = document.getElementById("chk-autopredict");
    const selectEngine = document.getElementById("select-engine");

    const predIcon = document.getElementById("pred-icon");
    const predLabel = document.getElementById("pred-label");
    const predConfidence = document.getElementById("pred-confidence");
    const predEngine = document.getElementById("pred-engine");
    const predEntropy = document.getElementById("pred-entropy");
    const probBars = document.getElementById("prob-bars");

    const statAccuracy = document.getElementById("stat-accuracy");
    const statLatency = document.getElementById("stat-latency");
    const statPredictions = document.getElementById("stat-predictions");
    const statTaught = document.getElementById("stat-taught");
    const teachFeedback = document.getElementById("teach-feedback");

    // QuickDraw Challenge elements
    const tabStudio = document.getElementById("tab-studio");
    const tabChallenge = document.getElementById("tab-challenge");
    const challengeBanner = document.getElementById("challenge-banner");
    const challengePrompt = document.getElementById("challenge-prompt");
    const challengeTimer = document.getElementById("challenge-timer");
    const challengeScore = document.getElementById("challenge-score");
    const btnSkipChallenge = document.getElementById("btn-skip-challenge");

    let isDrawing = false;
    let strokes = [];
    let currentStroke = [];
    let debounceTimer = null;

    // Challenge state
    let challengeActive = false;
    let currentTargetClass = null;
    let score = 0;
    let timeLeft = 10.0;
    let timerInterval = null;

    function initCanvas() {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        clearXaiCanvases();
    }

    function clearXaiCanvases() {
        normCtx.fillStyle = "#050811";
        normCtx.fillRect(0, 0, normCanvas.width, normCanvas.height);
        heatCtx.fillStyle = "#050811";
        heatCtx.fillRect(0, 0, heatCanvas.width, heatCanvas.height);
    }

    function getCanvasCoordinates(event) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const clientX = event.touches ? event.touches[0].clientX : event.clientX;
        const clientY = event.touches ? event.touches[0].clientY : event.clientY;
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY,
        };
    }

    function redrawAllStrokes() {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        for (const stroke of strokes) {
            if (stroke.points.length === 0) continue;
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = stroke.size;
            ctx.beginPath();
            ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
            for (let i = 1; i < stroke.points.length; i++) {
                const prev = stroke.points[i - 1];
                const curr = stroke.points[i];
                const midX = (prev.x + curr.x) / 2;
                const midY = (prev.y + curr.y) / 2;
                ctx.quadraticCurveTo(prev.x, prev.y, midX, midY);
            }
            ctx.stroke();
        }
    }

    function startDrawing(event) {
        event.preventDefault();
        isDrawing = true;
        const pt = getCanvasCoordinates(event);
        currentStroke = {
            size: parseInt(brushSlider.value, 10),
            points: [pt],
        };
        strokes.push(currentStroke);
        redrawAllStrokes();
    }

    function moveDrawing(event) {
        if (!isDrawing) return;
        event.preventDefault();
        const pt = getCanvasCoordinates(event);
        currentStroke.points.push(pt);
        redrawAllStrokes();
    }

    function stopDrawing(event) {
        if (!isDrawing) return;
        event.preventDefault();
        isDrawing = false;
        if (chkAutoPredict.checked) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(runPrediction, 160);
        }
    }

    // Render 28x28 matrix onto a 140x140 XAI canvas
    function renderGrid28x28(targetCtx, grid, mode = "grayscale") {
        const cell = 140 / 28;
        targetCtx.clearRect(0, 0, 140, 140);
        for (let r = 0; r < 28; r++) {
            for (let c = 0; c < 28; c++) {
                const val = Math.max(0, Math.min(1, grid[r][c] || 0));
                if (mode === "grayscale") {
                    const v = Math.round(val * 255);
                    targetCtx.fillStyle = `rgb(${v}, ${v}, ${v})`;
                } else {
                    // Inferno-like thermal colormap for CNN activation saliency
                    const rCol = Math.min(255, Math.round(val * 320));
                    const gCol = Math.min(255, Math.max(0, Math.round((val - 0.25) * 280)));
                    const bCol = Math.min(255, Math.round(Math.sin(val * Math.PI) * 180));
                    targetCtx.fillStyle = `rgb(${rCol}, ${gCol}, ${bCol})`;
                }
                targetCtx.fillRect(c * cell, r * cell, Math.ceil(cell), Math.ceil(cell));
            }
        }
    }

    function updatePredictionUI(pred) {
        predIcon.textContent = pred.predicted_icon || "✏";
        predLabel.textContent = pred.predicted_label || "Draw on Canvas";
        const confPct = ((pred.confidence || 0) * 100).toFixed(1);
        predConfidence.textContent = `${confPct}%`;
        predEngine.textContent = pred.engine_used.toUpperCase();
        predEntropy.textContent = (pred.entropy || 0).toFixed(2);
        statLatency.textContent = `${pred.latency_ms} ms`;

        if (pred.predicted_id !== "none") {
            statPredictions.textContent = parseInt(statPredictions.textContent || "0", 10) + 1;
        }

        if (pred.normalized_grid_28x28 && pred.activation_heatmap_28x28) {
            renderGrid28x28(normCtx, pred.normalized_grid_28x28, "grayscale");
            renderGrid28x28(heatCtx, pred.activation_heatmap_28x28, "heatmap");
        }

        if (!pred.top_predictions || pred.top_predictions.length === 0) {
            probBars.innerHTML = `<div class="empty-state">Draw a stroke or click "✨ Auto-Demo" to inspect class probabilities.</div>`;
            return;
        }

        probBars.innerHTML = pred.top_predictions
            .map((item) => {
                const pct = (item.probability * 100).toFixed(1);
                return `
                <div class="prob-row">
                    <div class="prob-name"><span>${item.icon}</span> <span>${item.label}</span></div>
                    <div class="prob-track"><div class="prob-fill" style="width: ${pct}%"></div></div>
                    <div class="prob-pct">${pct}%</div>
                </div>`;
            })
            .join("");

        // Check QuickDraw challenge win condition
        if (challengeActive && currentTargetClass && pred.predicted_id === currentTargetClass.id && pred.confidence >= 0.35) {
            score += 1;
            challengeScore.textContent = String(score);
            teachFeedback.textContent = `🎉 Correct! AI recognized ${currentTargetClass.icon} ${currentTargetClass.label} (${confPct}%)!`;
            nextChallengePrompt();
        }
    }

    async function runPrediction() {
        const dataUrl = canvas.toDataURL("image/png");
        const engine = selectEngine.value;
        try {
            const response = await fetch("/api/predict/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataUrl, engine }),
            });
            const data = await response.json();
            if (data.status === "ok") {
                updatePredictionUI(data.prediction);
            }
        } catch (err) {
            console.error("Prediction request failed:", err);
        }
    }

    async function teachCurrentCanvas(classId) {
        const dataUrl = canvas.toDataURL("image/png");
        teachFeedback.textContent = `⏳ Fine-tuning neural weights for '${classId}'...`;
        try {
            const response = await fetch("/api/teach/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataUrl, class_id: classId }),
            });
            const data = await response.json();
            if (data.status === "ok") {
                const res = data.teach_result;
                statTaught.textContent = String(res.total_taught_samples);
                teachFeedback.textContent = `✓ Taught '${res.taught_class}'! Post-adaptation confidence: ${res.new_confidence}%`;
                updatePredictionUI(data.prediction);
            } else {
                teachFeedback.textContent = `⚠ ${data.message}`;
            }
        } catch (err) {
            teachFeedback.textContent = `⚠ Failed to teach sample: ${err}`;
        }
    }

    async function loadAutoSample() {
        const targetId = challengeActive && currentTargetClass ? currentTargetClass.id : "";
        const url = targetId ? `/api/sample/?class_id=${encodeURIComponent(targetId)}` : "/api/sample/";
        try {
            const response = await fetch(url);
            const data = await response.json();
            if (data.status === "ok") {
                // Draw the 28x28 synthetic sample scaled onto the main 420x420 canvas
                strokes = [];
                ctx.fillStyle = "#000000";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                const grid = data.grid_28x28;
                const scale = canvas.width / 28;
                for (let r = 0; r < 28; r++) {
                    for (let c = 0; c < 28; c++) {
                        const v = grid[r][c];
                        if (v > 0.05) {
                            const gray = Math.round(v * 255);
                            ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
                            ctx.beginPath();
                            ctx.arc((c + 0.5) * scale, (r + 0.5) * scale, (scale * 0.58) * v, 0, Math.PI * 2);
                            ctx.fill();
                        }
                    }
                }
                updatePredictionUI(data.prediction);
            }
        } catch (err) {
            console.error("Failed to load sample:", err);
        }
    }

    async function recalibrateModel() {
        const btn = document.getElementById("btn-retrain");
        btn.disabled = true;
        btn.textContent = "⏳ Calibrating...";
        try {
            const response = await fetch("/api/retrain/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ samples_per_class: 90, epochs: 6 }),
            });
            const data = await response.json();
            if (data.status === "ok") {
                statAccuracy.textContent = `${data.training_report.validation_accuracy}%`;
                teachFeedback.textContent = `✓ Calibrated on ${data.training_report.samples_seen} samples (${data.training_report.validation_accuracy}% val accuracy)`;
            }
        } finally {
            btn.disabled = false;
            btn.textContent = "↻ Calibrate AI";
        }
    }

    // QuickDraw Challenge Logic
    function nextChallengePrompt() {
        const classes = window.DRAWING_CLASSES || [];
        if (classes.length === 0) return;
        currentTargetClass = classes[Math.floor(Math.random() * classes.length)];
        challengePrompt.textContent = `${currentTargetClass.icon} ${currentTargetClass.label}`;
        timeLeft = 10.0;
        strokes = [];
        redrawAllStrokes();
    }

    function setChallengeMode(enabled) {
        challengeActive = enabled;
        tabStudio.classList.toggle("active", !enabled);
        tabChallenge.classList.toggle("active", enabled);
        challengeBanner.classList.toggle("hidden", !enabled);
        clearInterval(timerInterval);
        if (enabled) {
            score = 0;
            challengeScore.textContent = "0";
            nextChallengePrompt();
            timerInterval = setInterval(() => {
                timeLeft = Math.max(0, timeLeft - 0.1);
                challengeTimer.textContent = `${timeLeft.toFixed(1)}s`;
                if (timeLeft <= 0) {
                    nextChallengePrompt();
                }
            }, 100);
        }
    }

    // Event bindings
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", moveDrawing);
    window.addEventListener("mouseup", stopDrawing);

    canvas.addEventListener("touchstart", startDrawing, { passive: false });
    canvas.addEventListener("touchmove", moveDrawing, { passive: false });
    window.addEventListener("touchend", stopDrawing, { passive: false });

    brushSlider.addEventListener("input", () => {
        brushVal.textContent = brushSlider.value;
    });

    document.getElementById("btn-undo").addEventListener("click", () => {
        strokes.pop();
        redrawAllStrokes();
        if (chkAutoPredict.checked) runPrediction();
    });

    document.getElementById("btn-clear").addEventListener("click", () => {
        strokes = [];
        redrawAllStrokes();
        clearXaiCanvases();
        runPrediction();
    });

    document.getElementById("btn-sample").addEventListener("click", loadAutoSample);
    document.getElementById("btn-predict").addEventListener("click", runPrediction);
    document.getElementById("btn-retrain").addEventListener("click", recalibrateModel);
    selectEngine.addEventListener("change", runPrediction);

    document.querySelectorAll(".class-pill").forEach((btn) => {
        btn.addEventListener("click", () => {
            const cid = btn.getAttribute("data-class-id");
            if (cid) teachCurrentCanvas(cid);
        });
    });

    tabStudio.addEventListener("click", () => setChallengeMode(false));
    tabChallenge.addEventListener("click", () => setChallengeMode(true));
    btnSkipChallenge.addEventListener("click", nextChallengePrompt);

    initCanvas();
})();
