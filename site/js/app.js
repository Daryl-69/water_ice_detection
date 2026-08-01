/* ═══════════════════════════════════════════════════════════════════════════
   Lunar Ice Prospector — App Logic
   Canvas-based map viewer with layer compositing, pan/zoom, pixel inspect
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── Config ───────────────────────────────────────────────────────────────
    const LAYERS_PATH = 'assets/layers/';
    const LAYER_FILES = {
        cpr: 'cpr_base.png',
        vol: 'vol_base.png',
        psr: 'psr_overlay.png',
        ice_phase1: 'ice_phase1.png',
        ice_ml: 'ice_ml.png',
        confidence: 'confidence.png',
        slope: 'slope_safety.png',
        dem: 'dem_terrain.png'
    };

    const FEATURE_COLORS = {
        CPR: '#f0f921',
        TRT: '#c4e020',
        VOL: '#5ec962',
        HLX: '#21918c',
        EVN: '#3b528b',
        SRD: '#472d7b',
        ODD: '#440154'
    };

    // ── State ────────────────────────────────────────────────────────────────
    const state = {
        images: {},
        loaded: 0,
        totalLayers: Object.keys(LAYER_FILES).length,
        // View transform
        offsetX: 0, offsetY: 0,
        zoom: 1,
        minZoom: 0.15,
        maxZoom: 5,
        // Interaction
        dragging: false,
        dragStartX: 0, dragStartY: 0,
        lastOffsetX: 0, lastOffsetY: 0,
        // Layers
        baseLayer: 'cpr',
        overlays: {
            psr: { visible: true, opacity: 0.6 },
            ice_phase1: { visible: true, opacity: 0.85 },
            ice_ml: { visible: false, opacity: 0.8 },
            confidence: { visible: false, opacity: 0.7 },
            slope: { visible: false, opacity: 0.7 },
            dem: { visible: false, opacity: 0.75 }
        },
        // Split view
        splitMode: false,
        splitX: 0.5,
        // Stats
        stats: null,
        imgW: 0, imgH: 0
    };

    // ── DOM refs ──────────────────────────────────────────────────────────────
    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');
    const viewport = document.getElementById('mapViewport');
    const crosshair = document.getElementById('crosshair');
    const coordText = document.getElementById('coordText');
    const pixelPopup = document.getElementById('pixelPopup');
    const popupBody = document.getElementById('popupBody');

    // ═══════════════════════════════════════════════════════════════════════════
    // IMAGE LOADING
    // ═══════════════════════════════════════════════════════════════════════════
    function loadAllLayers() {
        for (const [key, file] of Object.entries(LAYER_FILES)) {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                state.images[key] = img;
                state.loaded++;
                if (state.imgW === 0) {
                    state.imgW = img.naturalWidth;
                    state.imgH = img.naturalHeight;
                }
                if (state.loaded === state.totalLayers) {
                    onAllLoaded();
                }
            };
            img.onerror = () => {
                // silently skip failed layers
                state.loaded++;
                if (state.loaded === state.totalLayers) onAllLoaded();
            };
            img.src = LAYERS_PATH + file;
        }
    }

    function onAllLoaded() {
        // layers ready
        fitToView();
        render();
        loadStats();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RENDERING
    // ═══════════════════════════════════════════════════════════════════════════
    function resizeCanvas() {
        const rect = viewport.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    }

    function fitToView() {
        const rect = viewport.getBoundingClientRect();
        if (state.imgW === 0) return;
        const scaleX = rect.width / state.imgW;
        const scaleY = rect.height / state.imgH;
        state.zoom = Math.min(scaleX, scaleY) * 0.95;
        state.offsetX = (rect.width - state.imgW * state.zoom) / 2;
        state.offsetY = (rect.height - state.imgH * state.zoom) / 2;
    }

    function render() {
        const rect = viewport.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = '#06060f';
        ctx.fillRect(0, 0, w, h);

        ctx.save();
        ctx.translate(state.offsetX, state.offsetY);
        ctx.scale(state.zoom, state.zoom);

        // Enable sharp pixels when zoomed in
        ctx.imageSmoothingEnabled = state.zoom < 1.5;

        // Draw base layer
        const baseImg = state.images[state.baseLayer];
        if (baseImg) {
            ctx.drawImage(baseImg, 0, 0);
        }

        // Draw overlays in order
        const overlayOrder = ['psr', 'dem', 'slope', 'confidence', 'ice_ml', 'ice_phase1'];
        for (const key of overlayOrder) {
            const overlay = state.overlays[key];
            if (overlay && overlay.visible && state.images[key]) {
                ctx.globalAlpha = overlay.opacity;
                if (state.splitMode && key === 'ice_ml') {
                    // In split mode, draw ML only on right half
                    const splitPx = state.splitX * state.imgW;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(splitPx, 0, state.imgW - splitPx, state.imgH);
                    ctx.clip();
                    ctx.drawImage(state.images[key], 0, 0);
                    ctx.restore();
                } else if (state.splitMode && key === 'ice_phase1') {
                    // In split mode, draw Phase 1 only on left half
                    const splitPx = state.splitX * state.imgW;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(0, 0, splitPx, state.imgH);
                    ctx.clip();
                    ctx.drawImage(state.images[key], 0, 0);
                    ctx.restore();
                } else {
                    ctx.drawImage(state.images[key], 0, 0);
                }
                ctx.globalAlpha = 1.0;
            }
        }

        ctx.restore();

        // Split line
        if (state.splitMode) {
            const splitScreenX = state.offsetX + state.splitX * state.imgW * state.zoom;
            ctx.save();
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2;
            ctx.shadowColor = '#00d4ff';
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.moveTo(splitScreenX, 0);
            ctx.lineTo(splitScreenX, h);
            ctx.stroke();
            ctx.restore();
        }

        requestAnimationFrame(() => {}); // keep RAF chain alive for smooth interaction
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PAN & ZOOM
    // ═══════════════════════════════════════════════════════════════════════════
    function onMouseDown(e) {
        if (e.button !== 0) return;
        state.dragging = true;
        state.dragStartX = e.clientX;
        state.dragStartY = e.clientY;
        state.lastOffsetX = state.offsetX;
        state.lastOffsetY = state.offsetY;
        viewport.style.cursor = 'grabbing';
    }

    function onMouseMove(e) {
        const rect = viewport.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Update crosshair
        crosshair.style.left = mx + 'px';
        crosshair.style.top = my + 'px';
        crosshair.style.opacity = '1';

        // Pixel coordinates
        const px = Math.floor((mx - state.offsetX) / state.zoom);
        const py = Math.floor((my - state.offsetY) / state.zoom);
        if (px >= 0 && py >= 0 && px < state.imgW && py < state.imgH) {
            coordText.textContent = `X: ${px}  Y: ${py}  |  Zoom: ${(state.zoom * 100).toFixed(0)}%`;
        } else {
            coordText.textContent = `Zoom: ${(state.zoom * 100).toFixed(0)}%`;
        }

        if (state.dragging) {
            state.offsetX = state.lastOffsetX + (e.clientX - state.dragStartX);
            state.offsetY = state.lastOffsetY + (e.clientY - state.dragStartY);
            render();
        }
    }

    function onMouseUp() {
        state.dragging = false;
        viewport.style.cursor = 'grab';
    }

    function onMouseLeave() {
        crosshair.style.opacity = '0';
        state.dragging = false;
        viewport.style.cursor = 'grab';
    }

    function onWheel(e) {
        e.preventDefault();
        const rect = viewport.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(state.minZoom, Math.min(state.maxZoom, state.zoom * zoomFactor));

        // Zoom toward mouse position
        const scale = newZoom / state.zoom;
        state.offsetX = mx - scale * (mx - state.offsetX);
        state.offsetY = my - scale * (my - state.offsetY);
        state.zoom = newZoom;

        render();
    }

    function onClick(e) {
        if (Math.abs(e.clientX - state.dragStartX) > 3 || Math.abs(e.clientY - state.dragStartY) > 3) return;

        const rect = viewport.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const px = Math.floor((mx - state.offsetX) / state.zoom);
        const py = Math.floor((my - state.offsetY) / state.zoom);

        if (px >= 0 && py >= 0 && px < state.imgW && py < state.imgH) {
            showPixelInfo(px, py);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PIXEL INSPECT
    // ═══════════════════════════════════════════════════════════════════════════
    function getPixelColor(layerKey, x, y) {
        const img = state.images[layerKey];
        if (!img) return null;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = img.naturalWidth;
        tempCanvas.height = img.naturalHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(img, 0, 0);
        const pixel = tempCtx.getImageData(x, y, 1, 1).data;
        return { r: pixel[0], g: pixel[1], b: pixel[2], a: pixel[3] };
    }

    function showPixelInfo(px, py) {
        // Sample colors from layers
        const cprColor = getPixelColor('cpr', px, py);
        const psrColor = getPixelColor('psr', px, py);
        const p1Color = getPixelColor('ice_phase1', px, py);
        const mlColor = getPixelColor('ice_ml', px, py);
        const slopeColor = getPixelColor('slope', px, py);
        const confColor = getPixelColor('confidence', px, py);

        // Decode CPR from inferno colormap (approximate)
        let cprValue = '—';
        if (cprColor && cprColor.a > 0) {
            const brightness = (cprColor.r + cprColor.g + cprColor.b) / (3 * 255);
            cprValue = (brightness * 3).toFixed(3);
        }

        // Decode PSR
        const inPSR = psrColor && psrColor.a > 50;

        // Decode Phase 1
        const isPhase1 = p1Color && p1Color.a > 100;

        // Decode ML probability
        let mlProb = '—';
        if (mlColor && mlColor.a > 20) {
            mlProb = ((mlColor.a / 220) * 100).toFixed(1) + '%';
        }

        // Decode slope safety
        let slopeStatus = '—';
        let slopeBadge = '';
        if (slopeColor && slopeColor.a > 50) {
            if (slopeColor.g > 150 && slopeColor.r < 100) {
                slopeStatus = '< 15°';
                slopeBadge = '<span class="popup-badge safe">✓ Safe</span>';
            } else if (slopeColor.r > 200 && slopeColor.g > 100) {
                slopeStatus = '15° – 25°';
                slopeBadge = '<span class="popup-badge caution">⚠ Caution</span>';
            } else if (slopeColor.r > 150) {
                slopeStatus = '> 25°';
                slopeBadge = '<span class="popup-badge danger">✗ Danger</span>';
            }
        }

        // Confidence score
        let confScore = 0;
        if (confColor && confColor.a > 20) {
            confScore = Math.round((confColor.a - 50) / 50);
            confScore = Math.max(0, Math.min(4, confScore));
        }
        const confStars = '★'.repeat(confScore) + '☆'.repeat(4 - confScore);

        popupBody.innerHTML = `
            <div class="popup-row highlight">
                <span class="label">Ice Probability</span>
                <span class="value">${mlProb}</span>
            </div>
            <div class="popup-row">
                <span class="label">Phase 1 Detection</span>
                <span class="value" style="color: ${isPhase1 ? '#00d4ff' : '#555'}">${isPhase1 ? '✓ YES' : '✗ No'}</span>
            </div>
            <div class="popup-row">
                <span class="label">Confidence</span>
                <span class="value" style="color: #f0a500">${confStars}</span>
            </div>
            <div class="popup-divider"></div>
            <div class="popup-row">
                <span class="label">CPR</span>
                <span class="value">${cprValue}</span>
            </div>
            <div class="popup-row">
                <span class="label">In Shadow (PSR)</span>
                <span class="value" style="color: ${inPSR ? '#1a5fb4' : '#555'}">${inPSR ? '✓ YES' : '✗ No'}</span>
            </div>
            <div class="popup-row">
                <span class="label">Slope</span>
                <span class="value">${slopeStatus} ${slopeBadge}</span>
            </div>
            <div class="popup-divider"></div>
            <div class="popup-row">
                <span class="label">Pixel</span>
                <span class="value">(${px}, ${py})</span>
            </div>
        `;

        pixelPopup.classList.add('visible');
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STATS & FEATURE IMPORTANCE
    // ═══════════════════════════════════════════════════════════════════════════
    function loadStats() {
        fetch('data/stats.json')
            .then(r => r.json())
            .then(data => {
                state.stats = data;
                updateMetrics(data);
                renderFeatureBars(data.feature_importance);
            })
            .catch(err => {
                // stats unavailable, using defaults
                // Use defaults
                renderFeatureBars({
                    CPR: 0.351, TRT: 0.272, VOL: 0.162,
                    HLX: 0.123, EVN: 0.065, SRD: 0.025, ODD: 0.003
                });
            });
    }

    function updateMetrics(data) {
        const el = (id) => document.getElementById(id);
        if (data.survey_area_km2) el('metricSurvey').textContent = data.survey_area_km2.toLocaleString();
        if (data.phase1_ice_km2) el('metricIceP1').textContent = data.phase1_ice_km2;
        if (data.psr_area_km2) el('metricPSR').textContent = data.psr_area_km2.toLocaleString();
        if (data.ml_ice_50_km2) el('metricIceML').textContent = data.ml_ice_50_km2;
    }

    function renderFeatureBars(importance) {
        const container = document.getElementById('featureBars');
        if (!container) return;

        const maxVal = Math.max(...Object.values(importance));
        const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]);

        container.innerHTML = sorted.map(([name, val]) => {
            const pct = (val / maxVal * 100).toFixed(1);
            const color = FEATURE_COLORS[name] || '#888';
            return `
                <div class="feature-row">
                    <span class="feature-name">${name}</span>
                    <div class="feature-bar-track">
                        <div class="feature-bar-fill" data-value="${(val * 100).toFixed(1)}%"
                             style="width: 0%; background: ${color};">
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Animate bars
        setTimeout(() => {
            const fills = container.querySelectorAll('.feature-bar-fill');
            sorted.forEach(([name, val], i) => {
                if (fills[i]) {
                    fills[i].style.width = (val / maxVal * 100).toFixed(1) + '%';
                }
            });
        }, 200);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // LAYER CONTROLS
    // ═══════════════════════════════════════════════════════════════════════════
    function setupLayerControls() {
        // Base layer radio buttons
        document.querySelectorAll('input[name="baseLayer"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                state.baseLayer = e.target.value;
                // Update active class
                document.querySelectorAll('[data-layer="cpr"], [data-layer="vol"]').forEach(el => {
                    el.classList.toggle('active', el.dataset.layer === state.baseLayer);
                });
                render();
            });
        });

        // Overlay checkboxes
        const overlayMap = {
            layerPSR: 'psr',
            layerPhase1: 'ice_phase1',
            layerML: 'ice_ml',
            layerConf: 'confidence',
            layerSlope: 'slope',
            layerDEM: 'dem'
        };

        for (const [elId, layerKey] of Object.entries(overlayMap)) {
            const cb = document.getElementById(elId);
            if (!cb) continue;
            cb.addEventListener('change', () => {
                state.overlays[layerKey].visible = cb.checked;
                render();
            });
        }

        // Opacity sliders
        const opacityMap = {
            opacityPSR: 'psr',
            opacityPhase1: 'ice_phase1',
            opacityML: 'ice_ml',
            opacityConf: 'confidence',
            opacitySlope: 'slope',
            opacityDEM: 'dem'
        };

        for (const [elId, layerKey] of Object.entries(opacityMap)) {
            const slider = document.getElementById(elId);
            if (!slider) continue;
            slider.addEventListener('input', () => {
                state.overlays[layerKey].opacity = slider.value / 100;
                render();
            });
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPARE / SPLIT VIEW
    // ═══════════════════════════════════════════════════════════════════════════
    function setupCompareButtons() {
        const btnSingle = document.getElementById('btnSingle');
        const btnSplit = document.getElementById('btnSplit');
        const splitOverlay = document.getElementById('splitOverlay');

        btnSingle.addEventListener('click', () => {
            state.splitMode = false;
            btnSingle.classList.add('active');
            btnSplit.classList.remove('active');
            splitOverlay.classList.remove('active');
            render();
        });

        btnSplit.addEventListener('click', () => {
            state.splitMode = true;
            btnSplit.classList.add('active');
            btnSingle.classList.remove('active');
            splitOverlay.classList.add('active');

            // Enable both layers
            state.overlays.ice_phase1.visible = true;
            state.overlays.ice_ml.visible = true;
            document.getElementById('layerPhase1').checked = true;
            document.getElementById('layerML').checked = true;

            render();
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ZOOM BUTTONS
    // ═══════════════════════════════════════════════════════════════════════════
    function setupZoomControls() {
        document.getElementById('zoomIn').addEventListener('click', () => {
            zoomCenter(1.3);
        });
        document.getElementById('zoomOut').addEventListener('click', () => {
            zoomCenter(0.7);
        });
        document.getElementById('zoomFit').addEventListener('click', () => {
            fitToView();
            render();
        });
    }

    function zoomCenter(factor) {
        const rect = viewport.getBoundingClientRect();
        const cx = rect.width / 2;
        const cy = rect.height / 2;
        const newZoom = Math.max(state.minZoom, Math.min(state.maxZoom, state.zoom * factor));
        const scale = newZoom / state.zoom;
        state.offsetX = cx - scale * (cx - state.offsetX);
        state.offsetY = cy - scale * (cy - state.offsetY);
        state.zoom = newZoom;
        render();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CRATER SELECTOR
    // ═══════════════════════════════════════════════════════════════════════════
    function setupCraterSelector() {
        const sel = document.getElementById('craterSelect');
        // Approximate pixel regions for each crater (in downsampled coords)
        const craterViews = {
            all: { x: 0, y: 0, zoom: null }, // fit
            faustini: { x: 2200, y: 1800, zoom: 1.2 },
            shoemaker: { x: 1500, y: 2000, zoom: 1.0 },
            haworth: { x: 1300, y: 1600, zoom: 1.0 },
            cabeus: { x: 800, y: 2200, zoom: 0.8 },
            shackleton: { x: 1800, y: 1200, zoom: 1.2 }
        };

        sel.addEventListener('change', () => {
            const v = craterViews[sel.value];
            if (!v) return;

            if (sel.value === 'all' || !v.zoom) {
                fitToView();
            } else {
                const rect = viewport.getBoundingClientRect();
                state.zoom = v.zoom;
                state.offsetX = rect.width / 2 - v.x * state.zoom;
                state.offsetY = rect.height / 2 - v.y * state.zoom;
            }
            render();
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MODAL
    // ═══════════════════════════════════════════════════════════════════════════
    function setupModal() {
        const modal = document.getElementById('aboutModal');
        document.getElementById('btnAbout').addEventListener('click', () => modal.classList.add('visible'));
        document.getElementById('modalClose').addEventListener('click', () => modal.classList.remove('visible'));
        modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.remove('visible'); });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // POPUP
    // ═══════════════════════════════════════════════════════════════════════════
    function setupPopup() {
        document.getElementById('popupClose').addEventListener('click', () => {
            pixelPopup.classList.remove('visible');
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // KEYBOARD SHORTCUTS
    // ═══════════════════════════════════════════════════════════════════════════
    function setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                pixelPopup.classList.remove('visible');
                document.getElementById('aboutModal').classList.remove('visible');
            }
            if (e.key === '+' || e.key === '=') zoomCenter(1.2);
            if (e.key === '-') zoomCenter(0.8);
            if (e.key === '0') { fitToView(); render(); }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INIT
    // ═══════════════════════════════════════════════════════════════════════════
    function init() {
        resizeCanvas();
        window.addEventListener('resize', () => { resizeCanvas(); fitToView(); render(); });

        // Map interactions
        viewport.addEventListener('mousedown', onMouseDown);
        viewport.addEventListener('mousemove', onMouseMove);
        viewport.addEventListener('mouseup', onMouseUp);
        viewport.addEventListener('mouseleave', onMouseLeave);
        viewport.addEventListener('wheel', onWheel, { passive: false });
        viewport.addEventListener('click', onClick);

        // Touch support
        let lastTouchDist = 0;
        viewport.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                state.dragging = true;
                state.dragStartX = e.touches[0].clientX;
                state.dragStartY = e.touches[0].clientY;
                state.lastOffsetX = state.offsetX;
                state.lastOffsetY = state.offsetY;
            } else if (e.touches.length === 2) {
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                lastTouchDist = Math.sqrt(dx * dx + dy * dy);
            }
        }, { passive: true });

        viewport.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && state.dragging) {
                state.offsetX = state.lastOffsetX + (e.touches[0].clientX - state.dragStartX);
                state.offsetY = state.lastOffsetY + (e.touches[0].clientY - state.dragStartY);
                render();
            } else if (e.touches.length === 2) {
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (lastTouchDist > 0) {
                    const factor = dist / lastTouchDist;
                    zoomCenter(factor);
                }
                lastTouchDist = dist;
            }
        }, { passive: false });

        viewport.addEventListener('touchend', () => { state.dragging = false; lastTouchDist = 0; });

        // Controls
        setupLayerControls();
        setupCompareButtons();
        setupZoomControls();
        setupCraterSelector();
        setupModal();
        setupPopup();
        setupKeyboard();

        // Load layers
        loadAllLayers();
    }

    // Go!
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
