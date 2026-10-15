// Contour geometry stays in sample coordinates. Zoom and pan only transform paths.
const canvas = document.querySelector('#spectrum');
const ctx = canvas.getContext('2d');
const plot = document.querySelector('#plot');
const footer = document.querySelector('#viewer-footer');
const planeInput = document.querySelector('#plane');
const levelInput = document.querySelector('#level');
const factorInput = document.querySelector('#factor');
const aspectInput = document.querySelector('#aspect');
const status = document.querySelector('#status');
const checks = [...document.querySelectorAll('#layers input')];
const colors = {experimental: '#1763b0', simulated: '#c45a13', residual: '#a62d79'};
const cache = new Map();
const inflight = new Map();
const cacheLimit = 64 * 1024 * 1024;
let cacheBytes = 0;
let meta, view, current = new Map(), generation = 0, pending = false, drag = null;
let width = 1, height = 1, contourTimer;
const margin = {left: 76, right: 24, top: 22, bottom: 58};

function message(text, error = false) {
    status.textContent = text;
    status.classList.toggle('error', error);
}

async function get(url) {
    const response = await fetch(url);
    if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.error || `Viewer request failed (${response.status})`);
    }
    return response;
}

function paths(buffer) {
    const positiveCount = new DataView(buffer).getUint32(0, true);
    // DataView makes decoding independent of the browser machine's byte order.
    const values = new DataView(buffer, 4);
    const result = [new Path2D(), new Path2D()];
    let start = true;
    for (let i = 0; i < values.byteLength / 8; i++) {
        if (i === positiveCount) start = true;
        const x = values.getFloat32(i * 8, true), y = values.getFloat32(i * 8 + 4, true);
        if (!Number.isFinite(x) || !Number.isFinite(y)) { start = true; continue; }
        const path = result[i < positiveCount ? 0 : 1];
        if (start) path.moveTo(x, y); else path.lineTo(x, y);
        start = false;
    }
    return result;
}

async function layerPaths(plane, layer, level, factor) {
    const key = `${plane}/${layer}/${level}/${factor}`;
    if (cache.has(key)) {
        const entry = cache.get(key);
        cache.delete(key);
        cache.set(key, entry);
        return entry.result;
    }
    if (inflight.has(key)) return inflight.get(key);
    const promise = (async () => {
        const response = await get(`contours?${new URLSearchParams({plane, layer, level, factor})}`);
        const buffer = await response.arrayBuffer();
        const result = paths(buffer);
        // Budget includes a conservative allowance for the native Path2D copy.
        const bytes = buffer.byteLength * 4;
        if (bytes <= cacheLimit) {
            while (cacheBytes + bytes > cacheLimit && cache.size) {
                const oldest = cache.keys().next().value;
                cacheBytes -= cache.get(oldest).bytes;
                cache.delete(oldest);
            }
            cache.set(key, {result, bytes});
            cacheBytes += bytes;
        }
        return result;
    })().finally(() => inflight.delete(key));
    inflight.set(key, promise);
    return promise;
}

async function load() {
    if (!meta || !planeInput.checkValidity() || !levelInput.checkValidity() || !factorInput.checkValidity()) return;
    const revision = ++generation;
    const plane = Number(planeInput.value) - 1, level = Number(levelInput.value),
          factor = Number(factorInput.value);
    current = new Map();
    redraw();
    message(`Loading plane ${plane + 1}…`);
    try {
        const enabled = checks.filter(check => check.checked && !check.disabled);
        const layers = await Promise.all(enabled.map(async check =>
            [check.value, await layerPaths(plane, check.value, level, factor)]));
        if (revision !== generation) return;
        current = new Map(layers);
        message(`Plane ${plane + 1} of ${meta.shape[0]} · Base ${(level * meta.noise).toPrecision(4)}`);
        redraw();
    } catch (error) {
        if (revision === generation) message(error.message, true);
    }
}

function box() {
    return {left: margin.left, top: margin.top,
            w: Math.max(1, width - margin.left - margin.right),
            h: Math.max(1, height - margin.top - margin.bottom)};
}

function screen(x, y) {
    const {left, top, w, h} = box();
    return [left + (x - view.x0) * w / (view.x1 - view.x0),
            top + h - (y - view.y0) * h / (view.y1 - view.y0)];
}

function sample(px, py) {
    const {left, top, w, h} = box();
    return [view.x0 + (px - left) / w * (view.x1 - view.x0),
            view.y0 + (1 - (py - top) / h) * (view.y1 - view.y0)];
}

function ppm(index, axis) {
    const limits = meta[`${axis}lim`], count = meta.shape[axis === 'x' ? 2 : 1];
    return limits[0] + index / (count - 1) * (limits[1] - limits[0]);
}

function peakSample(peak) {
    return [(peak.x0_ppm - meta.xlim[0]) / (meta.xlim[1] - meta.xlim[0]) * (meta.shape[2] - 1),
            (peak.y0_ppm - meta.ylim[0]) / (meta.ylim[1] - meta.ylim[0]) * (meta.shape[1] - 1)];
}

function redraw() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
}

function draw() {
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!meta || !view) return;
    const {left, top, w, h} = box();
    const sx = w / (view.x1 - view.x0), sy = h / (view.y1 - view.y0);
    const transform = new DOMMatrix([sx, 0, 0, -sy,
        left - view.x0 * sx, top + h + view.y0 * sy]);
    ctx.save();
    ctx.beginPath(); ctx.rect(left, top, w, h); ctx.clip();
    ctx.lineWidth = 1;
    for (const [layer, signedPaths] of current) {
        ctx.strokeStyle = colors[layer];
        signedPaths.forEach((path, sign) => {
            // Transform the geometry, preserving a uniform one-pixel stroke.
            const visible = new Path2D();
            visible.addPath(path, transform);
            ctx.setLineDash(sign ? [4, 3] : []);
            ctx.stroke(visible);
        });
    }
    ctx.setLineDash([]);
    if (document.querySelector('#labels').checked) {
        ctx.font = '12px system-ui'; ctx.fillStyle = '#233047';
        for (const peak of meta.peaks) {
            const [x, y] = screen(...peakSample(peak));
            if (x < left || x > left + w || y < top || y > top + h) continue;
            ctx.fillRect(x - 2, y - 2, 4, 4);
            ctx.fillText(peak.name, x + 6, y - 6);
        }
    }
    ctx.restore();
    ctx.strokeStyle = '#c3cdda'; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.strokeRect(left, top, w, h);
    ctx.font = '12px system-ui'; ctx.fillStyle = '#526178';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let i = 0; i <= 5; i++) {
        const x = view.x0 + (view.x1 - view.x0) * i / 5;
        ctx.fillText(ppm(x, 'x').toFixed(3), left + w * i / 5, top + h + 10);
    }
    ctx.fillText(`${meta.x_label} (ppm)`, left + w / 2, top + h + 38);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i <= 5; i++) {
        const y = view.y0 + (view.y1 - view.y0) * i / 5;
        ctx.fillText(ppm(y, 'y').toFixed(3), left - 10, top + h - h * i / 5);
    }
    ctx.save(); ctx.translate(left - 60, top + h / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center'; ctx.fillText(`${meta.y_label} (ppm)`, 0, 0); ctx.restore();
}

function constrain() {
    for (const [axis, max] of [['x', meta.shape[2] - 1], ['y', meta.shape[1] - 1]]) {
        const span = Math.min(max, Math.max(1, view[`${axis}1`] - view[`${axis}0`]));
        view[`${axis}0`] = Math.min(max - span, Math.max(0, view[`${axis}0`]));
        view[`${axis}1`] = view[`${axis}0`] + span;
    }
}

function reset() {
    if (!meta) return;
    view = {x0: 0, x1: meta.shape[2] - 1, y0: 0, y1: meta.shape[1] - 1};
    document.querySelector('#peak').value = '';
    redraw();
}

function pointer(event) {
    const rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
}

function inside(px, py) {
    const {left, top, w, h} = box();
    return px >= left && px <= left + w && py >= top && py <= top + h;
}

canvas.addEventListener('wheel', event => {
    if (!view || !inside(...pointer(event))) return;
    event.preventDefault();
    const wheelDelta = event.shiftKey ? (event.deltaY || event.deltaX || 0) : event.deltaY;
    const delta = wheelDelta * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1);
    if (event.ctrlKey || event.shiftKey) {
        if (event.ctrlKey) {
            const level = Number(levelInput.value) * Math.exp(-delta * 0.002);
            levelInput.value = Number(Math.max(0.1, Math.min(1000, level)).toPrecision(4));
        } else {
            const multiplier = Number(factorInput.value) - delta * 0.0005;
            factorInput.value = Number(Math.max(1.01, Math.min(3, multiplier)).toFixed(3));
        }
        // A wheel burst changes controls immediately and extracts contours once it settles.
        ++generation;
        clearTimeout(contourTimer);
        contourTimer = setTimeout(load, 100);
        return;
    }
    const [x, y] = sample(...pointer(event));
    const factor = Math.exp(Math.max(-1, Math.min(1, delta * 0.002)));
    for (const [axis, anchor, max] of [['x', x, meta.shape[2] - 1], ['y', y, meta.shape[1] - 1]]) {
        const oldSpan = view[`${axis}1`] - view[`${axis}0`];
        const span = Math.max(1, Math.min(max, oldSpan * factor));
        const fraction = (anchor - view[`${axis}0`]) / oldSpan;
        view[`${axis}0`] = anchor - fraction * span;
        view[`${axis}1`] = view[`${axis}0`] + span;
    }
    constrain(); redraw();
}, {passive: false});

canvas.addEventListener('pointerdown', event => {
    if (!view || event.button !== 0 || !inside(...pointer(event))) return;
    canvas.focus(); canvas.setPointerCapture(event.pointerId);
    drag = {point: pointer(event), view: {...view}};
    canvas.style.cursor = 'grabbing';
});
canvas.addEventListener('pointermove', event => {
    if (!view) return;
    const [px, py] = pointer(event), [x, y] = sample(px, py);
    document.querySelector('#cursor').textContent = inside(px, py)
        ? `${ppm(x, 'x').toFixed(4)}, ${ppm(y, 'y').toFixed(4)} ppm` : '';
    if (!drag) return;
    const {w, h} = box();
    const dx = (px - drag.point[0]) / w * (drag.view.x1 - drag.view.x0);
    const dy = (py - drag.point[1]) / h * (drag.view.y1 - drag.view.y0);
    view = {x0: drag.view.x0 - dx, x1: drag.view.x1 - dx,
            y0: drag.view.y0 + dy, y1: drag.view.y1 + dy};
    constrain(); redraw();
});
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(event, () => { drag = null; canvas.style.cursor = 'crosshair'; });
}
canvas.addEventListener('dblclick', reset);
document.querySelector('#reset').addEventListener('click', reset);
document.querySelector('#labels').addEventListener('change', redraw);
aspectInput.addEventListener('change', resizeCanvas);
document.querySelector('#peak').addEventListener('change', event => {
    if (event.target.value === '') return;
    const [x, y] = peakSample(meta.peaks[Number(event.target.value)]);
    view = {x0: x - 24, x1: x + 24, y0: y - 24, y1: y + 24};
    constrain(); redraw();
});
for (const control of [planeInput, levelInput, factorInput, ...checks]) control.addEventListener('change', load);
canvas.addEventListener('keydown', event => {
    if (!meta || !['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'Home') { reset(); return; }
    planeInput.value = Math.max(1, Math.min(meta.shape[0], Number(planeInput.value) +
        (event.key === 'ArrowRight' ? 1 : -1)));
    load();
});
function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    width = canvas.getBoundingClientRect().width;
    const aspect = Number(aspectInput.value);
    // Presets use the full width, even when the resulting plot extends below the viewport.
    const plotTop = plot.getBoundingClientRect().top + window.scrollY;
    height = aspectInput.value === 'auto'
        ? Math.max(160, window.innerHeight - plotTop - footer.getBoundingClientRect().height - 2)
        : Math.max(1, width - margin.left - margin.right) / aspect + margin.top + margin.bottom;
    const cssHeight = `${height}px`;
    if (canvas.style.height !== cssHeight) canvas.style.height = cssHeight;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    redraw();
}
new ResizeObserver(resizeCanvas).observe(plot);
window.addEventListener('resize', resizeCanvas);

try {
    meta = await (await get('metadata')).json();
    document.title = `${meta.filename} | PeakFit`;
    document.querySelector('#filename').textContent = meta.filename;
    planeInput.max = meta.shape[0];
    document.querySelector('#plane-count').textContent = `/ ${meta.shape[0]}`;
    for (const check of checks) {
        check.disabled = !meta.layers.includes(check.value);
        if (check.disabled) { check.checked = false; check.parentElement.title = 'Supply --results to enable'; }
    }
    document.querySelector('#labels').disabled = !meta.peaks.length;
    const peakSelect = document.querySelector('#peak');
    peakSelect.disabled = !meta.peaks.length;
    meta.peaks.forEach((peak, index) => peakSelect.add(new Option(peak.name, index)));
    reset(); await load();
} catch (error) { message(error.message, true); }
