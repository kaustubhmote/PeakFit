// Run with node --test tests/browser_viewer.test.mjs. No npm dependencies.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = await readFile(new URL('../src/peakfit/plot/viewer.js', import.meta.url), 'utf8');

async function viewer({width = 1000, height = 700} = {}) {
    const requests = [], timers = new Map(), axisText = [], frames = [], strokes = [];
    let drawn = [], holdContours = false, failNext = false;
    const releases = [];
    const viewport = {width, height};
    let observer, bitmapResets = 0, pixelWidth = 0, pixelHeight = 0;
    let timerId = 0;
    const context = new Proxy({}, {get: (_target, key) => key === 'fillText'
        ? (...args) => axisText.push(args)
        : key === 'clearRect' ? () => { drawn = []; }
        : key === 'stroke' ? path => { if (path.source?.pointCount) drawn.push(path.source); }
        : key === 'strokeRect' ? (...args) => { frames.push(args); strokes.push([...drawn]); } : () => {}});
    const control = (value = '') => ({value, checked: true, disabled: false, listeners: {},
        addEventListener(name, callback) { this.listeners[name] = callback; },
        checkValidity() { return this.value !== ''; }, classList: {toggle() {}},
        parentElement: {}, style: {}, add() {}, focus() {}, setPointerCapture() {},
        getBoundingClientRect() { return {left: 0, top: 0, ...viewport}; },
        getContext() { return context; }});
    const elements = Object.fromEntries(['spectrum', 'plane', 'level', 'factor', 'status',
        'labels', 'peak', 'reset', 'cursor', 'filename', 'plane-count', 'aspect', 'plot', 'plot-area', 'plot-width'].map(id => [id, control()]));
    Object.defineProperties(elements.spectrum, {
        width: {get: () => pixelWidth, set: value => { pixelWidth = value; bitmapResets++; }},
        height: {get: () => pixelHeight, set: value => { pixelHeight = value; bitmapResets++; }},
    });
    elements['plot-area'].getBoundingClientRect = () => ({left: 0, top: 0, width: viewport.width + 2});
    elements['plot-width'].value = '100';
    elements.aspect.value = '1';
    elements.plane.value = '1'; elements.level.value = '5'; elements.factor.value = '1.4';
    elements.factor.options = [2, 1.8, 1.6, 1.4, 1.3, 1.2].map(value => ({value: String(value)}));
    const checks = ['experimental', 'simulated', 'residual'].map(value => control(value));
    class Path2D {
        pointCount = 0;
        moveTo() { this.pointCount++; }
        lineTo() { this.pointCount++; }
        addPath(path) { this.source = path; }
    }
    class DOMMatrix {}
    class ResizeObserver { constructor(callback) { this.callback = callback; observer = this; }
        observe() { this.callback(); } }
    const sandbox = {console, URLSearchParams, DataView, Map, Number, Math, Promise,
        Path2D, DOMMatrix, ResizeObserver, Option: class {}, window: {devicePixelRatio: 1, innerHeight: height + 50, scrollY: 0, addEventListener() {}},
        document: {querySelector: selector => elements[selector.slice(1)],
                   querySelectorAll: () => checks},
        requestAnimationFrame: callback => callback(),
        setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
        clearTimeout: id => timers.delete(id),
        fetch: async url => {
            requests.push(url);
            if (holdContours && url.startsWith('contours?')) {
                await new Promise(resolve => releases.push(resolve));
            }
            if (failNext && url.startsWith('contours?')) {
                failNext = false;
                return {ok: false, status: 500, json: async () => ({error: 'Contour extraction failed'})};
            }
            return {ok: true, json: async () => ({filename: 'test.ft3', shape: [2, 101, 101],
                xlim: [10, 0], ylim: [150, 50], x_label: 'H', y_label: 'N', noise: 1,
                layers: ['experimental'], contour_count: 25, peaks: []}),
                arrayBuffer: async () => {
                    const levels = new URL(url, 'http://localhost').searchParams.get('levels').split(',').map(Number);
                    const buffer = new ArrayBuffer(4 + levels.length * 32), data = new DataView(buffer);
                    data.setUint32(0, levels.length, true);
                    levels.forEach((height, i) => {
                        const offset = 4 + i * 32;
                        data.setFloat64(offset, height, true);
                        data.setUint32(offset + 8, 2, true);
                        data.setUint32(offset + 12, 0, true);
                        data.setFloat32(offset + 16, 0, true); data.setFloat32(offset + 20, 0, true);
                        data.setFloat32(offset + 24, 1, true); data.setFloat32(offset + 28, 1, true);
                    });
                    return buffer;
                }};
        }};
    await vm.runInNewContext(`(async () => { ${source}\n })()`, sandbox);
    const wheel = (deltaY, modifiers = {}) => {
        let prevented = false;
        elements.spectrum.listeners.wheel({clientX: 500, clientY: 350, deltaY, deltaMode: 0,
            preventDefault() { prevented = true; }, ...modifiers});
        assert.equal(prevented, true);
    };
    const settle = async () => {
        const pending = [...timers.values()]; timers.clear();
        await Promise.all(pending.map(callback => callback()));
    };
    const cursor = () => {
        elements.spectrum.listeners.pointermove({clientX: 600, clientY: 400});
        return elements.cursor.textContent;
    };
    const resize = (width, height) => {
        Object.assign(viewport, {width, height});
        sandbox.window.innerHeight = height + 50; observer.callback();
    };
    const blockContours = () => { holdContours = true; };
    const releaseContours = () => { holdContours = false; releases.splice(0).forEach(resolve => resolve()); };
    const failContours = () => { failNext = true; };
    return {elements, wheel, settle, requests, cursor, axisText, frames, resize, strokes,
            blockContours, releaseContours, failContours, bitmapResets: () => bitmapResets};
}

test('Ctrl-scroll changes base, preserves zoom, and coalesces wheel bursts', async () => {
    const app = await viewer(), before = app.cursor(), count = app.requests.length;
    app.wheel(-100, {ctrlKey: true}); app.wheel(-100, {ctrlKey: true});
    assert.ok(Number(app.elements.level.value) > 5);
    assert.equal(app.elements.factor.value, '1.4');
    assert.equal(app.requests.length, count);
    assert.equal(app.cursor(), before);
    await app.settle();
    assert.equal(app.requests.length, count + 1);
    const query = new URL(app.requests.at(-1), 'http://localhost').searchParams;
    assert.equal(query.get('levels').split(',').length, 2);
    app.wheel(100, {ctrlKey: true});
    assert.ok(Number(app.elements.level.value) < 7.459);
});

test('Shift-scroll changes spacing independently of base and zoom', async () => {
    const app = await viewer(), before = app.cursor(), firstPath = app.strokes.at(-1)[0];
    app.wheel(-100, {shiftKey: true});
    assert.equal(Number(app.elements.factor.value), 1.6);
    assert.equal(app.elements.level.value, '5');
    assert.equal(app.cursor(), before);
    await app.settle();
    assert.equal(new URL(app.requests.at(-1), 'http://localhost').searchParams.get('levels').split(',').length, 24);
    assert.equal(app.strokes.at(-1)[0], firstPath);
    app.wheel(100, {shiftKey: true});
    assert.equal(Number(app.elements.factor.value), 1.4);
});

test('plain scroll zooms around cursor without fetching new contours', async () => {
    const app = await viewer(), before = app.cursor(), count = app.requests.length;
    app.wheel(-100);
    assert.notEqual(app.cursor(), before);
    assert.equal(app.elements.level.value, '5');
    assert.equal(app.elements.factor.value, '1.4');
    assert.equal(app.requests.length, count);
});

test('modifier scrolling stays within valid contour bounds', async () => {
    const app = await viewer();
    app.wheel(-100000, {ctrlKey: true});
    assert.ok(Number(app.elements.level.value) <= 1000);
    assert.ok(Number(app.elements.level.value) > 0.1);
    app.wheel(100000, {ctrlKey: true});
    assert.ok(Number(app.elements.level.value) >= 0.1);
    assert.ok(Number(app.elements.level.value) < 5);
    app.wheel(-100000, {shiftKey: true});
    assert.equal(Number(app.elements.factor.value), 2);
    app.wheel(100000, {shiftKey: true});
    assert.equal(Number(app.elements.factor.value), 1.2);
});

test('changing planes preserves zoom and contour settings', async () => {
    const app = await viewer();
    app.wheel(-100);
    const before = app.cursor();
    app.elements.plane.value = '2';
    await app.elements.plane.listeners.change();
    assert.equal(app.cursor(), before);
    const query = new URL(app.requests.at(-1), 'http://localhost').searchParams;
    assert.equal(query.get('plane'), '1');
    const levels = query.get('levels').split(',').map(Number);
    assert.equal(levels.length, 25);
    assert.equal(levels[0], 5);
    assert.equal(levels[1], 7);
});


test('Shift-scroll accepts a horizontal wheel delta', async () => {
    const app = await viewer();
    app.wheel(0, {shiftKey: true, deltaX: -100});
    assert.equal(Number(app.elements.factor.value), 1.6);
    assert.equal(app.elements.level.value, '5');
});


test('square plots use full width and can extend below the viewport', async () => {
    const app = await viewer();
    let [left, top, w, h] = app.frames.at(-1);
    assert.equal(w, h);
    assert.equal(left, 76);
    assert.equal(top, 22);
    assert.equal(w, 900);
    assert.ok(h > 700);
    assert.equal(app.elements.spectrum.style.height, '980px');
    app.resize(600, 1000);
    [left, top, w, h] = app.frames.at(-1);
    assert.equal(w, h);
    assert.ok(left >= 76 && left + w <= 600 - 24);
    assert.ok(top >= 22 && top + h <= 1000 - 58);
});

test('floating-point aspect ratios preserve width and use cached contours', async () => {
    const app = await viewer(), count = app.requests.length;
    for (const ratio of [4 / 3, 16 / 9, 3 / 4]) {
        app.elements.aspect.value = String(ratio);
        app.elements.aspect.listeners.input();
        const [left, top, w, h] = app.frames.at(-1);
        assert.ok(Math.abs(w / h - ratio) < 1e-12);
        assert.ok(left >= 76 && left + w <= 1000 - 24);
        assert.equal(left, 76);
        assert.equal(top, 22);
        assert.equal(w, 900);
    }
    assert.equal(app.requests.length, count);
});

test('aspect changes preserve zoom and use the new rectangle for cursor and pan', async () => {
    const app = await viewer();
    app.wheel(-100);
    const ticks = () => app.axisText.slice(-14).map(([text]) => text);
    const before = ticks();
    app.elements.aspect.value = '1.7';
    app.elements.aspect.listeners.input();
    assert.deepEqual(ticks(), before);
    const [left, top, w, h] = app.frames.at(-1);
    app.elements.spectrum.listeners.pointermove({clientX: left + w / 2, clientY: top + h / 2});
    const center = app.elements.cursor.textContent;
    app.elements.aspect.value = '1';
    app.elements.aspect.listeners.input();
    const [squareLeft, squareTop, squareW, squareH] = app.frames.at(-1);
    app.elements.spectrum.listeners.pointermove({clientX: squareLeft + squareW / 2,
        clientY: squareTop + squareH / 2});
    assert.equal(app.elements.cursor.textContent, center);
    app.elements.spectrum.listeners.pointerdown({button: 0, pointerId: 1,
        clientX: squareLeft + squareW / 2, clientY: squareTop + squareH / 2});
    app.elements.spectrum.listeners.pointermove({clientX: squareLeft + squareW / 2 + 20,
        clientY: squareTop + squareH / 2});
    assert.notDeepEqual(ticks(), before);
});

test('portrait plots keep their width on short screens and cursor coordinates survive page scroll', async () => {
    const app = await viewer();
    app.elements.aspect.value = '0.75';
    app.elements.aspect.listeners.input();
    assert.equal(app.frames.at(-1)[2], 900);
    assert.equal(app.frames.at(-1)[3], 1200);
    assert.equal(app.elements.spectrum.style.height, '1280px');
    app.resize(1000, 400);
    assert.equal(app.elements.spectrum.style.height, '1280px');
    const before = app.cursor();
    app.elements.spectrum.getBoundingClientRect = () => ({left: 0, top: -300, width: 1000});
    app.elements.spectrum.listeners.pointermove({clientX: 600, clientY: 100});
    assert.equal(app.elements.cursor.textContent, before);
});


test('decimal width percentages resize the plot independently of the aspect ratio and zoom', async () => {
    const app = await viewer();
    app.wheel(-100);
    const ticks = () => app.axisText.slice(-14).map(([text]) => text);
    const before = ticks(), count = app.requests.length;
    app.elements.aspect.value = '1.25';
    app.elements.aspect.listeners.input();
    for (const percent of [50, 82.5, 150]) {
        app.elements['plot-width'].value = String(percent);
        app.elements['plot-width'].listeners.input();
        const [left, top, w, h] = app.frames.at(-1);
        assert.equal(left, 76);
        assert.equal(top, 22);
        assert.equal(w, 1000 * percent / 100 - 100);
        assert.equal(h, w / 1.25);
        assert.equal(Number.parseFloat(app.elements.plot.style.width), 1000 * percent / 100 + 2);
        assert.deepEqual(ticks(), before);
    }
    app.resize(800, 700);
    assert.equal(Number.parseFloat(app.elements.plot.style.width), 1202);
    assert.equal(app.requests.length, count);
});

test('incomplete or invalid sizing inputs leave the current layout intact', async () => {
    const app = await viewer(), before = app.frames.at(-1);
    for (const field of [app.elements.aspect, app.elements['plot-width']]) {
        const original = field.value;
        for (const value of ['', '0', '-1', 'Infinity']) {
            field.value = value;
            field.listeners.input();
            assert.deepEqual(app.frames.at(-1), before);
        }
        field.value = original;
    }
});


test('a contour update keeps the previous spectrum visible while its request is pending', async () => {
    const app = await viewer();
    assert.ok(app.strokes.at(-1).length > 0);
    app.blockContours();
    app.elements.level.value = '6';
    const update = app.elements.level.listeners.change();
    assert.ok(app.strokes.at(-1).length > 0, 'contour request must not produce an empty frame');
    app.releaseContours();
    await update;
    assert.ok(app.strokes.at(-1).length > 0);
});


test('a base step reuses 24 paths and requests only the new highest height', async () => {
    const app = await viewer(), previous = new Set(app.strokes.at(-1));
    assert.equal(previous.size, 25);
    app.wheel(-100, {ctrlKey: true});
    await app.settle();
    const requested = new URL(app.requests.at(-1), 'http://localhost').searchParams.get('levels').split(',');
    assert.equal(requested.length, 1);
    assert.equal(app.strokes.at(-1).filter(path => previous.has(path)).length, 24);
    const count = app.requests.length;
    app.wheel(100, {ctrlKey: true});
    await app.settle();
    assert.equal(app.requests.length, count);
    assert.deepEqual(new Set(app.strokes.at(-1)), previous);
});

test('failed contour extraction preserves the visible spectrum and allows a retry', async () => {
    const app = await viewer(), previous = app.strokes.at(-1);
    app.failContours();
    app.elements.factor.value = '1.6';
    await app.elements.factor.listeners.change();
    assert.equal(app.strokes.at(-1), previous);
    assert.equal(app.elements.status.textContent, 'Contour extraction failed');
    await app.elements.factor.listeners.change();
    assert.equal(app.strokes.at(-1).length, 25);
    assert.match(app.elements.status.textContent, /Plane 1 of 2/);
});

test('an older contour response cannot replace a newer completed level set', async () => {
    const app = await viewer();
    app.blockContours();
    app.elements.level.value = '6';
    const older = app.elements.level.listeners.change();
    app.elements.level.value = '5';
    await app.elements.level.listeners.change();
    const newest = app.strokes.at(-1);
    app.releaseContours();
    await older;
    assert.equal(app.strokes.at(-1), newest);
});


test('unchanged layout notifications do not reset the visible canvas bitmap', async () => {
    const app = await viewer(), count = app.bitmapResets();
    app.resize(1000, 700);
    assert.equal(app.bitmapResets(), count);
});


test('returning to a preset multiplier reuses the previously calculated heights', async () => {
    const app = await viewer();
    const previous = new Set(app.strokes.at(-1));
    app.wheel(-100, {shiftKey: true});
    await app.settle();
    assert.equal(Number(app.elements.factor.value), 1.6);
    const count = app.requests.length;
    app.wheel(100, {shiftKey: true});
    await app.settle();
    assert.equal(Number(app.elements.factor.value), 1.4);
    assert.equal(app.requests.length, count);
    assert.deepEqual(new Set(app.strokes.at(-1)), previous);
});
