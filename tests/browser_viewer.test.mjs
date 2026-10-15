// Run with node --test tests/browser_viewer.test.mjs. No npm dependencies.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = await readFile(new URL('../src/peakfit/plot/viewer.js', import.meta.url), 'utf8');

async function viewer({width = 1000, height = 700} = {}) {
    const requests = [], timers = new Map(), axisText = [], frames = [];
    const viewport = {width, height};
    let observer;
    let timerId = 0;
    const context = new Proxy({}, {get: (_target, key) => key === 'fillText'
        ? (...args) => axisText.push(args)
        : key === 'strokeRect' ? (...args) => frames.push(args) : () => {}});
    const control = (value = '') => ({value, checked: true, disabled: false, listeners: {},
        addEventListener(name, callback) { this.listeners[name] = callback; },
        checkValidity() { return this.value !== ''; }, classList: {toggle() {}},
        parentElement: {}, style: {}, add() {}, focus() {}, setPointerCapture() {},
        getBoundingClientRect() { return {left: 0, top: 0, ...viewport}; },
        getContext() { return context; }});
    const elements = Object.fromEntries(['spectrum', 'plane', 'level', 'factor', 'status',
        'labels', 'peak', 'reset', 'cursor', 'filename', 'plane-count', 'aspect'].map(id => [id, control()]));
    elements.aspect.value = '1';
    elements.plane.value = '1'; elements.level.value = '5'; elements.factor.value = '1.4';
    const checks = ['experimental', 'simulated', 'residual'].map(value => control(value));
    class Path2D { moveTo() {} lineTo() {} addPath() {} }
    class DOMMatrix {}
    class ResizeObserver { constructor(callback) { this.callback = callback; observer = this; }
        observe() { this.callback(); } }
    const sandbox = {console, URLSearchParams, DataView, Map, Number, Math, Promise,
        Path2D, DOMMatrix, ResizeObserver, Option: class {}, window: {devicePixelRatio: 1},
        document: {querySelector: selector => elements[selector.slice(1)],
                   querySelectorAll: () => checks},
        requestAnimationFrame: callback => callback(),
        setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
        clearTimeout: id => timers.delete(id),
        fetch: async url => {
            requests.push(url);
            return {ok: true, json: async () => ({filename: 'test.ft3', shape: [2, 101, 101],
                xlim: [10, 0], ylim: [150, 50], x_label: 'H', y_label: 'N', noise: 1,
                layers: ['experimental'], peaks: []}),
                arrayBuffer: async () => new ArrayBuffer(4)};
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
        Object.assign(viewport, {width, height}); observer.callback();
    };
    return {elements, wheel, settle, requests, cursor, axisText, frames, resize};
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
    assert.equal(Number(query.get('level')), Number(app.elements.level.value));
    app.wheel(100, {ctrlKey: true});
    assert.ok(Number(app.elements.level.value) < 7.459);
});

test('Shift-scroll changes spacing independently of base and zoom', async () => {
    const app = await viewer(), before = app.cursor();
    app.wheel(-100, {shiftKey: true});
    assert.equal(Number(app.elements.factor.value), 1.45);
    assert.equal(app.elements.level.value, '5');
    assert.equal(app.cursor(), before);
    await app.settle();
    assert.equal(new URL(app.requests.at(-1), 'http://localhost').searchParams.get('factor'), '1.45');
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
    assert.equal(Number(app.elements.level.value), 1000);
    app.wheel(100000, {ctrlKey: true});
    assert.equal(Number(app.elements.level.value), 0.1);
    app.wheel(-100000, {shiftKey: true});
    assert.equal(Number(app.elements.factor.value), 3);
    app.wheel(100000, {shiftKey: true});
    assert.equal(Number(app.elements.factor.value), 1.01);
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
    assert.equal(query.get('level'), '5');
    assert.equal(query.get('factor'), '1.4');
});


test('Shift-scroll accepts a horizontal wheel delta', async () => {
    const app = await viewer();
    app.wheel(0, {shiftKey: true, deltaX: -100});
    assert.equal(Number(app.elements.factor.value), 1.45);
    assert.equal(app.elements.level.value, '5');
});


test('plot defaults to a centered square and stays square on resize', async () => {
    const app = await viewer();
    let [left, top, w, h] = app.frames.at(-1);
    assert.equal(w, h);
    assert.equal(left - 76, 1000 - 24 - left - w);
    assert.equal(top - 22, 700 - 58 - top - h);
    app.resize(600, 1000);
    [left, top, w, h] = app.frames.at(-1);
    assert.equal(w, h);
    assert.ok(left >= 76 && left + w <= 600 - 24);
    assert.ok(top >= 22 && top + h <= 1000 - 58);
});

test('aspect control offers rectangular plots and the previous fit-window layout', async () => {
    const app = await viewer(), count = app.requests.length;
    for (const ratio of [4 / 3, 16 / 9, 3 / 4]) {
        app.elements.aspect.value = String(ratio);
        app.elements.aspect.listeners.change();
        const [left, top, w, h] = app.frames.at(-1);
        assert.ok(Math.abs(w / h - ratio) < 1e-12);
        assert.ok(left >= 76 && left + w <= 1000 - 24);
        assert.ok(top >= 22 && top + h <= 700 - 58);
    }
    app.elements.aspect.value = 'auto';
    app.elements.aspect.listeners.change();
    assert.deepEqual(app.frames.at(-1), [76, 22, 900, 620]);
    assert.equal(app.requests.length, count);
});

test('aspect changes preserve zoom and use the new rectangle for cursor and pan', async () => {
    const app = await viewer();
    app.wheel(-100);
    const ticks = () => app.axisText.slice(-14).map(([text]) => text);
    const before = ticks();
    app.elements.aspect.value = 'auto';
    app.elements.aspect.listeners.change();
    assert.deepEqual(ticks(), before);
    const [left, top, w, h] = app.frames.at(-1);
    app.elements.spectrum.listeners.pointermove({clientX: left + w / 2, clientY: top + h / 2});
    const center = app.elements.cursor.textContent;
    app.elements.aspect.value = '1';
    app.elements.aspect.listeners.change();
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
