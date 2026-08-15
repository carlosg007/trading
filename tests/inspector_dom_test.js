#!/usr/bin/env node
/*
 * inspector_dom_test.js - exercise the trade log's JavaScript for real.
 *
 *     node tests/inspector_dom_test.js <report.html>
 *
 * Run by tests/test_report_gates.py, which generates a small report first and
 * skips this file when node is not installed.
 *
 * The Python suite can assert that the search box, the sort headers and the
 * modal are PRESENT in the emitted HTML. It cannot assert that clicking a row
 * draws the right bars, and that is the half most likely to be wrong: an
 * off-by-one in the window slice, or an epoch in the wrong unit, produces a
 * chart that renders beautifully and shows a different trade.
 *
 * So this parses the generated table, builds the smallest DOM the report's own
 * script needs, runs that script unmodified, and drives it: type in the search
 * box, click a header, click a row, press Escape. Plotly is stubbed to capture
 * what it was asked to draw, and the captured traces are checked against the
 * embedded INSPECTOR payload.
 *
 * Exits non-zero on the first failed check, like every other suite here.
 */

'use strict';

const fs = require('fs');

const file = process.argv[2];
if (!file) { console.error('usage: inspector_dom_test.js <report.html>'); process.exit(2); }
const HTML = fs.readFileSync(file, 'utf8');

let failures = 0;
function check(label, ok, detail) {
  console.log('  ' + (ok ? 'PASS' : 'FAIL') + '  ' + label + (detail ? '  [' + detail + ']' : ''));
  if (!ok) failures++;
  return ok;
}

/* ---------------------------------------------------------------- DOM stub */
class El {
  constructor(tag, attrs = {}, text = '') {
    this.tagName = tag.toUpperCase();
    this.attrs = Object.assign({}, attrs);
    this._text = text;
    this.innerHTML = '';
    this.children = [];
    this.style = {};
    this.hidden = false;
    this.value = '';
    this.listeners = {};
    this.focused = 0;
  }
  /* Real textContent aggregates descendants; the row-level search in the
     report reads tr.textContent, so a stub that returned only its own text
     would make every search miss and the test would be measuring itself. */
  get textContent() {
    return this.children.length
      ? this.children.map((c) => c.textContent).join('')
      : this._text;
  }
  set textContent(v) { this._text = v; this.children = []; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  fire(type, ev) {
    (this.listeners[type] || []).forEach((fn) => fn(Object.assign({ target: this,
      preventDefault() {} }, ev)));
  }
  focus() { this.focused++; }
  appendChild(node) {
    const i = this.children.indexOf(node);
    if (i !== -1) this.children.splice(i, 1);   // appendChild MOVES an existing node
    this.children.push(node);
    return node;
  }
  get rows() { return this.children.filter((c) => c.tagName === 'TR'); }
  get cells() { return this.children.filter((c) => c.tagName === 'TD' || c.tagName === 'TH'); }
}

/* ------------------------------------------------------- parse the report */
function attrs(tag) {
  const out = {};
  const re = /([a-zA-Z-]+)="([^"]*)"/g;
  let m;
  while ((m = re.exec(tag)) !== null) out[m[1]] = m[2];
  return out;
}
function stripTags(s) { return s.replace(/<[^>]*>/g, '').replace(/&amp;/g, '&').trim(); }

const tableHtml = (HTML.match(/<table class="grid trades" id="trade-table">[\s\S]*?<\/table>/) || [''])[0];
if (!tableHtml) { console.error('no trade table in ' + file); process.exit(2); }

const table = new El('table');
const thead = new El('thead');
const headRow = new El('tr');
const headRe = /<th([^>]*)>([\s\S]*?)<\/th>/g;
let hm;
while ((hm = headRe.exec(tableHtml)) !== null) {
  headRow.appendChild(new El('th', attrs(hm[1]), stripTags(hm[2])));
}
thead.appendChild(headRow);
table.tHead = thead;

const tbody = new El('tbody');
const rowRe = /<tr([^>]*data-trade[^>]*)>([\s\S]*?)<\/tr>/g;
let rm;
while ((rm = rowRe.exec(tableHtml)) !== null) {
  const tr = new El('tr', attrs(rm[1]));
  const cellRe = /<td([^>]*)>([\s\S]*?)<\/td>/g;
  let cm;
  while ((cm = cellRe.exec(rm[2])) !== null) {
    tr.appendChild(new El('td', attrs(cm[1]), stripTags(cm[2])));
  }
  tbody.appendChild(tr);
}
table.tBodies = [tbody];

const nodes = {
  'trade-table': table,
  'trade-search': new El('input'),
  'trade-count': new El('span'),
  'trade-modal': new El('div'),
  'modal-close': new El('button'),
  'modal-title': new El('h3'),
  'modal-facts': new El('div'),
  'modal-chart': new El('div'),
};
nodes['trade-modal'].hidden = true;

const drawn = [];
global.document = {
  getElementById: (id) => nodes[id] || null,
  listeners: {},
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
  fire(type, ev) { (this.listeners[type] || []).forEach((fn) => fn(ev)); },
};
global.window = global;
global.Plotly = {
  newPlot(id, traces, layout, config) { drawn.push({ id, traces, layout, config }); },
  purge() { drawn.push({ purged: true }); },
};

/* ---------------------------------------- run the report's OWN script tag */
const payload = HTML.match(/window\.INSPECTOR=(\{[\s\S]*?\});<\/script>/);
if (!payload) { console.error('no INSPECTOR payload in ' + file); process.exit(2); }
global.INSPECTOR = JSON.parse(payload[1]);
window.INSPECTOR = global.INSPECTOR;

const scripts = HTML.match(/<script>([\s\S]*?)<\/script>/g) || [];
const iife = scripts.map((s) => s.replace(/^<script>/, '').replace(/<\/script>$/, ''))
  .filter((s) => s.indexOf('trade-table') !== -1)[0];
if (!iife) { console.error('no trade-table script in ' + file); process.exit(2); }
(0, eval)(iife);

/* ------------------------------------------------------------- the checks */
console.log('\nTrade log DOM (' + file.split('/').pop() + ')');
const rows = tbody.rows;
check('rows were parsed out of the report', rows.length > 3, rows.length + ' rows');
check('the row count is reported on load',
  /rows$/.test(nodes['trade-count'].textContent), nodes['trade-count'].textContent);

/* Search */
const needle = rows[1].cells[1].textContent.slice(0, 16);
nodes['trade-search'].value = needle;
nodes['trade-search'].fire('input');
const visible = rows.filter((r) => r.style.display !== 'none');
check('search filters the table', visible.length >= 1 && visible.length < rows.length,
  visible.length + ' of ' + rows.length + ' match "' + needle + '"');
check('the match is the row searched for', visible.indexOf(rows[1]) !== -1);
check('the filtered count is reported',
  /of \d+ shown/.test(nodes['trade-count'].textContent), nodes['trade-count'].textContent);
nodes['trade-search'].value = '';
nodes['trade-search'].fire('input');
check('clearing the search restores every row',
  rows.filter((r) => r.style.display !== 'none').length === rows.length);

/* Sort - column 8 is net P&L, which is signed and numeric */
const pnlCol = 8;
const th = headRow.cells[pnlCol];
th.fire('click');
let order = tbody.rows.map((r) => parseFloat(r.cells[pnlCol].getAttribute('data-v')));
let asc = order.every((v, i) => i === 0 || order[i - 1] <= v);
check('clicking a header sorts ascending, numerically', asc,
  order.slice(0, 3).map((v) => v.toFixed(2)).join(', ') + ' …');
check('the sorted column is marked for screen readers',
  th.getAttribute('aria-sort') === 'ascending');
th.fire('click');
order = tbody.rows.map((r) => parseFloat(r.cells[pnlCol].getAttribute('data-v')));
check('clicking again sorts descending',
  order.every((v, i) => i === 0 || order[i - 1] >= v) &&
  th.getAttribute('aria-sort') === 'descending');

/* A text column must not sort as a number, and vice versa */
headRow.cells[1].fire('click');
const times = tbody.rows.map((r) => r.cells[1].textContent);
check('a timestamp column sorts chronologically',
  times.every((v, i) => i === 0 || times[i - 1] <= v));

/* Trade inspector */
const T = INSPECTOR.trades, B = INSPECTOR.bars;
const pick = 2;
rows[pick].fire('click');
const plot = drawn.filter((d) => !d.purged).pop();
check('clicking a row opens the modal', nodes['trade-modal'].hidden === false);
check('and draws a chart', !!plot && plot.id === 'modal-chart');

const candle = plot.traces[0];
const t = T[pick];
check('the chart is a candlestick', candle.type === 'candlestick');
check('it spans the trade window, not the whole backtest',
  candle.x.length === t.hi - t.lo + 1,
  candle.x.length + ' bars, window ' + t.lo + '→' + t.hi);
check('the first candle is the window start',
  candle.x[0].getTime() === B.t[t.lo] && candle.open[0] === B.o[t.lo]);
check('OHLC arrays are all the same length',
  candle.open.length === candle.x.length && candle.high.length === candle.x.length &&
  candle.low.length === candle.x.length && candle.close.length === candle.x.length);
check('highs are not below lows anywhere',
  candle.high.every((h, i) => h >= candle.low[i]));

/* The epoch-unit bug: seconds instead of milliseconds renders the trade in
   1970 and nothing raises. Assert the dates land in a plausible decade. */
const year = candle.x[0].getUTCFullYear();
check('bar timestamps are milliseconds, not seconds', year > 2000 && year < 2100,
  'first bar ' + candle.x[0].toISOString());

const entry = plot.traces.find((tr) => tr.name === 'Entry');
const exit = plot.traces.find((tr) => tr.name === 'Exit');
check('an entry marker is drawn at the entry bar',
  !!entry && entry.x[0].getTime() === B.t[t.e] && entry.y[0] === B.o[t.e]);
check('an exit marker is drawn at the exit bar',
  !!exit && exit.x[0].getTime() === B.t[t.x] && exit.y[0] === B.o[t.x]);
check('the exit is at or after the entry', B.t[t.x] >= B.t[t.e]);
check('entry and exit differ by shape, not colour alone',
  entry.marker.symbol === 'triangle-up' && exit.marker.symbol === 'triangle-down');
check('and carry a printed label',
  entry.text[0] === 'IN' && exit.text[0] === 'OUT');
check('the modal titles the trade it drew',
  nodes['modal-title'].textContent.indexOf('Trade ' + rows[pick].cells[0].textContent) === 0,
  nodes['modal-title'].textContent);
check('the modal lists the trade facts',
  /Net P&L/.test(nodes['modal-facts'].innerHTML));

/* Close paths */
document.fire('keydown', { key: 'Escape' });
check('Escape closes the modal', nodes['trade-modal'].hidden === true);
check('and releases the chart', drawn[drawn.length - 1].purged === true);

rows[pick].fire('keydown', { key: 'Enter' });
check('Enter on a row opens it too (keyboard reachable)',
  nodes['trade-modal'].hidden === false);
nodes['modal-close'].fire('click');
check('the close button closes it', nodes['trade-modal'].hidden === true);
check('focus returns to the row that opened the modal', rows[pick].focused > 0);

console.log(failures ? '\nFAILED — ' + failures + ' check(s)' : '\nAll DOM checks passed.');
process.exit(failures ? 1 : 0);
