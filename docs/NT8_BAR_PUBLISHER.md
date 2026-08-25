# NT8 bar publisher — the contract the NinjaScript must honour

Writes one file per (symbol, timeframe) into the share this box reads as
`/mnt/backtest/artifacts/nt8_bars/` (override with `$BT_NT8_SPOOL`).

    MNQ_1h.csv
    ts,open,high,low,close,volume
    2026-08-25T17:00:00Z,29260.00,29280.25,29250.75,29274.75,12345

Rules, in the order they matter:

1. APPEND ON BAR CLOSE. `Calculate.OnBarClose`, one line per closed bar. The
   Linux side drops anything still forming anyway, but a script appending on
   every tick makes the file grow at tick rate for no benefit.
2. TIMESTAMPS CARRY AN OFFSET. `Z` or `+00:00`. A naive timestamp is REFUSED,
   because NT8 writes in the instrument's or the workstation's timezone unless
   the script converts, and guessed wrong the series shifts by hours while
   still looking like a market.
3. `ts` IS THE BAR'S CLOSE TIME. That is NinjaTrader's own convention and the
   reader expects it. If the script is changed to stamp the open instead, say
   so in a header line — `# stamp=open` — as the first line of the file.
4. NEVER REWRITE HISTORY. Append only. The reader dedupes on timestamp and
   keeps the last row, so a corrected bar can be re-appended, but rewriting the
   file underneath a reader is how a frame arrives half-written.
5. ONE INSTRUMENT PER FILE, root symbol in the name (`MNQ`, not `MNQ 12-26`).
   The expiry is not in the filename; the reader has no contract calendar and
   will not guess one.

Publish the timeframe the strategies trade (`1h` today). Publishing `1m`
instead also works — the reader aggregates it with the lake's own resampler —
and is the more flexible choice if more than one timeframe will be traded.
