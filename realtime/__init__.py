"""
realtime - the live regime service.

Three modules, and the split between them is the point:

  * `regime_daemon`      writes.  Classifies live bars into the four-quadrant
                                  standard and caches the answer to disk.
  * `regime_reader`      reads.   A dependency-light, non-blocking reader for
                                  execution scripts that must not pay for
                                  pandas, pandas_ta or a config validation
                                  pass just to ask "what quadrant is NQ in".
  * `crosstrade_formatter` formats. Wire payloads only; it sends nothing.
                                  `live/dispatcher.py` remains the only module
                                  in this repository that sends an order
                                  anywhere.

Nothing in `backtest/` may import from here, for the same reason nothing there
may import from `portfolio/`: a research run whose numbers depended on live
account state would be tuned to a broker rather than to a market.
"""
