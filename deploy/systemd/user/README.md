# User-scope units

The `systemctl --user` deployment set. Copy the whole directory and reload:

    cp deploy/systemd/user/trading-*.{service,timer} ~/.config/systemd/user/
    systemctl --user daemon-reload

`cp` dereferences symlinks, so this yields real files either way.

## Why some entries are files and some are symlinks

**The `.service` files are real files, because they genuinely differ** from
their system counterparts in `../`. Three kinds of line are changed and
nothing else:

| removed / changed | failure it causes under `--user` |
|---|---|
| `ProtectKernelModules=yes` | `218/CAPABILITIES` — implies `CapabilityBoundingSet=~CAP_SYS_MODULE`, and a user manager cannot narrow a bounding set. There is no `CapabilityBoundingSet` line to grep for, which is what makes it hard to find. |
| `User=` / `Group=` | `217/USER` — the fatal pair is `User=` **with** `RestrictNamespaces=yes`; systemd must keep `CAP_SYS_ADMIN` across the credential change to build the namespace. `User=` alone is fine, and so is `User=` with `ProtectSystem=strict`, `ProtectHome` or `PrivateTmp`. |
| `WantedBy=multi-user.target` → `default.target` | a system target means nothing to a user manager. |

Every other hardening directive is kept, each verified under `--user` *in
combination with* `ProtectSystem=strict` rather than bare — a bare probe
builds no mount namespace and passes things that fail in situ.

**The `.timer` files are symlinks, because they do not differ at all.** All
three already declare `WantedBy=timers.target`, which is correct in both
scopes. Committing byte-identical copies would create two files that must be
kept in sync for no reason; a symlink makes "no delta" structural instead of a
comment that rots.

If a timer ever *does* need a user-scope change, replace the symlink with a
real file and say why at the top of it, the way the services do.

## The system units are the ones in `../`

They keep `User=` and `ProtectKernelModules` and must. Under
`systemd --system` the manager is privileged, so the bounding set is real
protection — and dropping `User=` there would run these as **root**, leaving
root-owned files in `logs/`, `data/` and `/mnt/backtest/artifacts` that the
next non-root run cannot reopen.

## trading-master-live is ARMED

Its `ExecStart` carries `--live`. Loading the unit is not starting it, and
starting it places orders on the CrossTrade path.
