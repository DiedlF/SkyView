# Keeping the Skyview server up to date

| What | How it is updated | Your part |
|---|---|---|
| Ubuntu security updates | `unattended-upgrades`, daily 06:00-07:00 UTC | nothing |
| Reboots they require | automatic at **02:30 UTC** (03:30 Berlin winter / 04:30 summer), only when `/var/run/reboot-required` exists | nothing; the IPv64 "Skyview data freshness" check alerts if the site does not come back |
| Other Ubuntu updates | by hand, monthly (below) | ~5 min |
| Python packages | direct deps in `requirements.in`, exact versions in the compiled `requirements.txt` (`backend/`, `backend/observations/`); Dependabot opens PRs every Monday | review + merge the PR when CI is green |
| Known vulnerabilities | CI job `audit` (pip-audit) on every push/PR and nightly | fix = merge the Dependabot PR, or bump the pin by hand |
| GitHub Actions versions | Dependabot PR, monthly | merge |
| Leaflet / uPlot (`frontend/vendor/`) | by hand when needed | see below |

Policy file: `deploy/apt/52skyview-unattended-upgrades` (installed by every deploy).

## Monthly OS maintenance

```bash
ssh skyview-vps
apt-get update && apt-get full-upgrade      # keep existing config files when asked
apt-get autoremove --purge
test -f /var/run/reboot-required && systemctl reboot   # ~30 s downtime
```

After a reboot: `systemctl --failed` should be empty, `ufw status` active, and
`curl -s https://skyview.afriedl.ipv64.de/api/health/data` should say `"status":"ok"`
once the observation timer has run (~2 min).

Do **not** start or restart `nftables.service`: it is disabled on purpose, and its
start/stop actions flush ufw's firewall rules.

## Python dependencies

- The deploy installs the lock files exactly; it never upgrades on its own.
- Dependabot groups minor/patch bumps into one PR per environment and sends
  major versions separately. CI runs the web tests (`test`), the ingest tests
  against the ingest lock (`ingest-tests`) and `pip-audit` (`audit`).
- Changing a direct dependency by hand: edit `requirements.in`, then in that
  folder run `pip-compile --no-emit-index-url --strip-extras --output-file=requirements.txt requirements.in`
  (Python 3.12, `pip install pip-tools`). `pip-compile --upgrade-package NAME`
  bumps one package; the existing pins are kept otherwise.
- `zarr`/`numcodecs` in the web app are pinned on purpose (zarr v2 data format)
  and excluded from Dependabot's major updates.
- Dependabot PRs whose CI fails at "pip install" are conflicting sets; close them
  or comment `@dependabot recreate`.

## Frontend libraries

`frontend/vendor/leaflet-<version>/` and `frontend/vendor/uplot-<version>/` hold the
release files and licenses. To update: download the new release files into a new
versioned folder, point `frontend/index.html` at it (and the license links in
`frontend/legal.html`, `frontend/de/legal.html`), check the map in a browser,
delete the old folder.

## Optional: Ubuntu Pro (Livepatch)

Free for personal use on up to 5 machines. Applies most kernel fixes without a
reboot, so the automatic night reboots happen less often:
`pro attach <token>` (token from https://ubuntu.com/pro), then `pro enable livepatch`.
