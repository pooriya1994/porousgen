# Releasing PorousGen

Every version cited in a publication must be an immutable, archived release.

## One-time set-up: GitHub -> Zenodo archiving
1. Log in at https://zenodo.org with the GitHub account that owns the repository.
2. Zenodo -> *GitHub* (account menu) -> switch **on** `pooriya1994/porousgen`.
   From then on every GitHub *release* is archived automatically and gets a
   version DOI; the *concept DOI* always resolves to the latest version.
3. Add the concept DOI to `CITATION.cff` (`doi:` field) and the README badge.

## Each release
1. Make sure CI is green on `main` (Linux, Windows, macOS; Python 3.9 and 3.12).
2. Bump the version in **three** places, identically:
   `pyproject.toml` (`version`), `porousgen/generator.py` (`__version__`),
   `CITATION.cff` (`version`, `date-released`).
3. Move the `[Unreleased]` notes in `CHANGELOG.md` under the new version and date.
4. If the geometry of any reference case changed on purpose, regenerate the
   reference outputs and say why in the changelog:
   `python tools/make_reference.py`.
5. Commit, then tag and push:
   ```bash
   git tag -a v1.1.0 -m "PorousGen 1.1.0"
   git push origin main --tags
   ```
6. GitHub -> *Releases* -> *Draft a new release* -> choose tag `v1.1.0`,
   paste the changelog section, publish. Zenodo archives it within minutes;
   copy the version DOI into the article (Code metadata table, "Permanent
   link to reproducible capsule / archived version").
7. Record the benchmark for the release on the reference machine:
   `python benchmarks/benchmark.py --sizes 64 128 256 512 --out benchmarks/results_v1.1.0`
   and commit the `.md`, `.csv` and `_system.json` files.

The tag, the Zenodo DOI and the version printed by `porousgen --version` and
written into every `_info.txt` must all agree.
