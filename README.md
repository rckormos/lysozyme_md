# lyso-md

`lyso-md` is a restartable FASTA-to-MD pipeline for a narrow family of human lysozyme C designs bound to a fixed peptidoglycan ligand. This repository currently implements **Phase 0 (project/config/CLI scaffolding)** and **Phase 1 (external design-workspace initialization)**. Molecular transformations and HPC execution are intentionally stubbed until later phases.

## Design principles

- Source code, templates, tests, examples, and documentation live in Git.
- Per-design scientific workspaces live **outside** the repository.
- Original FASTA and shared GLYCAM-Web ZIP inputs are treated as immutable and symlinked into each design workspace.
- Every initialized workspace records SHA256 hashes and normalized configuration in `manifest.json`.
- Stage completion sentinels are JSON metadata files named `.done`; Phase 1 writes `.lyso-md/init/.done`.
- `--force` never deletes an existing workspace: it renames it to a timestamped sibling backup before creating a new one.
- Amber 22 and CPPTRAJ are external cluster dependencies expected on `$PATH` after `module load amber/22_rhel8`.
- Chai-1 will later be invoked after:

  ```bash
  source /home/rkormos/miniforge3/etc/profile.d/mamba.sh
  mamba activate env_chai
  chai-lab fold INPUT_FASTA OUTPUT_DIR
  ```

- LSF is the only intended scheduler.

## Install on the HPC

Create the Python environment from the repository root:

```bash
mamba env create -f environment.yml
mamba activate lyso-md
```

`environment.yml` includes the Python dependencies planned for later molecular-structure phases (`rdkit`, `networkx`, `parmed`, NumPy/SciPy), but it does **not** install Amber or Chai.

For development/testing:

```bash
pip install -e .
pytest
```

## Validate the bundled example configuration

The repository includes a tiny synthetic GLYCAM ZIP whose only purpose is to satisfy Phase 0/1 file-existence validation. It is **not scientifically usable**.

```bash
lyso-md validate-config config/example.yaml
```

For real work, point `glycam.bundle` to the original shared GLYCAM-Web ZIP.

## Configuration

The canonical schema is represented by `config/example.yaml`. Important design-specific validation values are configurable, including:

- `protein.expected_residues`
- `glycam.expected_heavy_atoms`
- `glycam.expected_residues`

Production and equilibration settings are intentionally narrow defaults rather than a large general-purpose Amber configuration surface.

Input paths in a source YAML are resolved relative to that YAML file. Unknown configuration keys fail validation.

## Initialize an external design workspace

Given `/path/to/configs/design_042.yaml`:

```bash
lyso-md init /path/to/configs/design_042.yaml --workspace-root /path/to/workspaces
```

This creates:

```text
/path/to/workspaces/design_042/
├── config.yaml
├── manifest.json
├── input/
│   ├── sequence.fasta -> /absolute/source/sequence.fasta
│   ├── glycan.smiles
│   └── glycam_structure.zip -> /absolute/shared/glycam_structure.zip
├── 01_chai/
├── 02_prepare/
├── 03_dry_relax/
├── 04_solvate/
├── 05_equilibrate/
├── 06_production/
├── 07_analysis/
├── logs/
└── .lyso-md/init/.done
```

The workspace's normalized `config.yaml` references `input/sequence.fasta` and `input/glycam_structure.zip`, while `manifest.json` retains the original absolute source paths, file sizes, and SHA256 hashes.

If `--workspace-root` is omitted, the design directory is created next to the source configuration file. Keep source design YAMLs outside the source repository when using this default.

### Safe reinitialization

Without `--force`, initialization refuses to touch an existing design directory.

With `--force`, an existing directory is preserved, for example:

```text
design_042.backup-20260811T210000Z/
```

and a fresh `design_042/` is created. Existing results are never silently removed.

## CLI in Phases 0-1

Implemented:

```bash
lyso-md init CONFIG [--workspace-root DIR] [--force]
lyso-md validate-config CONFIG
```

Reserved convenience-wrapper commands (validated stubs for now):

```bash
lyso-md prepare CONFIG [--from STAGE] [--through STAGE] [--dry-run]
lyso-md submit CONFIG [--from STAGE] [--through STAGE] [--dry-run]
lyso-md status CONFIG
lyso-md analyze CONFIG
```

`prepare` and `submit` are intentionally modeled as convenience wrappers with `--from` / `--through`; later phases will fill in their stage graph.

## LSF assumptions for later phases

The configuration exposes the stable cluster knobs (`project`, GPU queue/resource, memory, cores, and per-production-chunk walltime). Job names and stdout/stderr paths will be generated from design/stage names under the design's `logs/` directory. The implementation will target LSF directly rather than introducing a generic scheduler abstraction.

## Regression assets

Real high-fidelity regression assets should remain outside Git and can later be supplied through a path such as:

```bash
export LYSO_MD_REGRESSION_DATA=/path/to/asset_bundle_final
```

Normal Phase 0/1 tests require neither Amber, Chai, nor those real scientific assets.
