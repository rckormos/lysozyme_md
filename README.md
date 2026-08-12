# lyso-md

`lyso-md` is a restartable FASTA-to-MD pipeline for a narrow family of human lysozyme C designs bound to a fixed peptidoglycan ligand. This repository currently implements **Phases 0-2**, including external workspace initialization and LSF-submitted Chai-1 holo prediction. Amber/GLYCAM molecular preparation remains deferred to later phases.

## Design principles

- Source code, templates, tests, examples, and documentation live in Git.
- Per-design scientific workspaces live **outside** the repository.
- Original FASTA and shared GLYCAM-Web ZIP inputs are treated as immutable and symlinked into each design workspace.
- Every initialized workspace records SHA256 hashes and normalized configuration in `manifest.json`.
- Stage completion sentinels are JSON metadata files named `.done`; Phase 1 writes `.lyso-md/init/.done`.
- `--force` never deletes an existing workspace: it renames it to a timestamped sibling backup before creating a new one.
- Amber 22 and CPPTRAJ are external cluster dependencies expected on `$PATH` after `module load amber/22_rhel8`.
- Chai-1 is invoked inside the Phase 2 LSF worker after:

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

## CLI through Phase 2

Implemented:

```bash
lyso-md init CONFIG [--workspace-root DIR] [--force]
lyso-md validate-config CONFIG
lyso-md prepare CONFIG [--from chai] [--through chai] [--dry-run] [--local]
lyso-md submit CONFIG [--from chai] [--through chai] [--dry-run]
```

`prepare` and `submit` are convenience wrappers with `--from` / `--through`. In Phase 2, both submit Chai through LSF by default. `prepare --local` is the explicit direct-execution escape hatch for development. `status` and `analyze` remain stubs.

## LSF execution

LSF is targeted directly rather than through a generic scheduler abstraction. The configuration exposes project, GPU queue/resource, memory, and cores; `chai.walltime` controls the Chai allocation and defaults to `06:00`. Phase 2 writes `01_chai/chai.lsf`, submits it with `bsub`, records the returned job ID in `01_chai/submission.json`, and routes scheduler output to `logs/chai.%J.out` and `logs/chai.%J.err`.

## Regression assets

Real high-fidelity regression assets should remain outside Git and can later be supplied through a path such as:

```bash
export LYSO_MD_REGRESSION_DATA=/path/to/asset_bundle_final
```

Normal Phase 0/1 tests require neither Amber, Chai, nor those real scientific assets.

## Phase 2: Chai-1 holo prediction

Phase 2 is invoked from an initialized workspace. The normal path is an LSF submission:

```bash
lyso-md prepare /path/to/workspace/config.yaml --through chai --dry-run
lyso-md prepare /path/to/workspace/config.yaml --through chai
# equivalent submission entry point:
lyso-md submit /path/to/workspace/config.yaml --through chai
```

Direct execution is deliberately explicit:

```bash
lyso-md prepare /path/to/workspace/config.yaml --through chai --local
```

The stage writes `01_chai/chai_input.fasta`, `expected_command.sh`, `chai.lsf`, `submission.json`, `chai.log`, `validation.json`, the selected `pred.model_idx_N.pdb`, and `.done` only after the batch worker completes validation. Raw Chai output, including the CIF, is preserved under `01_chai/chai_output/`. LSF stdout/stderr are written under `logs/` with the actual job ID in the filename.

By default the generated shell command uses the St. Jude cluster setup supplied for this project:

```bash
source /home/rkormos/miniforge3/etc/profile.d/mamba.sh
mamba activate env_chai
chai-lab fold INPUT_FASTA OUTPUT_DIR
```

The command, mamba initialization script, environment name, and Chai walltime are configurable in the `chai:` section. Dry-run mode writes the exact command, inputs, and LSF script but does not call `bsub` and never writes `.done`.

Successful execution fails closed unless the selected model exists, the protein residue count matches configuration, the ligand is present, the ligand heavy-atom count matches the RDKit count from the configured stereospecific SMILES, all coordinates are finite, and PDB atom serials are unique.

## Phase 3: GLYCAM-Web bundle inspection

Phase 3 is a deterministic local preparation step; it does not require LSF, Amber, or Chai. After the Chai stage has completed, run:

```bash
lyso-md prepare /path/to/workspace/config.yaml --from glycam --through glycam
```

The original configured GLYCAM-Web ZIP remains untouched. Phase 3 safely extracts it under `02_prepare/glycam/extracted/`, ignoring macOS metadata entries while rejecting path traversal and ZIP symlinks. It locates `structure.off` and the two configured frcmod files by basename and fails if any required file is missing or ambiguous.

The configured OFF unit (normally `CONDENSEDSEQUENCE`) is parsed into atom names, atom types, charges, atomic numbers, residue assignments, coordinates, residue metadata, and connectivity. `02_prepare/glycam/glycam_summary.json` records this metadata plus source/output SHA256 hashes and configured count checks. `02_prepare/glycam/.done` is written only after the unit, required files, atom count, residue count, and OFF connectivity validate successfully.

Phase 3 never modifies `structure.off` or attempts to parameterize MurNAc from stock GLYCAM. The extracted GLYCAM-Web definitions remain the authoritative topology source for later phases.

## Phase 4: Chai-to-GLYCAM heavy-atom mapping

After validated Chai prediction and GLYCAM inspection, run the local deterministic mapping stage with:

```bash
lyso-md prepare /path/to/workspace/config.yaml --from mapping --through mapping
```

Phase 4 treats the configured stereospecific SMILES as the Chai ligand atom-identity graph and the GLYCAM-Web OFF unit as the authoritative topology/parameter graph. It validates Chai/SMILES/GLYCAM heavy-atom counts and element counts, requires the Chai ligand atom order to preserve the input SMILES element sequence, and performs an element-labelled NetworkX graph isomorphism against the GLYCAM heavy-atom graph. Candidate automorphisms are scored by local heavy-atom geometry. Parameter-identical graph-equivalent atoms (for example the GLYCAM carboxylate `O3A`/`O3B` pairs) are canonicalized deterministically only when they have the same residue, atom type, charge, degree, and neighbor set; ambiguity involving non-equivalent GLYCAM atoms fails closed.

Outputs are written under `02_prepare/mapping/`:

```text
atom_mapping.tsv
validation.json
.done
```

No coordinates, GLYCAM atom types, charges, residue identities, or connectivity are modified in this phase. The mapping TSV is the auditable input to Phase 5 coordinate transfer.

The synthetic GLYCAM ZIP shipped under `examples/` exists only so Phase 0/1 configuration tests have a file to validate. Phase 3 now detects that fixture explicitly and instructs the user to replace it with the real, unmodified GLYCAM-Web ZIP instead of reporting a generic missing-OFF-unit error.
