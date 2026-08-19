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

`prepare` and `submit` are convenience wrappers with `--from` / `--through`. In Phase 2, both submit Chai through LSF by default. `prepare --local` is the explicit direct-execution escape hatch for development. `status` reports stage checkpoints and recorded LSF job IDs; analysis remains deferred to Phase 16.

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

## Phase 5: coordinate transfer and glycan hydrogen repair

After the Phase 4 mapping is validated, run the local deterministic coordinate-transfer stage with:

```bash
lyso-md prepare /path/to/workspace/config.yaml --from coordinates --through coordinates
```

Phase 5 starts from the authoritative GLYCAM-Web `structure.off`. Every mapped GLYCAM heavy-atom coordinate is replaced by its corresponding Chai heavy-atom coordinate; atom names, atom types, charges, residue identities, connectivity, and all other OFF metadata are preserved. Hydrogens are then moved using parent-centered local heavy-atom frames. Carbon centers with exactly three heavy neighbors and one hydrogen are rebuilt explicitly in the missing tetrahedral direction while retaining the original GLYCAM C-H bond length. This special reconstruction is not applied to methyl, hydroxyl, amide, or other hydrogen environments.

Outputs are written under `02_prepare/coordinate_transfer/`:

```text
glycan_aligned.off
atom_mapping.tsv
tetrahedral_hydrogen_repairs.tsv
hydrogen_validation.json
.done
```

The stage fails closed if the Phase 4 mapping is incomplete, an OFF hydrogen has invalid connectivity, any coordinate becomes non-finite, the aligned OFF changes topology/parameters, a mapped heavy coordinate is not preserved on serialization, or an X-H bond length changes beyond tolerance.


## Phase 6: protein preparation and disulfide detection

Run the protein-only preparation stage with:

```bash
lyso-md prepare /path/to/workspace/config.yaml --from protein --through protein
```

Phase 6 extracts only protein `ATOM` records from the validated Chai PDB, removes hydrogens, preserves protein residue numbering, detects candidate disulfides from CYS SG-SG distances using the pipeline default cutoff of 2.4 Å, renames participating CYS residues to `CYX`, and emits the corresponding LEaP `bond` commands. Disulfide residue numbers are discovered from coordinates and are never hard-coded. The stage fails closed if the protein residue count is wrong, a CYS lacks SG, residue numbers are ambiguous for LEaP addressing, or a CYS SG is within the cutoff of multiple partners.

Outputs are written under `02_prepare/protein/`:

```text
protein_chai.pdb
disulfides.tsv
disulfide_bonds.leap
validation.json
.done
```

`protein_chai.pdb` contains no hydrogen or ligand records. The audit TSV records chain/residue identifiers and SG-SG distances, while `disulfide_bonds.leap` contains commands such as `bond protein.6.SG protein.128.SG`. The `.done` sentinel is written only after all validation checks pass.
## Phase 7: dry LEaP complex assembly

After the GLYCAM inspection, mapping, coordinate transfer, and protein preparation stages have completed, assemble the dry protein/glycan complex with Amber LEaP:

```bash
lyso-md prepare /path/to/workspace/config.yaml --from leap --through leap --dry-run
lyso-md prepare /path/to/workspace/config.yaml --from leap --through leap
```

Phase 7 is a local Amber stage and is not submitted through LSF. Amber must be available in `PATH`, normally after `module load amber/22_rhel8`. The generated `03_dry_relax/leap.in` sources `ff19SB` and `GLYCAM_06j-1`, loads the two GLYCAM-Web frcmod files, loads the aligned GLYCAM OFF unit and prepared protein, applies the detected disulfide bonds, checks/charges the assembled complex, and writes `complex_dry.pdb`, `complex_dry.parm7`, and `complex_dry.rst7`.

LEaP output is parsed fail-closed for fatal/unknown-residue/untypeable-atom/missing-parameter errors, non-integral total charge, missing outputs, inconsistent PDB/parm7/restart atom counts, and non-finite coordinates. `03_dry_relax/.done` is written only after all checks pass.

## Phase 8: dry hydrogen relaxation

After successful dry LEaP assembly, run the CPU-only hydrogen relaxation with:

```bash
module load amber/22_rhel8
lyso-md prepare /path/to/workspace/config.yaml --from hydrogen-relax --through hydrogen-relax
```

This stage deliberately uses the CPU `pmemd` executable rather than `pmemd.cuda`. It starts from `complex_dry.rst7`, uses that same restart as the positional-restraint reference, restrains non-hydrogen atoms at 100 kcal/mol/A^2, and performs the configured 1000-step nonperiodic minimization. The generated input uses the fixed Phase 8 protocol:

```text
imin=1, maxcyc=1000, ncyc=500, ntb=0, igb=0, cut=1000.0,
ntr=1, restraint_wt=100.0, restraintmask='!@H=', ntpr=50
```

Outputs are written under `03_dry_relax/hydrogen_relax/`:

```text
hrelax.in
hrelax.out
complex_hrelaxed.rst7
validation.json
.done
```

Completion is fail-closed: the output must contain a normal-completion marker and finite final energy/RMS/GMAX values, the restart must exist and contain finite coordinates, and its atom count must agree with the Phase 7 topology. A nonzero `pmemd` exit status is retained as a warning if Amber nevertheless reports normal completion and all scientific output checks pass; otherwise the stage fails and does not write `.done`.

## Phase 9: OPC solvation and KCl ionization

After Phase 8 hydrogen relaxation, run the solvation/ionization stage with:

```bash
module load amber/22_rhel8
lyso-md prepare /path/to/workspace/config.yaml --from solvate --through solvate
```

Phase 9 deliberately does **not** reload an Amber topology/restart into LEaP. Amber 22 LEaP does not provide the assumed `loadAmberParm` operation needed for that architecture, and reloading the dry complex PDB also loses the authoritative GLYCAM-Web residue templates. Instead, the final ionized construction is a fresh LEaP session that reconstructs the typed protein + `CONDENSEDSEQUENCE` complex, applies the detected disulfides, runs `solvateOct`, adds the calculated KCl with `addIonsRand`, and writes the final topology/restart/PDB with the canonical Amber command spellings `saveAmberParm` and `savePdb`.

Because the exact `solvateOct` box is produced by LEaP itself, Phase 9 first performs a **read-only LEaP geometry probe** using the same authoritative typed sources. The probe's periodic box supplies the exact volume for the configured 50 mM KCl calculation. The final ionized LEaP session then repeats the typed construction from source files; it never loads the probe topology. This two-invocation design is intentional: it preserves exact Amber box geometry without guessing or reimplementing `solvateOct`.

The final construction uses commands of the form:

```text
solvateOct complex OPCBOX 12.0
addIonsRand complex K+ <N_K>
addIonsRand complex Cl- <N_Cl>
saveAmberParm complex complex_solvated.parm7 complex_solvated.rst7
savePdb complex complex_solvated.pdb
```

After LEaP finishes, Python replaces only the first dry-solute atoms in the solvated restart/PDB with the coordinates from `complex_hrelaxed.rst7`, preserving water, ions, and periodic box vectors. Final validation checks atom counts, box preservation, K+/Cl- counts, neutral charge, finite coordinates, and coordinate transfer before writing `04_solvate/.done`.

The stage records both LEaP logs and inputs for provenance:

```text
04_solvate/
├── solvate_probe.in
├── solvate_probe.log
├── solvated_probe.parm7
├── solvated_probe.rst7
├── solvated_probe.pdb
├── solvate_ionize.in
├── solvate.log
├── complex_solvated.parm7
├── complex_solvated.rst7
├── complex_solvated.pdb
├── validation.json
└── .done
```

## Phase 10: periodic GPU minimization

Phase 10 submits two dependent LSF GPU jobs after the integrated Phase 9 solvation/ionization checkpoint:

1. 5000-step periodic minimization with 10 kcal/mol/A^2 restraints on non-water/non-ion heavy solute atoms.
2. 5000-step whole-system minimization with 5 kcal/mol/A^2 restraints using the first minimization restart as both the starting structure and positional-restraint reference.

Both stages use `pmemd.cuda`, `ntb=1`, and the configured 9 A cutoff. The restraint mask is:

```text
(!:WAT,K+,Cl-)&(!@H=)
```

The jobs are chained with LSF `done(JOBID)` dependency. Scheduler stdout/stderr are written under `logs/minimize_solvent.%J.*` and `logs/minimize_all.%J.*`. Each job validates its own restart, finite coordinates, normal completion, and absence of CUDA/NaN/SHAKE/vlimit failures. The overall `05_minimize/.done` sentinel is written only by the successful second-stage worker.

Use:

```bash
lyso-md prepare /path/to/workspace/config.yaml --from minimize --through minimize --dry-run
lyso-md prepare /path/to/workspace/config.yaml --from minimize --through minimize
```

## Phase 11: restrained NVT heating

After the Phase 10 all-system minimization checkpoint, submit the restrained NVT heating stage with:

```bash
module load amber/22_rhel8
lyso-md prepare /path/to/workspace/config.yaml --from heat --through heat --dry-run
lyso-md prepare /path/to/workspace/config.yaml --from heat --through heat
```

Phase 11 is a GPU LSF job using `pmemd.cuda`. It starts from the Phase 10 all-system minimization restart and uses that same restart as the positional-restraint reference. The fixed protocol heats fresh velocities from 10 K to 300 K over the configured 100 ps, with a 2 fs timestep, Langevin thermostat (`ntt=3`, `gamma_ln=5.0`), SHAKE on bonds to hydrogen, 5 kcal/mol/A^2 heavy-solute restraints, NVT periodic dynamics, and `iwrap=0`.

The generated Amber input is equivalent to:

```text
irest=0, ntx=1, nstlim=50000, dt=0.002,
ntb=1, ntp=0, cut=9.0,
ntt=3, gamma_ln=5.0, tempi=10.0, temp0=300.0,
ntr=1, restraint_wt=5.0,
restraintmask='(!:WAT,K+,Cl-)&(!@H=)',
ntc=2, ntf=2, iwrap=0
```

The stage is submitted with an LSF `done(JOBID)` dependency on the Phase 10 all-system minimization when that job is still running. If Phase 10 already has its `.done` checkpoint, no scheduler dependency is needed.

Outputs are written under `06_equilibrate/heat/`:

```text
heat.in
heat.out
heat.rst7
heat.nc
validation.json
.done
```

Validation requires normal Amber completion, a finite final temperature in the 250-350 K range, finite restart coordinates, matching topology/restart atom counts, and absence of CUDA, SHAKE, NaN/Inf, vlimit, or fatal diagnostics. `.done` is written only after all checks pass.

## Phase 12 — conservative NPT smoke test

After successful restrained NVT heating, submit the short NPT smoke test with:

```bash
lyso-md prepare design_042/config.yaml --from npt-smoke --through npt-smoke
```

The stage is a 5 ps conservative NPT smoke test using a 1 fs timestep, Berendsen barostat, `taup=5.0`, fresh 300 K velocities, 5 kcal/mol/A^2 heavy-solute restraints, and `iwrap=0`. The stage-start heating restart is used as both the coordinate input and restraint reference, preventing periodic-image restraint artifacts.

The LSF job depends on the Phase 11 heating job when that job is still running. Validation requires normal Amber completion, finite temperature/density, a positive density, a reasonable step-0 restraint energy (below the pipeline smoke-test threshold), a valid restart, matching topology atom counts, and absence of explicit CUDA/SHAKE/vlimit/NaN/Inf failures. The `.done` sentinel is written only after validation succeeds.

## Phase 13 — staged NPT equilibration

After the Phase 12 NPT smoke-test checkpoint, submit three dependent GPU LSF jobs:

1. 250 ps at 5 kcal/mol/A^2 heavy-solute restraints.
2. 250 ps at 1 kcal/mol/A^2 heavy-solute restraints.
3. 500 ps unrestrained.

All stages use `pmemd.cuda`, a 2 fs timestep, NPT (`ntb=2`, `ntp=1`, `barostat=1`, `taup=5.0`), Langevin temperature control at the configured temperature, `iwrap=0`, and SHAKE on hydrogen bonds. The two restrained stages use the current stage-start restart as both `-c` and `-ref`; the free stage has `ntr=0` and no restraint mask.

Use:

```bash
module load amber/22_rhel8
lyso-md prepare /path/to/workspace/config.yaml --from npt-equilibrate --through npt-equilibrate --dry-run
lyso-md prepare /path/to/workspace/config.yaml --from npt-equilibrate --through npt-equilibrate
```

The three jobs are chained with `done(JOBID)` dependencies. Each stage has its own `validation.json` and `.done` checkpoint under `06_equilibrate/npt_equilibrate/{restraint5,restraint1,free}/`. The aggregate `06_equilibrate/npt_equilibrate/.done` sentinel is written only by the successful unrestrained stage.

Validation is section-aware and requires normal Amber completion, finite temperature/density, positive density, finite restart coordinates, matching topology/restart atom counts, and absence of explicit CUDA/SHAKE/vlimit/NaN/Inf/fatal diagnostics. No density-equilibrium threshold is imposed at this stage; the purpose is to establish the final equilibrated checkpoint before production.

## Phase 14 — Chunked production MD

Production is restartable and split into configurable chunks (`production.chunk_ns`) rather than assuming a single long scheduler allocation. Each invocation of `lyso-md submit --from production --through production` determines the latest contiguous completed chunk, calculates remaining target time, prepares the next chunk, and submits one LSF GPU job. A later invocation resumes from that chunk's validated restart. Each chunk is retained separately; the aggregate `07_production/.done` sentinel is created only when the configured target duration has been reached.

Production uses `pmemd.cuda` with restart chaining (`irest=1`, `ntx=5`), 2 fs timestep, NPT/Langevin at the configured temperature and pressure, no restraints, and `iwrap=0`. Chunk validation records cumulative simulation time, observables, output hashes, and normal-completion/failure checks.


## Phase 15 — LSF orchestration

The pipeline provides an explicit LSF status view with `lyso-md status CONFIG`. It reports each implemented stage, its `.done` checkpoint, and any recorded LSF job IDs. Scheduler-specific stage modules retain their dependency expressions and required-file checks; the top-level `submit` command dispatches the appropriate LSF stage submission.

Phase 15 is scheduler orchestration. The standard CPPTRAJ analysis suite is Phase 16.
