from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProteinConfig(StrictModel):
    fasta: Path
    expected_residues: PositiveInt


class ChaiConfig(StrictModel):
    enabled: bool = True
    model_index: int = Field(default=0, ge=0)
    ligand_resname: str = Field(default="LIG", min_length=1, max_length=3)
    glycan_smiles: str = Field(min_length=1)
    command: str = "chai-lab fold"
    mamba_init: str | None = "/home/rkormos/miniforge3/etc/profile.d/mamba.sh"
    mamba_env: str | None = "env_chai"
    walltime: str = Field(default="06:00", pattern=r"^\d{1,3}:\d{2}$")


class GlycamConfig(StrictModel):
    bundle: Path
    unit_name: str = "CONDENSEDSEQUENCE"
    bacterial_frcmod: str = "frcmod.glycam06_bacterial_K3O"
    acid_frcmod: str = "frcmod.glycam06_intraring_doublebond_protonatedacids"
    expected_heavy_atoms: PositiveInt | None = 67
    expected_residues: PositiveInt | None = 5


class ForceFieldConfig(StrictModel):
    protein: str = "ff19SB"
    glycan: str = "GLYCAM_06j-1"
    water: str = "OPC"


class SolventConfig(StrictModel):
    buffer_angstrom: PositiveFloat = 12.0
    salt: Literal["KCl"] = "KCl"
    concentration_molar: float = Field(default=0.05, ge=0.0)


class MDConfig(StrictModel):
    temperature_k: PositiveFloat = 300.0
    pressure_bar: PositiveFloat = 1.0
    cutoff_angstrom: PositiveFloat = 9.0
    production_timestep_fs: PositiveFloat = 2.0


class EquilibrationConfig(StrictModel):
    hydrogen_relax_steps: PositiveInt = 1000
    solvent_min_steps: PositiveInt = 5000
    all_min_steps: PositiveInt = 5000
    heat_ps: PositiveFloat = 100
    npt_5_ps: PositiveFloat = 250
    npt_1_ps: PositiveFloat = 250
    npt_free_ps: PositiveFloat = 500


class ProductionConfig(StrictModel):
    target_ns: PositiveFloat = 1000
    chunk_ns: PositiveFloat = 250
    walltime_hours: PositiveFloat = 72

    @model_validator(mode="after")
    def chunk_not_larger_than_target(self) -> "ProductionConfig":
        if self.chunk_ns > self.target_ns:
            raise ValueError("production.chunk_ns must not exceed production.target_ns")
        return self


class SchedulerConfig(StrictModel):
    type: Literal["lsf"] = "lsf"
    project: str = Field(default="lysozyme_C_md", min_length=1)
    gpu_queue: str = Field(default="gpu", min_length=1)
    gpu_resource: str = Field(default="num=1/host", min_length=1)
    memory: str = Field(default="16GB", min_length=1)
    cores: PositiveInt = 1


class PipelineConfig(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    protein: ProteinConfig
    chai: ChaiConfig
    glycam: GlycamConfig
    forcefield: ForceFieldConfig
    solvent: SolventConfig
    md: MDConfig
    equilibration: EquilibrationConfig
    production: ProductionConfig
    scheduler: SchedulerConfig

    @field_validator("name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        if value in {".", ".."}:
            raise ValueError("name must be a safe directory name")
        return value


def _absolutize_input_paths(raw: dict, config_dir: Path) -> dict:
    """Resolve only true source-file fields relative to the source config."""
    data = dict(raw)
    protein = dict(data.get("protein", {}))
    glycam = dict(data.get("glycam", {}))
    if "fasta" in protein:
        p = Path(protein["fasta"]).expanduser()
        protein["fasta"] = p if p.is_absolute() else (config_dir / p).resolve()
    if "bundle" in glycam:
        p = Path(glycam["bundle"]).expanduser()
        glycam["bundle"] = p if p.is_absolute() else (config_dir / p).resolve()
    data["protein"] = protein
    data["glycam"] = glycam
    return data


def load_config(path: str | Path, *, check_files: bool = True) -> PipelineConfig:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must contain a YAML mapping: {path}")
    cfg = PipelineConfig.model_validate(_absolutize_input_paths(raw, path.parent))
    if check_files:
        missing: list[str] = []
        if not cfg.protein.fasta.is_file():
            missing.append(f"protein.fasta: {cfg.protein.fasta}")
        if not cfg.glycam.bundle.is_file():
            missing.append(f"glycam.bundle: {cfg.glycam.bundle}")
        if missing:
            raise ValueError("missing required input file(s): " + "; ".join(missing))
    return cfg


def config_as_yaml_dict(cfg: PipelineConfig) -> dict:
    return cfg.model_dump(mode="json")
