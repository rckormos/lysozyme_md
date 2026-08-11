# Phase 0/1 example inputs

`sequence.fasta` is a small example FASTA. `glycam_structure_fixture.zip` is a **synthetic existence fixture only** so `lyso-md validate-config config/example.yaml` works immediately after checkout. It is not a scientifically valid GLYCAM-Web bundle and must not be used for MD preparation.

For real designs, point `glycam.bundle` at the original shared GLYCAM-Web ZIP on the cluster. `lyso-md init` will symlink it into the external design workspace.
