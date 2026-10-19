"""Build proofs, reject holes, and audit the axioms of every named theorem."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
LAKE = shutil.which("lake") or str(Path.home() / ".elan/bin/lake")


def run(arguments):
    result = subprocess.run([LAKE, *arguments], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(output)
    return output


def theorem_names():
    names = []
    for source in sorted((ROOT / "Temf").glob("*.lean")):
        content = source.read_text()
        if re.search(r"\b(sorry|admit|axiom|unsafe)\b", content):
            raise ValueError(f"Forbidden proof escape in {source.name}")
        names.extend("Temf." + name for name in
                     re.findall(r"^theorem (\w+)", content, re.MULTILINE))
    return names


def audit_axioms(names):
    expected = "import Temf\n\n" + "\n".join("#print axioms " + n for n in names) + "\n"
    if (ROOT / "Audit.lean").read_text() != expected:
        raise ValueError("Audit.lean must list every theorem, in source/name order")
    output = run(["env", "lean", "-DwarningAsError=true", "Audit.lean"])
    records = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", output)
    if {name for name, _ in records} != set(names) or len(records) != len(names):
        raise ValueError("Incomplete or unexpected axiom audit: " + output)
    for name, axioms in records:
        used = {item.strip() for item in axioms.split(",") if item.strip()}
        if not used <= ALLOWED:
            raise ValueError(f"Unapproved axioms for {name}: {used - ALLOWED}")
    return output


def source_hashes():
    paths = sorted(ROOT.glob("*.lean")) + sorted((ROOT / "Temf").glob("*.lean"))
    paths += [ROOT / name for name in ("lean-toolchain", "lakefile.toml", "lake-manifest.json", "check.py")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main():
    names = theorem_names()
    run(["build"])
    audit = audit_axioms(names)
    report = {"status": "passed", "theorem_count": len(names),
              "lean_version": run(["env", "lean", "--version"]).strip(),
              "allowed_axioms": sorted(ALLOWED), "source_sha256": source_hashes()}
    destination = ROOT / ".lake/verification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / ".lake/axioms.log").write_text(audit)
    print(f"PASS: {len(names)} theorems; no proof holes or extra axioms.")
    print(f"Verification record: {destination}")


if __name__ == "__main__":
    main()
