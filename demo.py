#!/usr/bin/env python3
"""
Standalone demo (no LLM calls):

Input: an IUPAC name
Outputs:
  1) XML from OPSIN (prefers mollangdata_xml, falls back to opsin_xml)
  2) The prompt text used to generate the structure descriptions
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Tuple


def _extract_json_from_process_output(stdout: str) -> dict:
    """
    OPSIN prints a single JSON object to stdout (may include other logs).
    Parse the last line that looks like JSON.
    """
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            return json.loads(ln)
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stdout[start : end + 1])
    raise ValueError("Could not find JSON object in OPSIN output.")


def run_opsin_java(jar_path: Path, iupac: str, timeout_s: int = 30) -> dict:
    java = shutil.which("java")
    if not java:
        raise RuntimeError(
            "Could not find 'java' on PATH.\n"
            "Install a JDK/JRE (e.g., Temurin / Oracle JDK) and retry."
        )

    cmd = [
        java,
        "-cp",
        str(jar_path),
        "uk.ac.cam.ch.wwmm.opsin.ProcessSingleIupacForMolLangData",
        iupac,
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if p.returncode != 0:
        stderr = (p.stderr or "").strip()
        if "Unable to locate a Java Runtime" in stderr:
            raise RuntimeError(
                "Java was invoked but macOS could not locate a Java Runtime.\n"
                "Install a JDK/JRE (e.g., Temurin / Oracle JDK) and retry.\n"
                f"Raw stderr:\n{stderr}"
            )
        raise RuntimeError(f"OPSIN failed (rc={p.returncode}). stderr:\n{stderr}")
    return _extract_json_from_process_output(p.stdout)


def build_prompt_v13(prompts_dir: Path, iupac: str, smiles: str, xml_metadata: str) -> str:
    template = (prompts_dir / "default.md").read_text(encoding="utf-8")
    return (
        template.replace("{IUPAC}", iupac)
        .replace("{SMILES}", smiles)
        .replace("{XML_METADATA}", xml_metadata)
    )


def _find_default_jar(script_dir: Path) -> Path:
    jars = sorted(script_dir.glob("opsin-core-*-jar-with-dependencies.jar"))
    if not jars:
        raise FileNotFoundError("Could not find opsin-core-*-jar-with-dependencies.jar in this folder.")
    return jars[-1]


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Standalone demo: IUPAC -> XML + prompt (no LLM calls).")
    ap.add_argument("iupac", type=str, help="IUPAC name")
    ap.add_argument("--opsin-jar", type=str, default=None, help="Path to OPSIN jar (defaults to jar in this folder)")
    ap.add_argument("--timeout-s", type=int, default=30, help="Timeout for OPSIN call (seconds)")
    args = ap.parse_args()

    jar_path = Path(args.opsin_jar) if args.opsin_jar else _find_default_jar(script_dir)
    prompts_dir = script_dir / "prompts" / "smiles_iupac_metadata_v13"

    opsin_json = run_opsin_java(jar_path, args.iupac, timeout_s=args.timeout_s)

    opsin_smiles = opsin_json.get("opsin_smiles") or ""
    mollangdata_smiles = opsin_json.get("mollangdata_smiles") or ""
    opsin_xml = opsin_json.get("opsin_xml") or ""
    mollangdata_xml = opsin_json.get("mollangdata_xml") or ""

    smiles = mollangdata_smiles or opsin_smiles
    xml_metadata = mollangdata_xml or opsin_xml

    if not smiles:
        raise SystemExit("OPSIN did not return any SMILES.")
    if not xml_metadata:
        raise SystemExit("OPSIN did not return any XML.")

    prompt = build_prompt_v13(prompts_dir, args.iupac, smiles, xml_metadata)
    print("=== IUPAC ===")
    print(args.iupac)
    print()

    print("=== SMILES (used) ===")
    print(smiles)
    print()

    print("=== PROMPT ===")
    print(prompt)
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Keep it demo-friendly: print a readable error and exit 1
        print(str(e))
        raise SystemExit(1)
