#!/usr/bin/env bash
# Packages the assembled manuscript and supplement as a flat Overleaf project: both .tex files, the
# bibliography, the Springer class and style, and every image they include. Run after
# run_final_pipeline.sh has passed; writes overleaf_package/ and TabStress_Overleaf_Bundle.zip.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=springer_project
OUT=overleaf_package
ZIP=TabStress_Overleaf_Bundle.zip

rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"
cp "$SRC/TabStress.tex" "$SRC/TabStress_Supplementary.tex" "$SRC/TabStress.bib" \
   "$SRC/sn-jnl.cls" "$SRC/sn-mathphys.bst" "$OUT/"

cat "$SRC/TabStress.tex" "$SRC/TabStress_Supplementary.tex" \
  | grep -o 'includegraphics\[[^]]*\]{[^}]*}' | sed 's/.*{\(.*\)}/\1/' | sort -u \
  | while read -r img; do
      if [[ -f "$SRC/figures/$img" ]]; then cp "$SRC/figures/$img" "$OUT/"
      elif [[ -f "$SRC/$img" ]]; then cp "$SRC/$img" "$OUT/"
      else echo "missing image: $img" >&2; exit 1; fi
    done

(cd "$OUT" && zip -q -r "../$ZIP" .)
echo "$OUT: $(ls "$OUT" | wc -l | tr -d ' ') files; $ZIP: $(du -h "$ZIP" | cut -f1)"
