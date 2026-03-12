#!/usr/bin/env bash
# rename_sdfvd_fake.sh
# Przemianowuje pliki fake z datasetu SDFVD dodajac sufiks _c2pa przed rozszerzeniem.

FAKE_DIR="/mnt/c/Users/bart1/OneDrive/Desktop/magisterka/nagrania/SDFVD Small-scale Deepfake Forgery Video Dataset/SDFVD/SDFVD/videos_fake"

if [ ! -d "$FAKE_DIR" ]; then
    echo "[BLAD] Nie znaleziono katalogu: $FAKE_DIR"
    exit 1
fi

count=0
for f in "$FAKE_DIR"/*.mp4; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .mp4)
    if [[ "$base" != *_c2pa ]]; then
        new="$FAKE_DIR/${base}_c2pa.mp4"
        mv "$f" "$new"
        echo "Przemianowano: $(basename $f) -> ${base}_c2pa.mp4"
        ((count++))
    fi
done

echo ""
echo "Gotowe! Przemianowano $count plikow."
