import random
from collections import Counter
from pathlib import Path

def audit_wfdb_headers(data_dir: str | Path, sample_size: int = 500):
    dir_path = Path(data_dir)
    print(f"\nAudyt plików nagłówkowych (.hea) w: {dir_path.name}")

    all_headers = list(dir_path.rglob("*.hea"))
    print(f"Znaleziono łącznie: {len(all_headers)} plików .hea")

    if not all_headers:
        print("Brak plików do analizy.")
        return

    sample_headers = random.sample(all_headers, min(sample_size, len(all_headers)))
    comments_counter = Counter()

    for header_path in sample_headers:
        with open(header_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('#'):
                    tag = line.strip().split(':')[0] if ':' in line else 'Niestandardowy komentarz'
                    comments_counter[tag] += 1

    print(f"\nPrzeanalizowano {len(sample_headers)} losowych plików.")
    print("Zidentyfikowane tagi wewnątrz plików WFDB:")

    for tag, count in comments_counter.most_common():
        print(f" - {tag.ljust(15)} : znaleziono {count} razy")

if __name__=="__main__":
    audit_wfdb_headers('../data/raw/records100', sample_size=500)