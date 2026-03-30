"""
download_noises.py
소음 DB 다운로드 스크립트

지원 DB:
  1. DEMAND   — 16kHz 잡음 녹음 (방/교통 등) - 공개 도메인
  2. MUSAN    — music / speech / noise 혼합
  3. WHAM!    — 혼합 환경 소음 (wham_noise subset)

사용법:
  python data/download_noises.py --db demand --output data/raw
  python data/download_noises.py --db all --output data/raw
"""

import os
import sys
import argparse
import shutil
import tarfile
import zipfile
import urllib.request
from pathlib import Path


# ─────────────────────────────────────────────
# 다운로드 유틸
# ─────────────────────────────────────────────

def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """URL → 로컬 파일 다운로드 (진행률 표시)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'다운로드: {url}')
    print(f'  → {dest}')

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    sys.stdout.write(f'\r  {downloaded/1024/1024:.1f} / {total/1024/1024:.1f} MB ({pct:.1f}%)')
                    sys.stdout.flush()
    print()


def extract_archive(archive_path: Path, extract_to: Path):
    """tar.gz 또는 zip 압축 해제."""
    extract_to.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    print(f'압축 해제: {archive_path} → {extract_to}')
    if name.endswith('.tar.gz') or name.endswith('.tgz') or name.endswith('.tar.bz2'):
        with tarfile.open(archive_path) as tar:
            tar.extractall(extract_to)
    elif name.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_to)
    else:
        print(f'  알 수 없는 포맷: {name} — 수동 해제 필요')


# ─────────────────────────────────────────────
# DEMAND
# ─────────────────────────────────────────────

DEMAND_URLS = {
    # Zenodo - 16kHz 버전 (일부 환경만)
    # 전체: https://zenodo.org/record/1227121
    'DKITCHEN':  'https://zenodo.org/record/1227121/files/DKITCHEN_16k.zip',
    'DLIVING':   'https://zenodo.org/record/1227121/files/DLIVING_16k.zip',
    'NPARK':     'https://zenodo.org/record/1227121/files/NPARK_16k.zip',
    'OOFFICE':   'https://zenodo.org/record/1227121/files/OOFFICE_16k.zip',
    'PCAFETER':  'https://zenodo.org/record/1227121/files/PCAFETER_16k.zip',
    'TBUS':      'https://zenodo.org/record/1227121/files/TBUS_16k.zip',
    'TCAR':      'https://zenodo.org/record/1227121/files/TCAR_16k.zip',
    'TMETRO':    'https://zenodo.org/record/1227121/files/TMETRO_16k.zip',
}

DEMAND_SUBSET = ['DKITCHEN', 'OOFFICE', 'TBUS', 'TCAR']  # 기본 다운로드 subset


def download_demand(output_dir: Path, subset=None, skip_existing=True):
    """DEMAND 소음 DB 다운로드 (기본: DEMAND_SUBSET)."""
    subset = subset or DEMAND_SUBSET
    demand_dir = output_dir / 'DEMAND'

    for env_name in subset:
        if env_name not in DEMAND_URLS:
            print(f'  DEMAND [{env_name}] URL 없음 — 스킵')
            continue

        env_dir = demand_dir / env_name
        if skip_existing and env_dir.exists():
            print(f'  DEMAND [{env_name}] 이미 존재 — 스킵')
            continue

        url = DEMAND_URLS[env_name]
        archive = output_dir / f'DEMAND_{env_name}.zip'
        try:
            download_file(url, archive)
            extract_archive(archive, demand_dir)
            archive.unlink()
        except Exception as e:
            print(f'  DEMAND [{env_name}] 다운로드 실패: {e}')

    print(f'DEMAND 다운로드 완료: {demand_dir}')
    return str(demand_dir)


# ─────────────────────────────────────────────
# MUSAN
# ─────────────────────────────────────────────

MUSAN_URL = 'http://www.openslr.org/resources/17/musan.tar.gz'


def download_musan(output_dir: Path, skip_existing=True):
    """MUSAN 다운로드 (noise subset만 사용 권장)."""
    musan_dir = output_dir / 'MUSAN'
    if skip_existing and (musan_dir / 'noise').exists():
        print(f'MUSAN 이미 존재 — 스킵: {musan_dir}')
        return str(musan_dir)

    archive = output_dir / 'musan.tar.gz'
    try:
        download_file(MUSAN_URL, archive)
        extract_archive(archive, output_dir)
        # tar 내부 구조: musan/ → output_dir/musan
        extracted = output_dir / 'musan'
        if extracted.exists() and not musan_dir.exists():
            extracted.rename(musan_dir)
        if archive.exists():
            archive.unlink()
    except Exception as e:
        print(f'MUSAN 다운로드 실패: {e}')
        print('  수동 다운로드: http://www.openslr.org/17/')

    print(f'MUSAN 다운로드 완료: {musan_dir}')
    return str(musan_dir)


# ─────────────────────────────────────────────
# WHAM! (wham_noise)
# ─────────────────────────────────────────────

WHAM_URL = 'https://my-bucket.s3.amazonaws.com/wham_noise.zip'
# 실제 URL 확인: http://wham.whisper.ai/
WHAM_OFFICIAL_URL = 'http://wham.whisper.ai/'


def download_wham(output_dir: Path, skip_existing=True):
    """WHAM! noise subset 다운로드 시도. 자동 다운로드 불가 시 안내."""
    wham_dir = output_dir / 'WHAM'
    if skip_existing and wham_dir.exists():
        print(f'WHAM 이미 존재 — 스킵: {wham_dir}')
        return str(wham_dir)

    print('\n[WHAM!] 자동 다운로드를 위해 아래 URL에서 수동 다운로드 필요:')
    print(f'  {WHAM_OFFICIAL_URL}')
    print(f'  다운로드 후 압축 해제 → {wham_dir} 에 배치')
    wham_dir.mkdir(parents=True, exist_ok=True)
    return str(wham_dir)


# ─────────────────────────────────────────────
# LibriSpeech
# ─────────────────────────────────────────────

LIBRISPEECH_URL = 'https://www.openslr.org/resources/12/train-clean-100.tar.gz'


def download_librispeech(output_dir: Path, skip_existing=True):
    """LibriSpeech train-clean-100 다운로드 (~6GB)."""
    libri_dir = output_dir / 'LibriSpeech'
    subset_dir = libri_dir / 'train-clean-100'

    if skip_existing and subset_dir.exists():
        print(f'LibriSpeech train-clean-100 이미 존재 — 스킵: {subset_dir}')
        return str(libri_dir)

    print('[LibriSpeech] train-clean-100 다운로드 시작 (~6.3GB, 시간 소요)')
    archive = output_dir / 'train-clean-100.tar.gz'
    try:
        download_file(LIBRISPEECH_URL, archive)
        extract_archive(archive, output_dir)
        if archive.exists():
            archive.unlink()
    except Exception as e:
        print(f'LibriSpeech 다운로드 실패: {e}')
        print('  수동 다운로드: https://www.openslr.org/12/')

    print(f'LibriSpeech 다운로드 완료: {libri_dir}')
    return str(libri_dir)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='소음 DB 다운로드')
    parser.add_argument(
        '--db',
        choices=['demand', 'musan', 'wham', 'librispeech', 'all'],
        nargs='+',
        default=['all'],
        help='다운로드할 DB (복수 지정 가능)',
    )
    parser.add_argument('--output', default='data/raw', help='저장 디렉토리')
    parser.add_argument('--demand_subset', nargs='+', default=DEMAND_SUBSET,
                        help='DEMAND 환경 목록')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='이미 있으면 스킵')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dbs = args.db
    if 'all' in dbs:
        dbs = ['librispeech', 'demand', 'musan', 'wham']

    for db in dbs:
        print(f'\n{"="*50}')
        if db == 'demand':
            download_demand(output_dir, subset=args.demand_subset,
                            skip_existing=args.skip_existing)
        elif db == 'musan':
            download_musan(output_dir, skip_existing=args.skip_existing)
        elif db == 'wham':
            download_wham(output_dir, skip_existing=args.skip_existing)
        elif db == 'librispeech':
            download_librispeech(output_dir, skip_existing=args.skip_existing)

    print('\n=== 소음 DB 다운로드 완료 ===')
    print(f'저장 위치: {output_dir.resolve()}')
    print('\n다음 단계:')
    print('  python -m data.pipeline --env both --split all')


if __name__ == '__main__':
    main()
