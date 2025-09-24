# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, time, joblib
import numpy as np
import rasterio
from rasterio.windows import Window
import multiprocessing
from tqdm import tqdm

SOURCE_TIF = "Citra/DoveR_2019.tif"
TARGET_TIF = "Citra/SuperDove_2024.tif"

def dir_train():   return os.path.join("Model", "Train")
def dir_out_src(): return os.path.join("Model", "Inference", "Source")
def dir_out_tgt(): return os.path.join("Model", "Inference", "Target")

PREP = "preprocessing_pipeline.pkl"
MODELS = ["T-1","T-2","T-3","T-4","E-1","E-2","E-3"]

def _pin_single_thread_blas():
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"

_G_MODEL = None
_G_PREP  = None
_G_SRC   = None

def _init_worker(model_path, preprocessor_path, in_tif_path):
    global _G_MODEL, _G_PREP, _G_SRC
    _pin_single_thread_blas()
    _G_MODEL = joblib.load(model_path, mmap_mode="r")
    _G_PREP  = joblib.load(preprocessor_path)
    _G_SRC   = rasterio.open(in_tif_path)
    if hasattr(_G_MODEL, "set_params"):
        _G_MODEL.set_params(n_jobs=1)

def _iter_windows(src, block=256, band_index=1):
    if getattr(src, "is_tiled", False):
        return [w for _, w in src.block_windows(band_index)]
    W, H = src.width, src.height
    return [Window(c, r, min(block, W-c), min(block, H-r))
            for r in range(0, H, block) for c in range(0, W, block)]

def _process_window(window):
    img = _G_SRC.read(window=window).astype(np.float32, copy=True)
    B, H, W = img.shape
    X = img.reshape(B, -1).T
    X = np.ascontiguousarray(np.nan_to_num(X, nan=0.0), dtype=np.float32)
    Xtr = _G_PREP.transform(X)
    if hasattr(_G_MODEL, "n_jobs"):
        _G_MODEL.n_jobs = 1
    y = _G_MODEL.predict(Xtr).reshape(H, W).astype(np.uint8)
    return y, window

def _run_one(model_name, model_path, prep_path, in_tif, out_tif):
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    with rasterio.open(in_tif) as src_hdr:
        windows = _iter_windows(src_hdr, block=256, band_index=1)
        profile = src_hdr.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1,
                       tiled=True, compress="lzw",
                       BIGTIFF="IF_SAFER", blockxsize=256, blockysize=256)

    n_procs   = max(1, os.cpu_count() or 1)
    chunksize = max(1, len(windows) // (n_procs * 4) or 1)

    from rasterio.env import Env
    with Env(GDAL_CACHEMAX=4096, NUM_THREADS="ALL_CPUS"):
        ctx = multiprocessing.get_context("spawn")
        t0 = time.time()
        print(f"[{model_name}] -> {os.path.basename(out_tif)}")

        with rasterio.open(out_tif, "w", **profile) as dst, \
             ctx.Pool(processes=n_procs,
                      initializer=_init_worker,
                      initargs=(model_path, prep_path, in_tif),
                      maxtasksperchild=256) as pool, \
             tqdm(total=len(windows), unit="tile", ascii=True) as pbar:

            for res, win in pool.imap_unordered(_process_window, windows, chunksize=chunksize):
                dst.write(res, window=win, indexes=1)
                pbar.update(1)

        print(f"  > Selesai. [{time.time()-t0:.2f}s]")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    _pin_single_thread_blas()
    train_dir = dir_train()
    prep_path = os.path.join(train_dir, PREP)
    for name in MODELS:
        model_path = os.path.join(train_dir, f"{name}.pkl")
        out_src = os.path.join(dir_out_src(), f"{name}_classified.tif")
        _run_one(name, model_path, prep_path, SOURCE_TIF, out_src)
        out_tgt = os.path.join(dir_out_tgt(), f"{name}_classified.tif")
        _run_one(name, model_path, prep_path, TARGET_TIF, out_tgt)
