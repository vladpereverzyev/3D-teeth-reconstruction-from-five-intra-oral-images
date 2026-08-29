"""
reconstruct_api.py - CLI wrapper around the 5-view teeth reconstruction pipeline.

Runs as an isolated subprocess (invoked by webui/server.py) so the heavy legacy
stack (TensorFlow 2.6 / Ray / open3d) never shares a process with Flask.

Input : 5 intra-oral photos, one per view (upper, lower, left, right, frontal).
Output: two OBJ meshes (upper arch, lower arch) written into --out.

All progress is printed to stdout; the server redirects it to <out>/progress.log
and tails it for the UI. A final line "RECON_DONE" / "RECON_ERROR" signals the end.
"""

import argparse
import os
import sys
import time
import traceback

# The pipeline runs on CPU (see main.py). Force it here too, before importing TF.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(msg, flush=True)


def reconstruct(image_paths, out_dir, num_cpus=4):
    """
    image_paths: dict {"UPPER": path, "LOWER": path, "LEFT": path,
                       "RIGHT": path, "FRONTAL": path}
    out_dir    : directory to write upper.obj / lower.obj into
    """
    os.chdir(REPO_DIR)  # pipeline uses relative paths (./ssm, ./seg, ...)
    sys.path.insert(0, REPO_DIR)

    import numpy as np
    import ray

    from const import (
        SSM_DIR, REGIS_PARAM_DIR, NUM_PC,
        PHOTO_TYPES, VISIBLE_MASKS, RECONS_IMG_WIDTH,
    )
    from emopt5views import EMOpt5Views
    from seg.seg_const import IMG_SHAPE
    from seg.seg_model import ASPP_UNet
    from seg.utils import predict_teeth_contour
    # Reuse the exact optimization + meshing logic from main.py
    from main import loadMuEigValSigma, run_emopt, create_mesh_from_emopt_h5File

    os.makedirs(out_dir, exist_ok=True)

    log("[1/5] Ray init (CPU)...")
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, num_gpus=0, include_dashboard=False,
                 ignore_reinit_error=True)

    log("[2/5] Loading statistical shape model...")
    Mu, SqrtEigVals, Sigma = loadMuEigValSigma(SSM_DIR, numPC=NUM_PC)
    Mu_normals = EMOpt5Views.computePointNormals(Mu)

    transVecStd = 1.1463183505325343   # obtained by SSM (from main.py)
    rotVecStd = 0.13909168140778128
    PoseCovMats = np.load(os.path.join(REGIS_PARAM_DIR, "PoseCovMats.npy"))
    ScaleCovMat = np.load(os.path.join(REGIS_PARAM_DIR, "ScaleCovMat.npy"))

    # Assume every tooth is present (28 = 14 upper + 14 lower, wisdom teeth ignored).
    tooth_exist_mask = np.ones((28,), np.bool_)

    log("[3/5] Teeth-boundary segmentation (U-Net) on 5 photos...")
    weight_ckpt = os.path.join(REPO_DIR, "seg", "weights",
                               "weights-teeth-boundary-model.h5")
    model = ASPP_UNet(IMG_SHAPE, filters=[16, 32, 64, 128, 256])
    model.load_weights(weight_ckpt)

    edgeMasks = []
    for phtype in PHOTO_TYPES:
        imgfile = image_paths[phtype.name]
        log("      - {}: {}".format(phtype.name, os.path.basename(imgfile)))
        edge_mask = predict_teeth_contour(model, imgfile,
                                          resized_width=RECONS_IMG_WIDTH)
        edgeMasks.append(edge_mask)

    log("[4/5] EM optimization (this is the slow part on CPU)...")
    emopt = EMOpt5Views(
        edgeMasks, PHOTO_TYPES, VISIBLE_MASKS, tooth_exist_mask,
        Mu, Mu_normals, SqrtEigVals, Sigma,
        PoseCovMats, ScaleCovMat, transVecStd, rotVecStd,
    )
    emopt = run_emopt(emopt)

    h5File = os.path.join(out_dir, "result.h5")
    emopt.saveDemo2H5(h5File)

    log("[5/5] Building watertight meshes (Poisson) and exporting OBJ...")
    create_mesh_from_emopt_h5File(h5File, meshDir=out_dir, save_name="result")

    # create_mesh_from_emopt_h5File writes into <out_dir>/result/
    produced = os.path.join(out_dir, "result")
    upper = os.path.join(produced, "Pred_Upper_Mesh_Tag=result.obj")
    lower = os.path.join(produced, "Pred_Lower_Mesh_Tag=result.obj")
    return upper, lower


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper", required=True)
    ap.add_argument("--lower", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--frontal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_cpus", type=int, default=4)
    args = ap.parse_args()

    image_paths = {
        "UPPER": args.upper, "LOWER": args.lower, "LEFT": args.left,
        "RIGHT": args.right, "FRONTAL": args.frontal,
    }
    t0 = time.time()
    try:
        upper, lower = reconstruct(image_paths, args.out, num_cpus=args.num_cpus)
        log("UPPER_OBJ={}".format(upper))
        log("LOWER_OBJ={}".format(lower))
        log("Elapsed: {:.1f}s".format(time.time() - t0))
        log("RECON_DONE")
    except Exception:
        log(traceback.format_exc())
        log("RECON_ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
