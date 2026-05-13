import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.watermark_pipeline import (
    main,
    optimize_alpha_bayesian,
    estimate_bit_error_rate,
    choose_repeat_k_from_ber,
    choose_repeat_k_to_fill,
    choose_payload_repeat_to_fill,
    get_text_payload_bits_len,
    get_id_payload_bits_len,
    text_to_bin_image,
    id_to_bin_image,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run robust watermarking experiments on host images."
    )
    parser.add_argument("--img-dir", default="data/host_images", help="Directory containing host images.")
    parser.add_argument(
        "--wm-raw",
        default="data/watermark/cict.png",
        help="Path to watermark image when mode=image.",
    )
    parser.add_argument(
        "--mode",
        default="image",
        choices=["image", "text", "id"],
        help="Watermark mode: image, text, or id.",
    )
    parser.add_argument("--text-input", default="Hello World", help="Watermark text when mode=text.")
    parser.add_argument("--id-input", default="012345678910", help="Watermark ID when mode=id.")
    parser.add_argument("--text-encoding", default="utf-8", help="Text encoding for mode=text.")
    parser.add_argument("--repeat-k", type=int, default=3, help="Bit repetition factor (odd number).")
    parser.add_argument("--payload-repeat", type=int, default=1, help="Payload repetition factor.")
    parser.add_argument(
        "--auto-repeat-k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable automatic selection of repeat-k for text/id.",
    )
    parser.add_argument("--repeat-k-max", type=int, default=7, help="Maximum repeat-k to consider.")
    parser.add_argument("--payload-repeat-max", type=int, default=9, help="Maximum payload repeat to consider.")
    parser.add_argument(
        "--fill-repeat-k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable repeat-k fill to better utilize 32x32.",
    )
    parser.add_argument(
        "--fill-repeat-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable payload repetition fill to better utilize 32x32.",
    )
    parser.add_argument(
        "--repeat-k-target-success",
        type=float,
        default=0.99,
        help="Target bit success probability for auto repeat-k.",
    )
    parser.add_argument("--alpha", type=int, default=100, help="Base embedding strength.")
    parser.add_argument("--n", type=int, default=12, help="Number of selected ROIs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--auto-optimize-alpha-global",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable global Bayesian optimization for alpha.",
    )
    parser.add_argument(
        "--auto-optimize-alpha-per-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable per-image Bayesian optimization for alpha.",
    )
    parser.add_argument(
        "--use-affine-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable affine correction in extraction.",
    )
    parser.add_argument(
        "--global-bo-trials",
        type=int,
        default=5,
        help="Number of BO trials in global optimization.",
    )
    parser.add_argument(
        "--per-image-bo-trials",
        type=int,
        default=10,
        help="Number of BO trials in per-image optimization.",
    )
    return parser


def prompt_if_missing_args(mode, text_input, id_input):
    if "--mode" in sys.argv:
        return mode, text_input, id_input

    print("Select watermark mode:")
    print("  1) image")
    print("  2) text")
    print("  3) id")
    choice = input("Enter choice [1-3] (default=1): ").strip()

    mode_map = {"1": "image", "2": "text", "3": "id"}
    mode = mode_map.get(choice, "image")

    if mode == "text" and "--text-input" not in sys.argv:
        user_text = input(f"Enter text (default='{text_input}'): ").strip()
        if user_text:
            text_input = user_text

    if mode == "id" and "--id-input" not in sys.argv:
        user_id = input(f"Enter numeric ID (default='{id_input}'): ").strip()
        if user_id:
            id_input = user_id

    return mode, text_input, id_input


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    img_dir = args.img_dir
    wm_raw = args.wm_raw
    MODE = args.mode
    text_input = args.text_input
    id_input = args.id_input
    text_encoding = args.text_encoding
    repeat_k = args.repeat_k
    payload_repeat = args.payload_repeat
    auto_repeat_k = args.auto_repeat_k
    repeat_k_max = args.repeat_k_max
    payload_repeat_max = args.payload_repeat_max
    fill_repeat_k = args.fill_repeat_k
    fill_repeat_payload = args.fill_repeat_payload
    repeat_k_target_success = args.repeat_k_target_success
    alpha = args.alpha
    N = args.n
    seed = args.seed
    auto_optimize_alpha_global = args.auto_optimize_alpha_global
    auto_optimize_alpha_per_image = args.auto_optimize_alpha_per_image
    use_affine_correction = args.use_affine_correction

    MODE, text_input, id_input = prompt_if_missing_args(MODE, text_input, id_input)

    if not os.path.isdir(img_dir):
        print(f"[!] Input directory not found: {img_dir}")
        raise SystemExit(1)

    if MODE == "image" and not os.path.isfile(wm_raw):
        print(f"[!] Watermark image not found: {wm_raw}")
        raise SystemExit(1)

    all_img_paths = sorted(
        [
            os.path.join(img_dir, img_file)
            for img_file in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, img_file))
        ]
    )

    if len(all_img_paths) == 0:
        print("[!] No input images found in the directory.")
        raise SystemExit(0)

    global_alpha = alpha
    calib_paths = all_img_paths[: min(5, len(all_img_paths))]

    if auto_repeat_k and MODE in ("text", "id"):
        if MODE == "text":
            wm_bin_raw = text_to_bin_image(text_input, repeat_k=1, encoding=text_encoding)
        else:
            wm_bin_raw = id_to_bin_image(id_input, repeat_k=1)

        ber = estimate_bit_error_rate(
            host_paths=calib_paths[: min(3, len(calib_paths))],
            wm_bin=wm_bin_raw,
            alpha_val=global_alpha,
            N=N,
            seed=seed,
            use_affine_correction=use_affine_correction,
        )
        repeat_k = choose_repeat_k_from_ber(
            ber,
            target_success=repeat_k_target_success,
            max_k=repeat_k_max,
        )
        print(f"[*] Auto repeat-k: ber={ber} -> repeat_k={repeat_k}")

    if MODE in ("text", "id") and (fill_repeat_k or fill_repeat_payload):
        if MODE == "text":
            base_bits_len = get_text_payload_bits_len(text_input, encoding=text_encoding)
        else:
            base_bits_len = get_id_payload_bits_len(id_input)

        if fill_repeat_payload:
            payload_repeat = choose_payload_repeat_to_fill(
                base_bits_len,
                repeat_k,
                payload_repeat,
                min_side=32,
                max_repeat=payload_repeat_max,
            )

        if fill_repeat_k:
            repeat_k = choose_repeat_k_to_fill(
                base_bits_len * payload_repeat,
                repeat_k,
                min_side=32,
                max_k=repeat_k_max,
            )

        print(f"[*] Fill strategy -> repeat_k={repeat_k}, payload_repeat={payload_repeat}")

    if auto_optimize_alpha_global:
        best_alpha, best_score, _ = optimize_alpha_bayesian(
            host_paths=calib_paths,
            wm_raw_path=wm_raw,
            MODE=MODE,
            text_input=text_input,
            id_input=id_input,
            repeat_k=repeat_k,
            payload_repeat=payload_repeat,
            text_encoding=text_encoding,
            N=N,
            seed=seed,
            use_affine_correction=use_affine_correction,
            alpha_min=80,
            alpha_max=120,
            n_trials=args.global_bo_trials,
            psnr_min=35,
            ssim_min=None,
            nc_min=0.88,
            weight_psnr=0.45,
            weight_nc=0.55,
            weight_ber=0.20,
            random_seed=seed,
        )
        global_alpha = int(round(best_alpha))
        print(f"[*] Global BO selected alpha={global_alpha} (score={best_score:.4f})")

    for img_path in all_img_paths:
        img_file = os.path.basename(img_path)
        current_alpha = global_alpha

        if auto_optimize_alpha_per_image:
            best_alpha, best_score, _ = optimize_alpha_bayesian(
                host_paths=[img_path],
                wm_raw_path=wm_raw,
                MODE=MODE,
                text_input=text_input,
                id_input=id_input,
                repeat_k=repeat_k,
                payload_repeat=payload_repeat,
                text_encoding=text_encoding,
                N=N,
                seed=seed,
                use_affine_correction=use_affine_correction,
                alpha_min=85,
                alpha_max=120,
                n_trials=args.per_image_bo_trials,
                psnr_min=38,
                ssim_min=None,
                nc_min=0.90,
                weight_psnr=0.50,
                weight_nc=0.50,
                weight_ber=0.20,
                random_seed=seed,
            )
            current_alpha = int(round(best_alpha))
            print(f"[*] Per-image BO for {img_file}: alpha={current_alpha} (score={best_score:.4f})")

        print(f"Processing: {img_file}")
        main(
            img_path,
            wm_raw,
            MODE,
            text_input,
            id_input,
            repeat_k,
            payload_repeat,
            text_encoding,
            current_alpha,
            N,
            seed,
            use_affine_correction=use_affine_correction,
        )

