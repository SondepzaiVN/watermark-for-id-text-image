import argparse
import os
import sys
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.watermark_pipeline import (
    main,
    extract_only,
    optimize_alpha_bayesian,
    optimize_pareto_roi_alpha,
    optimize_roi_alpha_grid,
    optimize_fast,
    estimate_bit_error_rate,
    choose_repeat_k_from_ber,
    choose_repeat_k_to_fill,
    choose_payload_repeat_to_fill,
    get_text_payload_bits_len,
    get_id_payload_bits_len,
    text_to_bin_image,
    id_to_bin_image,
)


def save_fill_k_preview_images(mode, text_input, id_input, text_encoding, payload_repeat, repeat_k_before_fill, repeat_k_after_fill, payload_seed):
    if mode not in ("text", "id"):
        return

    os.makedirs("results", exist_ok=True)
    before_path = os.path.join("results", "watermark_binary_before_fill_k.jpg")
    after_path = os.path.join("results", "watermark_binary_after_fill_k.jpg")

    if mode == "text":
        wm_before = text_to_bin_image(
            text_input,
            repeat_k=repeat_k_before_fill,
            payload_repeat=payload_repeat,
            encoding=text_encoding,
            payload_seed=payload_seed,
        )
        wm_after = text_to_bin_image(
            text_input,
            repeat_k=repeat_k_after_fill,
            payload_repeat=payload_repeat,
            encoding=text_encoding,
            payload_seed=payload_seed,
        )
    else:
        wm_before = id_to_bin_image(
            id_input,
            repeat_k=repeat_k_before_fill,
            payload_repeat=payload_repeat,
            payload_seed=payload_seed,
        )
        wm_after = id_to_bin_image(
            id_input,
            repeat_k=repeat_k_after_fill,
            payload_repeat=payload_repeat,
            payload_seed=payload_seed,
        )

    cv2.imwrite(before_path, (wm_before * 255).astype("uint8"))
    cv2.imwrite(after_path, (wm_after * 255).astype("uint8"))
    print(f"[*] Saved before-fill watermark: {before_path} (repeat_k={repeat_k_before_fill})")
    print(f"[*] Saved after-fill watermark: {after_path} (repeat_k={repeat_k_after_fill})")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run robust watermarking experiments on host images."
    )
    parser.add_argument("--img-dir", default="data/host_images", help="Directory containing host images.")
    parser.add_argument(
        "--img-path",
        default=None,
        help="Single image path to process (overrides --img-dir).",
    )
    parser.add_argument(
        "--wm-raw",
        default="data/watermark/cict.png",
        help="Path to watermark image when mode=image.",
    )
    parser.add_argument(
        "--mode",
        default="image",
        choices=["image", "text", "id", "extract"],
        help="Watermark mode: image, text, id, or extract.",
    )
    parser.add_argument("--text-input", default="Hello World", help="Watermark text when mode=text.")
    parser.add_argument("--id-input", default="012345678910", help="Watermark ID when mode=id.")
    parser.add_argument(
        "--text-encoding",
        default="utf-8",
        choices=["ascii", "utf-8"],
        help="Text encoding for mode=text. Note: UTF-8 can store Vietnamese characters.",
    )
    parser.add_argument("--repeat-k", type=int, default=3, help="Bit repetition factor (odd number).")
    parser.add_argument("--payload-repeat", type=int, default=1, help="Payload repetition factor.")
    parser.add_argument(
        "--auto-repeat-k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable automatic selection of repeat-k for text/id.",
    )
    parser.add_argument(
        "--repeat-k-max",
        type=int,
        default=7,
        help="Maximum repeat-k to consider for BER-based auto selection.",
    )
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
    parser.add_argument("--n", type=int, default=10, help="Number of selected ROIs.")
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
        "--auto-optimize-pareto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable Pareto optimization over ROI count and alpha.",
    )
    parser.add_argument(
        "--pareto-per-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable per-image Pareto optimization.",
    )
    parser.add_argument(
        "--auto-optimize-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable per-image grid search for ROI count and alpha.",
    )
    parser.add_argument("--pareto-alpha-min", type=float, default=80.0, help="Pareto alpha min.")
    parser.add_argument("--pareto-alpha-max", type=float, default=120.0, help="Pareto alpha max.")
    parser.add_argument("--pareto-alpha-step", type=float, default=5.0, help="Pareto alpha step.")
    parser.add_argument("--pareto-n-min", type=int, default=6, help="Pareto N min.")
    parser.add_argument("--pareto-n-max", type=int, default=14, help="Pareto N max.")
    parser.add_argument("--pareto-n-step", type=int, default=2, help="Pareto N step.")
    parser.add_argument(
        "--pareto-n-values",
        default="1,2,4,8,12,16,24",
        help="Comma-separated ROI counts for Pareto search (overrides min/max/step).",
    )
    parser.add_argument("--pareto-psnr-min", type=float, default=40.0, help="Pareto PSNR min.")
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
    parser.add_argument(
        "--fast-optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast binary-search optimizer (5-10x faster). Disable with --no-fast-optimize to use full grid search.",
    )
    return parser


def prompt_if_missing_args(mode, text_input, id_input, img_dir, img_path, text_encoding):
    if "--mode" in sys.argv:
        selected_mode = mode
    else:
        print("Select watermark mode:")
        print("  1) image")
        print("  2) text")
        print("  3) id")
        print("  4) extract")
        choice = input("Enter choice [1-4] (default=1): ").strip()

        mode_map = {"1": "image", "2": "text", "3": "id", "4": "extract"}
        selected_mode = mode_map.get(choice, "image")

    # Prompt for text encoding when in text mode and not provided via CLI
    if selected_mode == "text" and "--text-encoding" not in sys.argv:
        print("Select text encoding:")
        print("  1) ASCII (basic Latin)")
        print("  2) UTF-8 (supports Vietnamese and other languages)")
        enc_choice = input("Enter choice [1-2] (default=2): ").strip()
        enc_map = {"1": "ascii", "2": "utf-8"}
        text_encoding = enc_map.get(enc_choice, text_encoding)
        if text_encoding.lower() == "utf-8":
            print("[*] Note: UTF-8 can store Vietnamese characters.")

    if selected_mode == "text" and "--text-input" not in sys.argv:
        user_text = input(f"Enter text (default='{text_input}'): ").strip()
        if user_text:
            text_input = user_text

    if selected_mode == "id" and "--id-input" not in sys.argv:
        user_id = input(f"Enter numeric ID (default='{id_input}'): ").strip()
        if user_id:
            id_input = user_id

    if selected_mode == "extract" and "--img-dir" not in sys.argv and img_path is None:
        img_dir = "data/host_images_extract"

    if "--img-path" not in sys.argv and "--img-dir" not in sys.argv:
        print("Select input scope:")
        print("  0) directory")
        print("  1) single image")
        scope = input("Enter choice [0-1] (default=0): ").strip()
        if scope == "1":
            if not os.path.isdir(img_dir):
                print(f"[!] Input directory not found: {img_dir}")
            else:
                img_name = input(f"Enter image filename in '{img_dir}': ").strip()
                if img_name:
                    img_path = os.path.join(img_dir, img_name)

    return selected_mode, text_input, id_input, img_dir, img_path, text_encoding


def resize_image_keep_ratio(image, target_min=None, target_max=None):
    height, width = image.shape[:2]
    min_side = min(height, width)
    max_side = max(height, width)

    if target_min is not None and min_side < target_min:
        scale = target_min / float(min_side)
    elif target_max is not None and max_side > target_max:
        scale = target_max / float(max_side)
    else:
        return image

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def prompt_resize_for_host_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Failed to read image: {image_path}")
        return image_path

    height, width = image.shape[:2]
    min_side = min(height, width)
    max_side = max(height, width)

    target_min = None
    target_max = None

    if max_side > 2048:
        print(f"[*] Anh host {os.path.basename(image_path)} co canh lon hon 2048px.")
        print("Chon cach resize (giu ti le):")
        print("  1) Canh lon nhat = 2048px (phu hop Facebook)")
        print("  2) Canh lon nhat = 1080px (phu hop Instagram/Thread)")
        print("  3) Canh lon nhat = 1024px (phu hop Zalo)")
        print("  4) Khong resize")
        choice = input("Nhap lua chon [1-4] (default=4): ").strip() or "4"
        if choice == "1":
            target_max = 2048
        elif choice == "2":
            target_max = 1080
        elif choice == "3":
            target_max = 1024
    elif max_side > 1024:
        print(f"[*] Anh host {os.path.basename(image_path)} co canh lon hon 1024px.")
        print("Chon cach resize (giu ti le):")
        print("  1) Canh lon nhat = 1024px (phu hop Zalo)")
        print("  2) Khong resize")
        choice = input("Nhap lua chon [1-2] (default=2): ").strip() or "2"
        if choice == "1":
            target_max = 1024
    elif max_side > 1080:
        print(f"[*] Anh host {os.path.basename(image_path)} co canh lon hon 1080px.")
        print("Chon cach resize (giu ti le):")
        print("  1) Canh lon nhat = 1080px (phu hop Instagram/Thread)")
        print("  2) Khong resize")
        choice = input("Nhap lua chon [1-2] (default=2): ").strip() or "2"
        if choice == "1":
            target_max = 1080
    elif min_side < 320:
        print(f"[*] Anh host {os.path.basename(image_path)} co canh nho hon 320px.")
        print("Chon cach resize (giu ti le):")
        print("  1) Canh nho nhat = 320px")
        print("  2) Khong resize")
        choice = input("Nhap lua chon [1-2] (default=2): ").strip() or "2"
        if choice == "1":
            target_min = 320

    if target_min is None and target_max is None:
        return image_path

    resized = resize_image_keep_ratio(image, target_min=target_min, target_max=target_max)
    if resized is image:
        return image_path

    os.makedirs("results/resize", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    resized_path = os.path.join("results/resize", f"{base_name}_resized.jpg")
    cv2.imwrite(resized_path, resized)
    print(f"[*] Saved resized image: {resized_path}")
    return resized_path


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    img_dir = args.img_dir
    img_path = args.img_path
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
    auto_optimize_pareto = args.auto_optimize_pareto
    pareto_per_image = args.pareto_per_image
    auto_optimize_grid = args.auto_optimize_grid
    use_affine_correction = args.use_affine_correction
    fast_optimize = args.fast_optimize

    MODE, text_input, id_input, img_dir, img_path, text_encoding = prompt_if_missing_args(
        MODE, text_input, id_input, img_dir, img_path, text_encoding
    )

    if (
        MODE == "extract"
        and args.img_dir == "data/host_images"
        and "--img-dir" not in sys.argv
        and img_path is None
    ):
        img_dir = "data/host_images_extract"

    if img_path is None:
        if not os.path.isdir(img_dir):
            print(f"[!] Input directory not found: {img_dir}")
            raise SystemExit(1)
    else:
        if not os.path.isfile(img_path):
            print(f"[!] Input image not found: {img_path}")
            raise SystemExit(1)

    if MODE == "image" and not os.path.isfile(wm_raw):
        print(f"[!] Watermark image not found: {wm_raw}")
        raise SystemExit(1)

    if img_path is None:
        all_img_paths = sorted(
            [
                os.path.join(img_dir, img_file)
                for img_file in os.listdir(img_dir)
                if os.path.isfile(os.path.join(img_dir, img_file))
            ]
        )
    else:
        all_img_paths = [img_path]

    if len(all_img_paths) == 0:
        print("[!] No input images found in the directory.")
        raise SystemExit(0)

    if MODE == "extract":
        for img_path in all_img_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            key_path = os.path.join("results", f"{img_name}.npz")
            if not os.path.isfile(key_path):
                print(f"[!] Key not found for {img_name}: {key_path}")
                continue
            output_dir = os.path.join("results", img_name, "extract")
            print(f"Extracting: {img_name}")
            extract_only(
                img_path,
                key_path,
                output_dir,
                use_affine_correction=use_affine_correction,
            )
        raise SystemExit(0)

    processed_paths = []
    for img_path in all_img_paths:
        processed_paths.append(prompt_resize_for_host_image(img_path))
    all_img_paths = processed_paths

    global_alpha = alpha
    calib_paths = all_img_paths[: min(5, len(all_img_paths))]

    n_values = []
    if args.pareto_n_values:
        for token in args.pareto_n_values.split(","):
            token = token.strip()
            if token:
                try:
                    n_values.append(int(token))
                except ValueError:
                    pass

    if auto_repeat_k and MODE in ("text", "id"):
        if MODE == "text":
            wm_bin_raw = text_to_bin_image(
                text_input,
                repeat_k=1,
                encoding=text_encoding,
                payload_seed=seed,
            )
        else:
            wm_bin_raw = id_to_bin_image(id_input, repeat_k=1, payload_seed=seed)

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

    repeat_k_before_fill = repeat_k
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

        repeat_k_before_fill = repeat_k
        if fill_repeat_k:
            repeat_k = choose_repeat_k_to_fill(
                base_bits_len * payload_repeat,
                repeat_k,
                min_side=32,
            )

        print(f"[*] Fill strategy -> repeat_k={repeat_k}, payload_repeat={payload_repeat}")

    if MODE in ("text", "id"):
        save_fill_k_preview_images(
            MODE,
            text_input,
            id_input,
            text_encoding,
            payload_repeat,
            repeat_k_before_fill,
            repeat_k,
            seed,
        )
        print(
            f"[*] Payload settings -> seed={seed} | repeat_k={repeat_k} | payload_repeat={payload_repeat}"
        )

    if auto_optimize_pareto and not pareto_per_image:
        best_alpha, best_n, pareto_front = optimize_pareto_roi_alpha(
            host_paths=calib_paths,
            wm_raw_path=wm_raw,
            MODE=MODE,
            text_input=text_input,
            id_input=id_input,
            repeat_k=repeat_k,
            payload_repeat=payload_repeat,
            text_encoding=text_encoding,
            seed=seed,
            use_affine_correction=use_affine_correction,
            alpha_min=args.pareto_alpha_min,
            alpha_max=args.pareto_alpha_max,
            alpha_step=args.pareto_alpha_step,
            n_min=args.pareto_n_min,
            n_max=args.pareto_n_max,
            n_step=args.pareto_n_step,
            n_values=n_values,
            psnr_min=args.pareto_psnr_min,
            random_seed=seed,
        )
        global_alpha = int(round(best_alpha))
        N = int(best_n)
        print(
            f"[*] Pareto selected alpha={global_alpha}, N={N} | front_size={len(pareto_front)}"
        )

    if auto_optimize_alpha_global and not auto_optimize_pareto:
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

        if auto_optimize_grid:
            if fast_optimize:
                best_alpha, best_n = optimize_fast(
                    host_paths=[img_path],
                    wm_raw_path=wm_raw,
                    MODE=MODE,
                    text_input=text_input,
                    id_input=id_input,
                    repeat_k=repeat_k,
                    payload_repeat=payload_repeat,
                    text_encoding=text_encoding,
                    seed=seed,
                    use_affine_correction=use_affine_correction,
                    alpha_min=args.pareto_alpha_min,
                    alpha_max=args.pareto_alpha_max,
                    alpha_step=args.pareto_alpha_step,
                    n_values=n_values,
                    psnr_min=args.pareto_psnr_min,
                    random_seed=seed,
                )
                print(f"[*] Fast optimizer for {img_file}: alpha={best_alpha:.1f}, N={best_n}")
            else:
                best_alpha, best_n = optimize_roi_alpha_grid(
                    host_paths=[img_path],
                    wm_raw_path=wm_raw,
                    MODE=MODE,
                    text_input=text_input,
                    id_input=id_input,
                    repeat_k=repeat_k,
                    payload_repeat=payload_repeat,
                    text_encoding=text_encoding,
                    seed=seed,
                    use_affine_correction=use_affine_correction,
                    alpha_min=args.pareto_alpha_min,
                    alpha_max=args.pareto_alpha_max,
                    alpha_step=args.pareto_alpha_step,
                    n_values=n_values,
                    psnr_min=args.pareto_psnr_min,
                    random_seed=seed,
                )
                print(f"[*] Grid per-image {img_file}: alpha={best_alpha:.1f}, N={best_n}")
            current_alpha = int(round(best_alpha))
            N = int(best_n)

        if auto_optimize_pareto and pareto_per_image:
            best_alpha, best_n, pareto_front = optimize_pareto_roi_alpha(
                host_paths=[img_path],
                wm_raw_path=wm_raw,
                MODE=MODE,
                text_input=text_input,
                id_input=id_input,
                repeat_k=repeat_k,
                payload_repeat=payload_repeat,
                text_encoding=text_encoding,
                seed=seed,
                use_affine_correction=use_affine_correction,
                alpha_min=args.pareto_alpha_min,
                alpha_max=args.pareto_alpha_max,
                alpha_step=args.pareto_alpha_step,
                n_min=args.pareto_n_min,
                n_max=args.pareto_n_max,
                n_step=args.pareto_n_step,
                n_values=n_values,
                psnr_min=args.pareto_psnr_min,
                random_seed=seed,
            )
            current_alpha = int(round(best_alpha))
            N = int(best_n)
            print(
                f"[*] Pareto per-image {img_file}: alpha={current_alpha}, N={N} | front_size={len(pareto_front)}"
            )

        if auto_optimize_alpha_per_image and not auto_optimize_pareto and not auto_optimize_grid:
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

