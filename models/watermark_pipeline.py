import cv2
import numpy as np
import pywt
import os
import importlib
import math
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim_func
import pandas as pd


def keypoints_to_array(keypoints):
    if not keypoints:
        return np.zeros((0, 7), dtype=np.float32)
    rows = np.zeros((len(keypoints), 7), dtype=np.float32)
    for i, kp in enumerate(keypoints):
        rows[i] = [
            kp.pt[0],
            kp.pt[1],
            kp.size,
            kp.angle,
            kp.response,
            float(kp.octave),
            float(kp.class_id),
        ]
    return rows


def array_to_keypoints(rows):
    if rows is None or len(rows) == 0:
        return []
    keypoints = []
    for row in rows:
        x, y, size, angle, response, octave, class_id = row
        keypoints.append(
            cv2.KeyPoint(
                float(x),
                float(y),
                float(size),
                float(angle),
                float(response),
                int(round(octave)),
                int(round(class_id)),
            )
        )
    return keypoints


def save_key_file(
    key_path,
    pos,
    safe_kp_orig,
    safe_des_orig,
    wm_shape,
    orig_shape,
    alpha,
    seed,
    mode,
    text_input,
    id_input,
    repeat_k,
    payload_repeat,
    text_encoding,
    host_path,
    wm_raw_path,
):
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    safe_des = safe_des_orig if safe_des_orig is not None else np.zeros((0, 1), dtype=np.float32)
    np.savez_compressed(
        key_path,
        pos_kp=keypoints_to_array(pos),
        safe_kp=keypoints_to_array(safe_kp_orig),
        safe_des=safe_des,
        wm_shape=np.array(wm_shape, dtype=np.int32),
        orig_shape=np.array(orig_shape, dtype=np.int32),
        alpha=np.array([alpha], dtype=np.float32),
        seed=np.array([seed], dtype=np.int32),
        mode=np.array([mode]),
        text_input=np.array([text_input]),
        id_input=np.array([id_input]),
        repeat_k=np.array([repeat_k], dtype=np.int32),
        payload_repeat=np.array([payload_repeat], dtype=np.int32),
        text_encoding=np.array([text_encoding]),
        host_path=np.array([host_path]),
        wm_raw_path=np.array([wm_raw_path]),
    )


def load_key_file(key_path):
    data = np.load(key_path, allow_pickle=True)
    wm_shape = tuple(int(v) for v in data.get("wm_shape", []))
    orig_shape = tuple(int(v) for v in data.get("orig_shape", []))
    alpha = float(data.get("alpha", [0.0])[0])
    seed = int(data.get("seed", [0])[0])
    mode = str(data.get("mode", ["image"])[0])
    repeat_k = int(data.get("repeat_k", [1])[0])
    payload_repeat = int(data.get("payload_repeat", [1])[0])
    text_encoding = str(data.get("text_encoding", ["utf-8"])[0])
    return {
        "pos": array_to_keypoints(data.get("pos_kp", np.zeros((0, 7), dtype=np.float32))),
        "safe_kp_orig": array_to_keypoints(data.get("safe_kp", np.zeros((0, 7), dtype=np.float32))),
        "safe_des_orig": data.get("safe_des", np.array([])),
        "wm_shape": wm_shape,
        "orig_shape": orig_shape,
        "alpha": alpha,
        "seed": seed,
        "mode": mode,
        "text_input": str(data.get("text_input", [""])[0]),
        "id_input": str(data.get("id_input", [""])[0]),
        "repeat_k": repeat_k,
        "payload_repeat": payload_repeat,
        "text_encoding": text_encoding,
        "host_path": str(data.get("host_path", [""])[0]),
        "wm_raw_path": str(data.get("wm_raw_path", [""])[0]),
    }

# --- 1. SCRAMBLING FUNCTIONS ---
def redistribute_block(block, undo=False):
    m = block.shape[0]
    B = np.zeros_like(block)
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            # [cite_start]Formula from Equation (16) in the paper [cite: 328-335]
            new_i = (2*i - 1) % (m + 1)
            new_j = (2*j - 1) % (m + 1)
            if not undo:
                B[new_i-1, new_j-1] = block[i-1, j-1]
            else:
                B[i-1, j-1] = block[new_i-1, new_j-1]
    return B

def scramble_watermark(wm_bin, seed):
    """Shuffle watermark bit positions using a deterministic seed."""
    flat_wm = wm_bin.flatten()
    indices = np.arange(len(flat_wm))
    np.random.seed(seed)
    np.random.shuffle(indices)
    scrambled_wm = flat_wm[indices]
    return scrambled_wm, indices

def unscramble_watermark(scrambled_wm, indices, shape):
    """Restore bit positions back to the original image shape."""
    unscrambled = np.zeros_like(scrambled_wm)
    unscrambled[indices] = scrambled_wm
    return unscrambled.reshape(shape)

# --- 2. BASIC DCT FUNCTIONS ---

def apply_dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def calculate_entropy(roi):
    """Compute Shannon entropy for an image region."""
    # Convert to a grayscale histogram representation
    marg = np.histogram(roi, bins=256, range=(0, 255))[0]
    marg = marg / np.sum(marg) # Estimate probability p(i)
    marg = marg[marg > 0] # Remove zeros to avoid log issues
    return -np.sum(marg * np.log2(marg))

COEFF_POS = [(1, 1)]

JPEG_GRID_SIZE = 8
SIFT_RATIO_TEST = 0.75
MIN_AFFINE_MATCHES = 2
MIN_AFFINE_INLIERS = 4
MIN_AFFINE_INLIER_RATIO = 0.75

def filter_keypoints_in_annulus(kp, des, cx, cy, inner_radius, safe_radius):
    """Filter keypoints/descriptors inside a safe annulus region."""
    if kp is None or des is None or len(kp) == 0 or len(des) == 0:
        return [], np.array([])

    safe_kp = []
    safe_des = []
    for i, p in enumerate(kp):
        dist = np.sqrt((p.pt[0] - cx) ** 2 + (p.pt[1] - cy) ** 2)
        if inner_radius < dist < safe_radius:
            safe_kp.append(p)
            safe_des.append(des[i])

    if len(safe_des) == 0:
        return safe_kp, np.array([])
    return safe_kp, np.array(safe_des)

def align_roi_center_to_jpeg_grid(x, y, half_s, img_w, img_h, grid=JPEG_GRID_SIZE):
    """Align ROI center so its top-left corner lies on an 8x8 JPEG grid."""
    roi_size = 2 * half_s
    if roi_size <= 0 or roi_size > img_w or roi_size > img_h:
        return None

    min_x0, max_x0 = 0, img_w - roi_size
    min_y0, max_y0 = 0, img_h - roi_size

    def nearest_grid_anchor(value, min_v, max_v):
        low = int(np.ceil(min_v / grid) * grid)
        high = int(np.floor(max_v / grid) * grid)
        if low <= high:
            aligned = int(np.round(value / grid) * grid)
            if aligned < low:
                aligned = low
            elif aligned > high:
                aligned = high
            return aligned
        return int(np.clip(value, min_v, max_v))

    x0 = nearest_grid_anchor(x - half_s, min_x0, max_x0)
    y0 = nearest_grid_anchor(y - half_s, min_y0, max_y0)
    x1 = x0 + roi_size
    y1 = y0 + roi_size

    cx = x0 + half_s
    cy = y0 + half_s
    return cx, cy, x0, y0, x1, y1

# --- 3. WATERMARK EMBEDDING ---
def embed_watermark(host_bgr, wm_bin, alpha=20.0, N=16, seed=42, inner_ratio=0): 
    """
    Embed the watermark into high-entropy ROIs inside an annulus region
    to reduce data loss under rotation and minimize overlap near the center.
    """
    h_img, w_img = host_bgr.shape[:2]
    cx, cy = w_img // 2, h_img // 2

    scrambled_bits, _ = scramble_watermark(wm_bin, seed)
    embed_length = len(scrambled_bits)

    # --- COMPUTE ROI SIZE ---
    bits_per_block = 2 * len(COEFF_POS)
    num_blocks_needed = int(np.ceil(embed_length / bits_per_block))
    roi_side = max(int(np.ceil(np.sqrt(num_blocks_needed)) * 4), 32)
    if roi_side % 2 != 0: roi_side += 1
    half_s = roi_side // 2

    # --- DEFINE VALID EMBEDDING REGION (ANNULUS) ---
    # Outer safe region to reduce data loss under rotation
    safe_radius = min(cx, cy) - half_s - 10 
    # Inner exclusion region to reduce overlap and improve spatial distribution
    inner_radius = min(cx, cy) * inner_ratio

    Y, Cr, Cb = cv2.split(cv2.cvtColor(host_bgr, cv2.COLOR_BGR2YCrCb))
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(Y, None)
    
    if kp is None or len(kp) == 0:
        print("[!] No SIFT keypoints were detected.")
        return host_bgr, np.array([]), [], np.array([]), []

    # 1. Filter keypoints that lie inside the annulus region
    candidate_indices = []
    for i, p in enumerate(kp):
        dist = np.sqrt((p.pt[0] - cx)**2 + (p.pt[1] - cy)**2)
        # Condition: outside inner_radius and inside safe_radius
        if inner_radius < dist < safe_radius:
            candidate_indices.append(i)

    safe_kp_orig = [kp[i] for i in candidate_indices]
    safe_des_orig = des[candidate_indices] if len(candidate_indices) > 0 else np.array([])
            
    # 2. Score ROI candidates by entropy
    scored_candidates = []
    for idx in candidate_indices:
        p = kp[idx]
        x, y = int(round(p.pt[0])), int(round(p.pt[1]))
        aligned = align_roi_center_to_jpeg_grid(x, y, half_s, w_img, h_img)
        if aligned is None:
            continue
        ax, ay, x0, y0, x1, y1 = aligned
        
        # Probe the ROI candidate
        roi_test = Y[y0:y1, x0:x1]
        
        # Ensure ROI stays fully inside image boundaries
        if roi_test.shape[0] == roi_side and roi_test.shape[1] == roi_side:
            entropy_val = calculate_entropy(roi_test)
            scored_candidates.append((idx, entropy_val, ax, ay, x0, y0, x1, y1))

    # Sort candidates by entropy from high to low
    scored_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    # 3. Select non-overlapping ROIs
    selected_kp = []
    selected_des_list = []

    target_roi = max(int(N), 1)

    for idx, entropy, cand_x, cand_y, x0, y0, x1, y1 in scored_candidates:
        if len(selected_kp) >= target_roi:
            break
            
        is_overlapping = False
        for accepted_p in selected_kp:
            acc_x, acc_y = accepted_p.pt
            # Check whether the new ROI overlaps any accepted ROI
            if abs(cand_x - acc_x) < roi_side and abs(cand_y - acc_y) < roi_side:
                is_overlapping = True
                break
        
        if not is_overlapping:
            kp_src = kp[idx]
            aligned_kp = cv2.KeyPoint(float(cand_x), float(cand_y), kp_src.size,
                                      kp_src.angle, kp_src.response, kp_src.octave, kp_src.class_id)
            selected_kp.append(aligned_kp)
            selected_des_list.append(des[idx])

    print(f"[*] Found {len(selected_kp)} entropy-optimized ROIs (target N={target_roi}).")

    if len(selected_kp) == 0:
        print("[!] Warning: no valid ROI remained after entropy filtering.")
        return host_bgr, np.array([]), [], safe_des_orig, safe_kp_orig

    selected_des = np.array(selected_des_list)

    # 4. Embed watermark bits (SWT + DCT + QIM)
    for p in selected_kp:
        x, y = int(p.pt[0]), int(p.pt[1])
        aligned = align_roi_center_to_jpeg_grid(x, y, half_s, w_img, h_img)
        if aligned is None:
            continue
        x, y, x0, y0, x1, y1 = aligned
        roi = Y[y0:y1, x0:x1].astype(np.float64)
        
        # One-level SWT decomposition
        coeffs = pywt.swt2(roi, 'haar', level=1)
        cA, (lh, hl, hh) = coeffs[0]
        
        idx_bit = 0
        # Iterate over 4x4 blocks in HL and LH subbands
        for i in range(0, hl.shape[0], 4):
            for j in range(0, hl.shape[1], 4):
                if idx_bit >= embed_length: break
                
                for band in [hl, lh]:
                    # Redistribute block elements
                    block_sorted = redistribute_block(band[i:i+4, j:j+4], undo=False)
                    # Apply DCT
                    b_dct = apply_dct2(block_sorted)
                    
                    for pos_coeff in COEFF_POS:
                        if idx_bit < embed_length:
                            val = b_dct[pos_coeff]
                            bit = scrambled_bits[idx_bit]
                            step = alpha
                            
                            # QIM quantization
                            k = np.floor(np.ceil(val / step) / 2)
                            if bit == 1: 
                                q1, q2 = 2*k*step + 0.5*step, 2*k*step - 1.5*step
                            else: 
                                q1, q2 = 2*k*step - 0.5*step, 2*k*step + 1.5*step
                            
                            b_dct[pos_coeff] = q2 if abs(val - q2) < abs(val - q1) else q1
                            idx_bit += 1
                            
                    # Inverse transform
                    band[i:i+4, j:j+4] = redistribute_block(apply_idct2(b_dct), undo=True)
        
        # Reconstruct image after SWT
        new_coeffs = [(cA, (lh, hl, hh))]
        Y[y0:y1, x0:x1] = np.round(pywt.iswt2(new_coeffs, 'haar')).clip(0, 255)

    watermarked = cv2.cvtColor(cv2.merge([Y, Cr, Cb]), cv2.COLOR_YCrCb2BGR)
    return watermarked, selected_des, selected_kp, safe_des_orig, safe_kp_orig

# --- 4. WATERMARK EXTRACTION ---
def extract_watermark(wat_bgr, safe_des_orig, safe_kp_orig, alpha, list_pos, wm_shape, seed,
                      use_affine_correction=False, inner_ratio=0):
    """
    Extract watermark bits with optional geometric correction:
    - If enough matches are available, use affine correction (rotation + scale + RANSAC).
    - Otherwise continue with direct extraction.
    """
    sift = cv2.SIFT_create()
    Y, _, _ = cv2.split(cv2.cvtColor(wat_bgr, cv2.COLOR_BGR2YCrCb))
    kp_att, des_att = sift.detectAndCompute(Y, None)
    
    h, w = wat_bgr.shape[:2]
    cx, cy = w / 2, h / 2 # Absolute image center
    
    corrected_img = wat_bgr.copy()
    attack_angle = 0.0

    # Recompute ROI size to match the safe-radius used during embedding.
    embed_length = np.prod(wm_shape)
    bits_per_block = 2 * len(COEFF_POS)
    num_blocks_needed = int(np.ceil(embed_length / bits_per_block))
    roi_side = max(int(np.ceil(np.sqrt(num_blocks_needed)) * 4), 32)
    if roi_side % 2 != 0:
        roi_side += 1
    half_s = roi_side // 2

    safe_radius = min(cx, cy) - half_s - 10
    inner_radius = min(cx, cy) * inner_ratio

    if use_affine_correction and (safe_des_orig is None or len(safe_des_orig) < MIN_AFFINE_MATCHES or safe_kp_orig is None or len(safe_kp_orig) < MIN_AFFINE_MATCHES):
        print(f"[!] Insufficient original safe-radius SIFT data (minimum {MIN_AFFINE_MATCHES} required).")
        return np.zeros(wm_shape, dtype=np.uint8)

    if use_affine_correction and des_att is not None and len(des_att) > 0:
        kp_att_safe, des_att_safe = filter_keypoints_in_annulus(
            kp_att, des_att, cx, cy, inner_radius, safe_radius
        )
        print(f"[*] SIFT safe-radius: original={len(safe_kp_orig)} | attacked={len(kp_att_safe)}")

        if des_att_safe is not None and len(des_att_safe) > 0:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            raw_matches = bf.knnMatch(safe_des_orig, des_att_safe, k=2)

            good_matches = []
            for m_n in raw_matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < SIFT_RATIO_TEST * n.distance:
                        good_matches.append(m)

            print(f"[*] Good matches found in safe-radius: {len(good_matches)}")

            if len(good_matches) >= MIN_AFFINE_MATCHES:
                src_pts = np.float32([safe_kp_orig[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_att_safe[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M_affine, inlier_mask = cv2.estimateAffinePartial2D(
                    src_pts,
                    dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0
                )

                if M_affine is not None:
                    inlier_count = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
                    inlier_ratio = inlier_count / max(len(good_matches), 1)

                    if inlier_count >= MIN_AFFINE_INLIERS and inlier_ratio >= MIN_AFFINE_INLIER_RATIO:
                        raw_rot_rad = np.arctan2(M_affine[1, 0], M_affine[0, 0])
                        attack_angle = np.degrees(raw_rot_rad)
                        M_inv = cv2.invertAffineTransform(M_affine)
                        corrected_img = cv2.warpAffine(
                            wat_bgr,
                            M_inv,
                            (w, h),
                            flags=cv2.INTER_LANCZOS4,
                            borderValue=(255, 255, 255)
                        )
                    else:
                        print(
                            f"[!] Affine rejected: inlier={inlier_count}/{len(good_matches)} "
                            f"(ratio={inlier_ratio:.2f}), requires >= {MIN_AFFINE_INLIERS} and >= {MIN_AFFINE_INLIER_RATIO:.2f}."
                        )
                else:
                    print("[!] Could not estimate an affine matrix from safe-radius matches.")
            else:
                print(f"[!] Good safe-radius matches < {MIN_AFFINE_MATCHES}, skipping affine correction.")
        else:
            print("[!] No SIFT descriptors in the attacked safe-radius region.")
    elif not use_affine_correction:
        print("[*] Affine correction: OFF")

    print(f" [Rot: {attack_angle:>6.2f}°] ", end="")

    # --- BIT EXTRACTION STAGE ---
    Y_corr, _, _ = cv2.split(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2YCrCb))
    embed_length = np.prod(wm_shape)
    _, scramble_indices = scramble_watermark(np.zeros(wm_shape), seed)

    extracted_llr_list = []
    for p_orig in list_pos:
        x, y = int(p_orig.pt[0]), int(p_orig.pt[1])
        aligned = align_roi_center_to_jpeg_grid(x, y, half_s, Y_corr.shape[1], Y_corr.shape[0])
        if aligned is None:
            continue
        x, y, x0, y0, x1, y1 = aligned
            
        roi = Y_corr[y0:y1, x0:x1].astype(np.float64)
        coeffs = pywt.swt2(roi, 'haar', level=1)
        cA, (lh, hl, hh) = coeffs[0]
        
        bit_llr = []
        idx_bit = 0 
        for i in range(0, hl.shape[0], 4):
            for j in range(0, hl.shape[1], 4):
                if idx_bit >= embed_length: break
                for band in [hl, lh]:
                    block_sorted = redistribute_block(band[i:i+4, j:j+4], undo=False)
                    b_dct = apply_dct2(block_sorted)
                    for pos_coeff in COEFF_POS:
                        if idx_bit < embed_length:
                            val = b_dct[pos_coeff]
                            # Compute a hard bit, then accumulate soft evidence instead of hard voting.
                            hard_bit = int(np.ceil(val / alpha)) % 2

                            # Confidence from distance to nearest quantization threshold:
                            # r = |c - c_hat| / Delta, with c_hat as nearest threshold and Delta = alpha.
                            c_hat = np.round(val / alpha) * alpha
                            r = np.abs(val - c_hat) / alpha

                            # Map r -> correctness probability p in [0.5, 1.0), then convert to log-likelihood ratio.
                            eps = 1e-6
                            p_correct = np.clip(0.5 + r, 0.5 + eps, 1.0 - eps)
                            llr = np.log(p_correct / (1.0 - p_correct))

                            # Bit=1 contributes positive LLR, bit=0 contributes negative LLR.
                            bit_llr.append(llr if hard_bit == 1 else -llr)
                            idx_bit += 1
        extracted_llr_list.append(bit_llr)

    if not extracted_llr_list:
        return np.zeros(wm_shape, dtype=np.uint8)

    # Soft voting: sum log-likelihood across ROIs for each bit.
    llr_array = np.array(extracted_llr_list)
    final_bits = (np.sum(llr_array, axis=0) >= 0).astype(np.uint8)
    
    return unscramble_watermark(final_bits, scramble_indices, wm_shape)

# --- 6. ATTACK FUNCTIONS ---

def attack_salt_pepper(img, amount=0.01):
    out = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[tuple(coords)] = 255
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[tuple(coords)] = 0
    return out

def attack_jpeg(img, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def attack_zalo_compress(img, quality=85, downscale=0.75):
    """Approximate Zalo compression: downscale -> JPEG -> upscale."""
    h, w = img.shape[:2]
    scale = float(np.clip(downscale, 0.2, 1.0))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, encimg = cv2.imencode('.jpg', resized, encode_param)
    decoded = cv2.imdecode(encimg, 1)
    return cv2.resize(decoded, (w, h), interpolation=cv2.INTER_LINEAR)

def attack_gaussian_filter(img, size=3, sigma=0.1):
    return cv2.GaussianBlur(img, (size, size), sigma)

def attack_median_filter(img, size=3):
    return cv2.medianBlur(img, size)

def attack_average_filter(img, size=3):
    return cv2.blur(img, (size, size))

def attack_sharpening(img, amount=0.01):
    """Unsharp masking with a small gain to emulate mild sharpening."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    gain = float(max(amount, 0.0) * 10.0)
    sharpened = cv2.addWeighted(img, 1.0 + gain, blurred, -gain, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def attack_rotation(img, angle=5):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

def attack_cropping_corner(img, percent=25):
    out = img.copy()
    h, _ = img.shape[:2]
    side = int(np.sqrt(percent / 100) * h)
    out[0:side, 0:side] = 255
    return out

def attack_center_cropping(img, percent=50):
    out = img.copy()
    h, w = img.shape[:2]
    side = int(np.sqrt(percent / 100) * h)
    start_h, start_w = (h - side) // 2, (w - side) // 2
    out[start_h:start_h + side, start_w:start_w + side] = 255
    return out

def attack_scaling(img, factor=0.9):
    h, w = img.shape[:2]
    res = cv2.resize(img, (int(w * factor), int(h * factor)))
    return cv2.resize(res, (w, h))

def attack_brightness(img, value=50):
    return cv2.convertScaleAbs(img, alpha=1, beta=value)

def attack_contrast(img, factor=1.2):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def attack_brightness_adjustment(img, factor=1.5):
    """Brightness scaling in HSV value channel."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def attack_speckle_noise(img, sigma=0.01):
    """Speckle (multiplicative) noise, common in radar/medical imaging."""
    noise = np.random.normal(0, sigma, img.shape)
    out = img.astype(np.float64) + img.astype(np.float64) * noise
    return np.clip(out, 0, 255).astype(np.uint8)

def attack_gamma_correction(img, gamma=1.2):
    """Gamma correction for nonlinear brightness transformation."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def attack_jpeg2000(img):
    """
    JPEG2000 compression, an important robustness test in the paper [cite: 534, 549, 851].
    Compression ratio (CR) is typically in the range 2 to 10 [cite: 549].
    """
    # OpenCV supports .jp2 encoding if jasper/openjpeg is available.
    # Note: OpenCV does not expose 'compression_ratio' directly for JP2 like JPEG quality,
    # so this is approximated with a basic encode/decode cycle.
    _, encimg = cv2.imencode('.jp2', img)
    return cv2.imdecode(encimg, 1)

def attack_histogram_equalization(img):
    """
    Histogram equalization to test robustness under illumination redistribution [cite: 493, 625, 850].
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # Apply only on the Y (luminance) channel to limit color distortion
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)

def attack_histogram_equalization_blend(img, blend=0.5):
    """Blend original luminance with equalized luminance."""
    blend = float(np.clip(blend, 0.0, 1.0))
    eq = attack_histogram_equalization(img)
    return cv2.addWeighted(eq, blend, img, 1.0 - blend, 0)


def get_attack_scenarios():
    """Standard attack scenarios used for evaluation."""
    return [
        ("Original_No_Attack", lambda x: x),
        ("No Attack", lambda x: x),
        ("JPEG_Q90", lambda x: attack_jpeg(x, 90)),
        ("JPEG_QF90", lambda x: attack_jpeg(x, 90)),
        ("JPEG_Q80", lambda x: attack_jpeg(x, 80)),
        ("JPEG_Q70", lambda x: attack_jpeg(x, 70)),
        ("ZaloCompress_Q85_S0.75", lambda x: attack_zalo_compress(x, 85, 0.75)),
        ("SaltPepper_0.01", lambda x: attack_salt_pepper(x, 0.01)),
        ("SaltPepperNoise_0.01", lambda x: attack_salt_pepper(x, 0.01)),
        ("SaltPepper_0.05", lambda x: attack_salt_pepper(x, 0.05)),
        ("SaltPepper_0.1", lambda x: attack_salt_pepper(x, 0.1)),
        ("Sharpening_0.01", lambda x: attack_sharpening(x, 0.01)),
        ("Average_3x3", lambda x: attack_average_filter(x, 3)),
        ("AverageFiltering_3x3", lambda x: attack_average_filter(x, 3)),
        ("Median_3x3", lambda x: attack_median_filter(x, 3)),
        ("MedianFiltering_3x3", lambda x: attack_median_filter(x, 3)),
        ("Gaussian_3x3_s0.1", lambda x: attack_gaussian_filter(x, 3, 0.1)),
        ("GaussianBlur", lambda x: attack_gaussian_filter(x, 3, 0.8)),
        ("Gaussian_3x3_s0.2", lambda x: attack_gaussian_filter(x, 3, 0.2)),
        ("Gaussian_3x3_s0.5", lambda x: attack_gaussian_filter(x, 3, 0.5)),
        ("Gaussian_3x3_s0.7", lambda x: attack_gaussian_filter(x, 3, 0.7)),
        ("Gaussian_3x3_s0.9", lambda x: attack_gaussian_filter(x, 3, 0.9)),
        ("Gaussian_5x5_s0.1", lambda x: attack_gaussian_filter(x, 5, 0.1)),
        ("Gaussian_5x5_s0.2", lambda x: attack_gaussian_filter(x, 5, 0.2)),
        ("Gaussian_5x5_s0.5", lambda x: attack_gaussian_filter(x, 5, 0.5)),
        ("Gaussian_5x5_s0.7", lambda x: attack_gaussian_filter(x, 5, 0.7)),
        ("Gaussian_5x5_s0.9", lambda x: attack_gaussian_filter(x, 5, 0.9)),
        ("Gaussian_7x7_s0.1", lambda x: attack_gaussian_filter(x, 7, 0.1)),
        ("Gaussian_7x7_s0.2", lambda x: attack_gaussian_filter(x, 7, 0.2)),
        ("Gaussian_7x7_s0.5", lambda x: attack_gaussian_filter(x, 7, 0.5)),
        ("Gaussian_7x7_s0.7", lambda x: attack_gaussian_filter(x, 7, 0.7)),
        ("Gaussian_7x7_s0.9", lambda x: attack_gaussian_filter(x, 7, 0.9)),
        ("Rotate_1deg", lambda x: attack_rotation(x, 1)),
        ("Rotate_1.5deg", lambda x: attack_rotation(x, 1.5)),
        ("Rotate_2deg", lambda x: attack_rotation(x, 2)),
        ("Rotate_5deg", lambda x: attack_rotation(x, 5)),
        ("Rotate_10deg", lambda x: attack_rotation(x, 10)),
        ("Rotation_10", lambda x: attack_rotation(x, 10)),
        ("Rotate_20deg", lambda x: attack_rotation(x, 20)),
        ("Rotate_30deg", lambda x: attack_rotation(x, 30)),
        ("Rotate_45deg", lambda x: attack_rotation(x, 45)),
        ("Rotate_90deg", lambda x: attack_rotation(x, 90)),
        ("Cropping_10%", lambda x: attack_center_cropping(x, 10)),
        ("CornerCrop_25%", lambda x: attack_cropping_corner(x, 25)),
        ("CornerCrop_30%", lambda x: attack_cropping_corner(x, 30)),
        ("CornerCrop_35%", lambda x: attack_cropping_corner(x, 35)),
        ("CenterCrop_20%", lambda x: attack_center_cropping(x, 20)),
        ("CenterCrop_25%", lambda x: attack_center_cropping(x, 25)),
        ("CenterCrop_30%", lambda x: attack_center_cropping(x, 30)),
        ("Scale_0.9", lambda x: attack_scaling(x, 0.9)),
        ("Scale_0.7", lambda x: attack_scaling(x, 0.7)),
        ("Brightness_+50", lambda x: attack_brightness(x, 50)),
        ("Brightness_+60", lambda x: attack_brightness(x, 60)),
        ("Contrast_1.2", lambda x: attack_contrast(x, 1.2)),
        ("Contrast_1.4", lambda x: attack_contrast(x, 1.4)),
        ("BrightnessAdjustment_1.5", lambda x: attack_brightness_adjustment(x, 1.5)),
        ("Speckle_0.01", lambda x: attack_speckle_noise(x, 0.01)),
        ("Speckle_0.02", lambda x: attack_speckle_noise(x, 0.02)),
        ("Gamma_0.5", lambda x: attack_gamma_correction(x, 0.5)),
        ("Gamma_0.8", lambda x: attack_gamma_correction(x, 0.8)),
        ("JPEG2000", lambda x: attack_jpeg2000(x)),
        ("HistogramEqualization_0.5", lambda x: attack_histogram_equalization_blend(x, 0.5)),
        ("HistogramEqualization", lambda x: attack_histogram_equalization(x))
    ]


def optimize_alpha_bayesian(host_paths, wm_raw_path, MODE, text_input, id_input, repeat_k, payload_repeat, text_encoding, N, seed,
                            use_affine_correction=False, alpha_min=40.0, alpha_max=180.0,
                            n_trials=30, psnr_min=30.0, ssim_min=None, nc_min=0.85,
                            weight_psnr=0.45, weight_nc=0.55, weight_ber=0.20,
                            psnr_ref=50.0, psnr_penalty=0.05, nc_penalty=3.0,
                            ssim_penalty=8.0,
                            attack_names=None, random_seed=42):
    """Optimize alpha with Bayesian Optimization (Optuna/TPE)."""
    try:
        optuna = importlib.import_module("optuna")
    except ImportError:
        raise ImportError("Missing optuna package. Install it with: pip install optuna")

    if isinstance(host_paths, str):
        host_paths = [host_paths]
    host_paths = list(host_paths)
    if len(host_paths) == 0:
        raise ValueError("host_paths is empty.")

    if MODE == "text":
        wm_bin_raw = text_to_bin_image(
            text_input,
            repeat_k=repeat_k,
            payload_repeat=payload_repeat,
            encoding=text_encoding,
        )
        wm_raw = wm_bin_raw * 255
    elif MODE == "id":
        wm_bin_raw = id_to_bin_image(id_input, repeat_k=repeat_k, payload_repeat=payload_repeat)
        wm_raw = wm_bin_raw * 255
    else:
        wm_raw = cv2.imread(wm_raw_path, cv2.IMREAD_GRAYSCALE)

    if wm_raw is None:
        raise ValueError("Unable to read input watermark.")

    orig_h, orig_w = wm_raw.shape
    wm_padded, _ = pad_watermark(wm_raw, color=255)
    wm_orig_bin = (wm_raw > 127).astype(np.uint8)
    wm = (wm_padded > 127).astype(np.uint8)

    all_attacks = get_attack_scenarios()
    if attack_names:
        selected_attacks = [(name, fn) for name, fn in all_attacks if name in set(attack_names)]
        if len(selected_attacks) == 0:
            raise ValueError("No valid attack found in attack_names.")
    else:
        selected_attacks = [
            ("Original_No_Attack", lambda x: x),
            ("JPEG_Q70", lambda x: attack_jpeg(x, 70)),
            ("JPEG_Q50", lambda x: attack_jpeg(x, 50)),
            ("SaltPepper_0.05", lambda x: attack_salt_pepper(x, 0.05)),
            ("Rotate_5deg", lambda x: attack_rotation(x, 5)),
            ("CenterCrop_25%", lambda x: attack_center_cropping(x, 25)),
        ]

    weight_sum = max(weight_psnr + weight_nc, 1e-12)
    weight_psnr_norm = weight_psnr / weight_sum
    weight_nc_norm = weight_nc / weight_sum

    def objective(trial):
        alpha_val = trial.suggest_float("alpha", alpha_min, alpha_max)

        # Fix per-trial randomness so BO compares alpha values fairly.
        trial_seed = int(random_seed + trial.number * 1009)
        np.random.seed(trial_seed)

        psnr_vals, ssim_vals, nc_vals, ber_vals = [], [], [], []
        valid_runs = 0

        for host_path in host_paths:
            host = cv2.imread(host_path)
            if host is None:
                continue

            wat, _, pos, safe_des_orig, safe_kp_orig = embed_watermark(host, wm, alpha_val, N, seed)
            if len(pos) == 0:
                continue

            for attack_idx, (_, attack_fn) in enumerate(selected_attacks):
                np.random.seed(trial_seed + attack_idx)
                att_img = attack_fn(wat)
                ext_padded = extract_watermark(
                    att_img,
                    safe_des_orig,
                    safe_kp_orig,
                    alpha_val,
                    pos,
                    wm.shape,
                    seed=seed,
                    use_affine_correction=use_affine_correction,
                )
                ext = ext_padded[:orig_h, :orig_w]
                psnr, ssim, nc, ber = evaluate(host, att_img, wm_orig_bin, ext)
                psnr_vals.append(psnr)
                ssim_vals.append(ssim)
                nc_vals.append(nc)
                ber_vals.append(ber)
                valid_runs += 1

        if valid_runs == 0:
            return -1e6

        avg_psnr = float(np.mean(psnr_vals))
        avg_ssim = float(np.mean(ssim_vals))
        avg_nc = float(np.mean(nc_vals))
        avg_ber = float(np.mean(ber_vals))

        norm_psnr = float(np.clip(avg_psnr / max(psnr_ref, 1e-12), 0.0, 1.0))
        norm_nc = float(np.clip(avg_nc, 0.0, 1.0))

        penalty = (
            max(0.0, psnr_min - avg_psnr) * psnr_penalty
            + max(0.0, nc_min - avg_nc) * nc_penalty
        )
        if ssim_min is not None:
            penalty += max(0.0, ssim_min - avg_ssim) * ssim_penalty

        score = (
            weight_psnr_norm * norm_psnr
            + weight_nc_norm * norm_nc
            - weight_ber * avg_ber
            - penalty
        )

        trial.set_user_attr("avg_psnr", avg_psnr)
        trial.set_user_attr("avg_ssim", avg_ssim)
        trial.set_user_attr("avg_nc", avg_nc)
        trial.set_user_attr("avg_ber", avg_ber)
        trial.set_user_attr("norm_psnr", norm_psnr)
        trial.set_user_attr("norm_nc", norm_nc)
        trial.set_user_attr("penalty", penalty)

        return score

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_alpha = float(study.best_params["alpha"])
    best_score = float(study.best_value)
    best_trial = study.best_trial
    print(
        f"[*] BO done | best_alpha={best_alpha:.3f} | score={best_score:.4f} | "
        f"PSNR={best_trial.user_attrs.get('avg_psnr', 0.0):.3f} | "
        f"SSIM={best_trial.user_attrs.get('avg_ssim', 0.0):.4f} | "
        f"NC={best_trial.user_attrs.get('avg_nc', 0.0):.4f} | "
        f"BER={best_trial.user_attrs.get('avg_ber', 1.0):.4f} | "
        f"penalty={best_trial.user_attrs.get('penalty', 0.0):.4f}"
    )

    return best_alpha, best_score, study




# --- 7. EVALUATION FUNCTIONS ---

def repeat_bits(bit_str, repeat_k):
    if repeat_k < 1:
        raise ValueError("repeat_k must be >= 1.")
    if repeat_k == 1:
        return bit_str
    if repeat_k % 2 == 0:
        raise ValueError("repeat_k must be odd for majority voting.")
    return ''.join(b * repeat_k for b in bit_str)

def decode_repeated_bits(bits, repeat_k):
    if repeat_k < 1:
        raise ValueError("repeat_k must be >= 1.")
    if repeat_k == 1:
        return bits
    if repeat_k % 2 == 0:
        raise ValueError("repeat_k must be odd for majority voting.")

    decoded_bits = []
    for i in range(0, len(bits), repeat_k):
        group = bits[i:i + repeat_k]
        if len(group) < repeat_k:
            break
        decoded_bits.append(1 if np.sum(group) >= (repeat_k // 2 + 1) else 0)
    return decoded_bits

def bits_to_square_matrix(bit_array, min_side=32):
    num_bits = len(bit_array)
    side = int(np.ceil(np.sqrt(num_bits)))
    if side % 4 != 0:
        side = ((side // 4) + 1) * 4
    side = max(side, min_side)

    total_pixels = side * side
    if len(bit_array) < total_pixels and len(bit_array) > 0:
        reps = int(np.ceil(total_pixels / len(bit_array)))
        bit_array = np.tile(bit_array, reps)[:total_pixels]
    else:
        # If empty or already large enough, pad/truncate as before
        if len(bit_array) < total_pixels:
            bit_array = np.pad(bit_array, (0, total_pixels - len(bit_array)), 'constant')
        else:
            bit_array = bit_array[:total_pixels]

    return bit_array.reshape((side, side))

def get_text_payload_bits_len(text, encoding="utf-8"):
    payload = text.encode(encoding)
    return 16 + len(payload) * 8

def get_id_payload_bits_len(id_text):
    if id_text is None:
        raise ValueError("ID input is required for mode=id.")
    if len(id_text) == 0:
        raise ValueError("ID input cannot be empty.")
    if not id_text.isdigit():
        raise ValueError("ID input must contain digits only (0-9).")
    return len(id_text) * 4 + 4

def choose_repeat_k_to_fill(base_bits_len, repeat_k, min_side=32, max_k=None):
    if base_bits_len <= 0:
        return repeat_k
    target_bits = min_side * min_side
    if base_bits_len >= target_bits:
        return repeat_k

    max_fit = target_bits // base_bits_len
    if max_fit < 1:
        return repeat_k
    if max_fit % 2 == 0:
        max_fit -= 1
    if max_fit < 1:
        return repeat_k
    if max_k is not None:
        max_k = max_k if max_k % 2 == 1 else max_k - 1
        if max_k >= 1:
            max_fit = min(max_fit, max_k)

    if max_fit >= repeat_k:
        return max_fit
    return repeat_k

def choose_payload_repeat_to_fill(base_bits_len, repeat_k, payload_repeat, min_side=32, max_repeat=None):
    if base_bits_len <= 0 or repeat_k <= 0:
        return payload_repeat
    target_bits = min_side * min_side
    bits_per_copy = base_bits_len * repeat_k
    if bits_per_copy >= target_bits:
        return payload_repeat

    max_fit = target_bits // bits_per_copy
    if max_fit < 1:
        return payload_repeat
    if max_repeat is not None:
        max_fit = min(max_fit, max_repeat)

    return max(payload_repeat, max_fit)

def bytes_to_bit_string(payload):
    return ''.join(format(b, '08b') for b in payload)

def bit_string_to_bytes(bit_str):
    data = []
    for i in range(0, len(bit_str), 8):
        chunk = bit_str[i:i + 8]
        if len(chunk) < 8:
            break
        data.append(int(chunk, 2))
    return bytes(data)

def text_to_payload_bits(text, encoding="utf-8"):
    payload = text.encode(encoding)
    payload_len = len(payload)
    if payload_len > 65535:
        raise ValueError("Text payload too long for 16-bit length header.")
    header = format(payload_len, '016b')
    return header + bytes_to_bit_string(payload)

def id_to_payload_bits(id_text):
    if id_text is None:
        raise ValueError("ID input is required for mode=id.")
    if len(id_text) == 0:
        raise ValueError("ID input cannot be empty.")
    if not id_text.isdigit():
        raise ValueError("ID input must contain digits only (0-9).")
    digits = [int(ch) for ch in id_text]
    bits = ''.join(format(d, '04b') for d in digits)
    return bits + '1111'

def payload_bits_from_bin_image_text(bin_img, repeat_k=3, payload_repeat=1):
    bits = bin_img.flatten()
    decoded_bits = decode_repeated_bits(bits, repeat_k)
    bit_str = ''.join(map(str, decoded_bits))
    if payload_repeat < 1 or len(bit_str) < 16:
        return ""

    header_bits = bit_str[:16]
    payload_len = int(header_bits, 2)
    copy_len = 16 + payload_len * 8
    if payload_repeat > 1 and copy_len > 0:
        bit_str = vote_payload_copies(bit_str, copy_len, payload_repeat)
    return bit_str[:copy_len]

def payload_bits_from_bin_image_id(bin_img, repeat_k=1, payload_repeat=1):
    bits = bin_img.flatten()
    decoded_bits = decode_repeated_bits(bits, repeat_k)
    bit_str = ''.join(map(str, decoded_bits))
    if payload_repeat < 1:
        return ""

    _, copy_len = decode_id_bits(bit_str)
    if payload_repeat > 1 and copy_len > 0:
        bit_str = vote_payload_copies(bit_str, copy_len, payload_repeat)
    return bit_str[:copy_len]

def payload_ber(orig_bits, extracted_bits):
    if orig_bits is None:
        orig_bits = ""
    if extracted_bits is None:
        extracted_bits = ""
    n = len(orig_bits)
    if n == 0:
        return 0.0
    compare_len = min(n, len(extracted_bits))
    errors = sum(1 for i in range(compare_len) if orig_bits[i] != extracted_bits[i])
    missing = n - compare_len
    return float(errors + missing) / float(n)

def text_to_bin_image(text, repeat_k=3, payload_repeat=1, encoding="utf-8"):
    """Convert text to bits (UTF-8 by default) and repeat each bit for robustness."""
    payload = text.encode(encoding)
    payload_len = len(payload)
    if payload_len > 65535:
        raise ValueError("Text payload too long for 16-bit length header.")

    header = format(payload_len, '016b')
    bits = header + bytes_to_bit_string(payload)
    if payload_repeat < 1:
        raise ValueError("payload_repeat must be >= 1.")
    if payload_repeat > 1:
        bits = bits * payload_repeat

    robust_bits = repeat_bits(bits, repeat_k)
    bit_array = np.array([int(b) for b in robust_bits], dtype=np.uint8)
    
    # 3. Compute a suitable square matrix size
    num_bits = len(bit_array)

    print(
        f"[*] Text '{text[:10]}...' -> embed {len(bits)} raw bits (total protected bits: {num_bits}) "
        f"| encoding={encoding} | payload_repeat={payload_repeat}"
    )
    return bits_to_square_matrix(bit_array)

def bin_image_to_text(bin_img, repeat_k=3, payload_repeat=1, encoding="utf-8"):
    """Decode bits using majority voting over repeated groups."""
    bits = bin_img.flatten()
    decoded_bits = decode_repeated_bits(bits, repeat_k)

    bit_str = ''.join(map(str, decoded_bits))
    if len(bit_str) < 16:
        return ""

    if payload_repeat < 1:
        return ""

    if payload_repeat > 1:
        header_bits = bit_str[:16]
        payload_len = int(header_bits, 2)
        copy_len = 16 + payload_len * 8
        if copy_len > 0:
            bit_str = vote_payload_copies(bit_str, copy_len, payload_repeat)

    if len(bit_str) < 16:
        return ""

    header_bits = bit_str[:16]
    payload_len = int(header_bits, 2)
    payload_bits = bit_str[16:16 + payload_len * 8]
    payload = bit_string_to_bytes(payload_bits)
    return payload.decode(encoding, errors="replace")

def id_to_bin_image(id_text, repeat_k=1, payload_repeat=1):
    """Convert numeric ID to 4-bit digits, optionally with repetition."""
    if id_text is None:
        raise ValueError("ID input is required for mode=id.")
    if len(id_text) == 0:
        raise ValueError("ID input cannot be empty.")
    if not id_text.isdigit():
        raise ValueError("ID input must contain digits only (0-9).")

    digits = [int(ch) for ch in id_text]
    bits = ''.join(format(d, '04b') for d in digits)

    # Terminator nibble (0xF) so decoder knows where to stop.
    bits += '1111'

    if payload_repeat < 1:
        raise ValueError("payload_repeat must be >= 1.")
    if payload_repeat > 1:
        bits = bits * payload_repeat

    robust_bits = repeat_bits(bits, repeat_k)
    bit_array = np.array([int(b) for b in robust_bits], dtype=np.uint8)
    print(
        f"[*] ID '{id_text[:10]}...' -> embed {len(bits)} raw bits (total protected bits: {len(bit_array)}) "
        f"| payload_repeat={payload_repeat}"
    )
    return bits_to_square_matrix(bit_array)

def bin_image_to_id(bin_img, repeat_k=1, payload_repeat=1):
    """Decode 4-bit digits until terminator nibble (0xF)."""
    bits = bin_img.flatten()
    decoded_bits = decode_repeated_bits(bits, repeat_k)

    bit_str = ''.join(map(str, decoded_bits))
    if payload_repeat < 1:
        return ""

    decoded_id, copy_len = decode_id_bits(bit_str)
    if payload_repeat > 1 and copy_len > 0:
        bit_str = vote_payload_copies(bit_str, copy_len, payload_repeat)
        decoded_id, _ = decode_id_bits(bit_str)
    return decoded_id

def decode_id_bits(bit_str):
    digits = []
    used_len = 0
    for i in range(0, len(bit_str), 4):
        nibble = bit_str[i:i + 4]
        if len(nibble) < 4:
            break
        value = int(nibble, 2)
        used_len = i + 4
        if value == 15:
            break
        if value > 9:
            break
        digits.append(str(value))
    return "".join(digits), used_len

def vote_payload_copies(bit_str, copy_len, payload_repeat):
    copies = []
    for i in range(payload_repeat):
        start = i * copy_len
        end = start + copy_len
        if end > len(bit_str):
            break
        copies.append(bit_str[start:end])
    if len(copies) == 0:
        return ""
    if len(copies) == 1:
        return copies[0]

    voted_bits = []
    for i in range(copy_len):
        ones = sum(1 for c in copies if c[i] == '1')
        zeros = len(copies) - ones
        voted_bits.append('1' if ones >= zeros else '0')
    return ''.join(voted_bits)

def sanitize_excel_text(value):
    if value is None:
        return value
    text = str(value)
    return ''.join(ch for ch in text if ch >= ' ' or ch in '\t\n\r')

def evaluate(orig, wat, wm, ext_wm):
    psnr = cv2.PSNR(orig, wat)
    ssim_val = ssim_func(orig, wat, channel_axis=2)

    wm_bin = (wm > 0).astype(np.float64)
    ext_bin = (ext_wm > 0).astype(np.float64)

    numerator = np.sum(wm_bin * ext_bin)
    denominator = np.sqrt(np.sum(wm_bin ** 2) * np.sum(ext_bin ** 2))
    nc = numerator / (denominator + 1e-12)

    ber = np.sum(wm_bin != ext_bin) / wm_bin.size

    return psnr, ssim_val, nc, ber


def char_accuracy(orig_text, extracted_text):
    """Compute character-level accuracy using Levenshtein distance.
    Returns fraction in [0.0, 1.0]."""
    if orig_text is None:
        orig_text = ""
    if extracted_text is None:
        extracted_text = ""
    s = str(orig_text)
    t = str(extracted_text)
    n = len(s)
    m = len(t)
    if n == 0 and m == 0:
        return 1.0
    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    lev = dp[n][m]
    maxlen = max(n, 1)
    acc = max(0.0, 1.0 - float(lev) / float(maxlen))
    return acc


def estimate_bit_error_rate(host_paths, wm_bin, alpha_val, N, seed,
                            use_affine_correction=False, attack_names=None):
    if isinstance(host_paths, str):
        host_paths = [host_paths]
    host_paths = list(host_paths)
    if len(host_paths) == 0:
        raise ValueError("host_paths is empty.")

    all_attacks = get_attack_scenarios()
    if attack_names:
        selected_attacks = [(name, fn) for name, fn in all_attacks if name in set(attack_names)]
        if len(selected_attacks) == 0:
            raise ValueError("No valid attack found in attack_names.")
    else:
        selected_attacks = [
            ("JPEG_Q70", lambda x: attack_jpeg(x, 70)),
            ("SaltPepper_0.05", lambda x: attack_salt_pepper(x, 0.05)),
            ("Rotate_5deg", lambda x: attack_rotation(x, 5)),
        ]

    ber_vals = []
    for host_path in host_paths:
        host = cv2.imread(host_path)
        if host is None:
            continue
        wat, _, pos, safe_des_orig, safe_kp_orig = embed_watermark(host, wm_bin, alpha_val, N, seed)
        if len(pos) == 0:
            continue
        for attack_name, attack_fn in selected_attacks:
            att_img = attack_fn(wat)
            ext_padded = extract_watermark(
                att_img,
                safe_des_orig,
                safe_kp_orig,
                alpha_val,
                pos,
                wm_bin.shape,
                seed=seed,
                use_affine_correction=use_affine_correction,
            )
            ber = np.sum(wm_bin != (ext_padded > 0).astype(np.uint8)) / wm_bin.size
            ber_vals.append(float(ber))

    if len(ber_vals) == 0:
        return None
    return float(np.mean(ber_vals))


def majority_success_prob(k, ber):
    success = 0.0
    threshold = k // 2 + 1
    for i in range(threshold, k + 1):
        success += math.comb(k, i) * ((1 - ber) ** i) * (ber ** (k - i))
    return success


def choose_repeat_k_from_ber(ber, target_success=0.99, max_k=7):
    if ber is None:
        return 3
    if ber <= 0.0:
        return 1
    if ber >= 0.5:
        return max_k

    for k in range(1, max_k + 1, 2):
        if majority_success_prob(k, ber) >= target_success:
            return k
    return max_k


def pad_watermark(wm, color=255):
    h, w = wm.shape[:2]
    side = max(h, w)
    if side % 4 != 0:
        side = ((side // 4) + 1) * 4
    pad_h = side - h
    pad_w = side - w
    padded_wm = cv2.copyMakeBorder(wm, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_CONSTANT, value=color)
    return padded_wm, (h, w)


# --- 8. MAIN EXECUTION ---

def main(host_path, wm_raw_path, MODE, text_input, id_input, repeat_k, payload_repeat, text_encoding, alpha_val, N, seed, use_affine_correction=False):
    host = cv2.imread(host_path)
    
    # Host image name (without extension)
    host_name = os.path.splitext(os.path.basename(host_path))[0]
    
    # Create result/<host_name>/attack and result/<host_name>/recover folders
    result_base = os.path.join("results", host_name)
    attack_dir = os.path.join(result_base, "attack")
    recover_dir = os.path.join(result_base, "recover")
    
    os.makedirs(attack_dir, exist_ok=True)
    os.makedirs(recover_dir, exist_ok=True)
    print(f"[+] Created directory: {attack_dir}")
    print(f"[+] Created directory: {recover_dir}")
    print(f"[*] Affine correction mode: {'ON' if use_affine_correction else 'OFF'}")

    if MODE == "text":
        # Convert input text into a binary watermark image
        wm_bin_raw = text_to_bin_image(
            text_input,
            repeat_k=repeat_k,
            payload_repeat=payload_repeat,
            encoding=text_encoding,
        )
        wm_raw = wm_bin_raw * 255
    elif MODE == "id":
        # Convert numeric ID into a binary watermark image
        wm_bin_raw = id_to_bin_image(id_input, repeat_k=repeat_k, payload_repeat=payload_repeat)
        wm_raw = wm_bin_raw * 255
    else:
        wm_raw = cv2.imread(wm_raw_path, cv2.IMREAD_GRAYSCALE)

    if host is None or wm_raw is None:
        print("Error: input file not found.")
    else:
        # --- STEP 1: WATERMARK PADDING ---
        # Keep original size for later cropping

        orig_h, orig_w = wm_raw.shape
        # Pad watermark so dimensions are divisible by 4
        wm_padded, _ = pad_watermark(wm_raw, color=255)
        
        wm_orig_bin = (wm_raw > 127).astype(np.uint8)
        wm = (wm_padded > 127).astype(np.uint8)
        cv2.imwrite("results/watermark_binary.jpg", (wm * 255).astype(np.uint8))
        # wm = wm_raw
        if wm_padded.shape != wm_raw.shape:
            print(f"[*] Watermark padded: {wm_raw.shape} -> {wm_padded.shape}")

        print(f"Embedded watermark size: {wm.shape}")
        
        print("--- Running watermark embedding ---")
        # alpha_val = 100.0  # Embedding strength (higher value improves robustness)
        # cv2.imwrite('wtm.jpg', wm)
        wat, _, pos, safe_des_orig, safe_kp_orig = embed_watermark(host, wm, alpha_val, N, seed)
        key_path = os.path.join("results", f"{host_name}.npz")
        save_key_file(
            key_path,
            pos=pos,
            safe_kp_orig=safe_kp_orig,
            safe_des_orig=safe_des_orig,
            wm_shape=wm.shape,
            orig_shape=(orig_h, orig_w),
            alpha=alpha_val,
            seed=seed,
            mode=MODE,
            text_input=text_input,
            id_input=id_input,
            repeat_k=repeat_k,
            payload_repeat=payload_repeat,
            text_encoding=text_encoding,
            host_path=host_path,
            wm_raw_path=wm_raw_path,
        )
        print(f"[*] Key saved: {key_path}")
        extract_dir = os.path.join("data", "host_images_extract")
        os.makedirs(extract_dir, exist_ok=True)
        host_filename = os.path.basename(host_path)
        extract_path = os.path.join(extract_dir, host_filename)
        cv2.imwrite(extract_path, wat)
        print(f"[*] Saved no-attack image: {extract_path}")

        attacks = get_attack_scenarios()

        print(f"\n{'Attack':<25} | {'PSNR':<8.2} | {'SSIM':<8.4} | {'NC':<8.4} | {'BER':<8.4}")
        print("-" * 55)

        results_list = []
        excel_final_path = "results/watermark_evaluation_summary.xlsx"
        for name, func in attacks:
            
            att_img = func(wat)
            cv2.imwrite(os.path.join(attack_dir, f"attack_{name}.png"), att_img)
            ext_padded = extract_watermark(
                att_img,
                safe_des_orig,
                safe_kp_orig,
                alpha_val,
                pos,
                wm.shape,
                seed=seed,
                use_affine_correction=use_affine_correction,
            )
            ext = ext_padded[:orig_h, :orig_w]
            psnr, ssim, nc, ber = evaluate(host, att_img, wm_orig_bin, ext)

            row = {
            "Attack": name,
            "Alpha": round(float(alpha_val), 4),
            "PSNR": round(psnr, 4),
            "SSIM": round(ssim, 4),
            "NC": round(nc, 4),
            "BER": round(ber, 4)
            }
            
            # Keep printed metrics normalized and consistent
            if MODE == "text":
                extracted_text = bin_image_to_text(
                    ext,
                    repeat_k=repeat_k,
                    payload_repeat=payload_repeat,
                    encoding=text_encoding,
                )
                extracted_text_clean = sanitize_excel_text(extracted_text)
                # Character-level accuracy vs input
                char_acc = char_accuracy(text_input, extracted_text_clean)
                orig_bits = text_to_payload_bits(text_input, encoding=text_encoding)
                extracted_bits = payload_bits_from_bin_image_text(
                    ext,
                    repeat_k=repeat_k,
                    payload_repeat=payload_repeat,
                )
                bit_ber = payload_ber(orig_bits, extracted_bits)
                row["ExtractedText"] = extracted_text_clean
                row["CharAcc"] = round(float(char_acc), 4)
                row["PayloadBER"] = round(float(bit_ber), 4)
                print(
                    f"{name:<25} | {psnr:<8.2f} | {ssim:<8.4f} | {nc:<8.4f} | {ber:<8.4f} | "
                    f"Extracted text: {extracted_text_clean} | CharAcc: {char_acc:.4f} | PayloadBER: {bit_ber:.4f}"
                )
            elif MODE == "id":
                extracted_id = bin_image_to_id(ext, repeat_k=repeat_k, payload_repeat=payload_repeat)
                extracted_id_clean = sanitize_excel_text(extracted_id)
                char_acc = char_accuracy(id_input, extracted_id_clean)
                orig_bits = id_to_payload_bits(id_input)
                extracted_bits = payload_bits_from_bin_image_id(
                    ext,
                    repeat_k=repeat_k,
                    payload_repeat=payload_repeat,
                )
                bit_ber = payload_ber(orig_bits, extracted_bits)
                row["ExtractedID"] = extracted_id_clean
                row["CharAcc"] = round(float(char_acc), 4)
                row["PayloadBER"] = round(float(bit_ber), 4)
                print(
                    f"{name:<25} | {psnr:<8.2f} | {ssim:<8.4f} | {nc:<8.4f} | {ber:<8.4f} | "
                    f"Extracted id: {extracted_id_clean} | CharAcc: {char_acc:.4f} | PayloadBER: {bit_ber:.4f}"
                )
            else:
                print(f"{name:<25} | {psnr:<8.2f} | {ssim:<8.4f} | {nc:<8.4f} | {ber:<8.4f}")
            
            results_list.append(row)
            
            # Scale recovered watermark for visualization
            cv2.imwrite(os.path.join(recover_dir, f"recovered_{name}.png"), ext * 255)

        df = pd.DataFrame(results_list)
    
            # Save to Excel
        if not os.path.exists(excel_final_path):
            # Create file if it does not exist
            with pd.ExcelWriter(excel_final_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=host_name, index=False)
        else:
            # Append mode, replacing existing sheet if present
            with pd.ExcelWriter(excel_final_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                # Overwrite the host sheet when rerunning the same image
                df.to_excel(writer, sheet_name=host_name, index=False)

        print("-" * 55)
        print(f"Results were saved to '{result_base}/'.")
        print(f"  - Attacked images: {attack_dir}")
        print(f"  - Recovered images: {recover_dir}")
        print(f"[*] Updated results for host image '{host_name}' in file: {excel_final_path}")


def extract_only(host_path, key_path, output_dir, use_affine_correction=False):
    host = cv2.imread(host_path)
    if host is None:
        raise ValueError(f"Unable to read image: {host_path}")

    key = load_key_file(key_path)
    if not key["wm_shape"] or not key["orig_shape"]:
        raise ValueError(f"Key missing wm_shape/orig_shape: {key_path}")

    os.makedirs(output_dir, exist_ok=True)
    attack_dir = os.path.join(output_dir, "attack")
    recover_dir = os.path.join(output_dir, "recover")
    os.makedirs(attack_dir, exist_ok=True)
    os.makedirs(recover_dir, exist_ok=True)

    orig_host = None
    if key["host_path"]:
        orig_host = cv2.imread(key["host_path"])
        if orig_host is None:
            print(f"[!] Original host not found: {key['host_path']}")

    if key["mode"] == "text":
        if not key.get("text_input"):
            raise ValueError("Missing text_input in key file.")
        wm_bin_raw = text_to_bin_image(
            key.get("text_input", ""),
            repeat_k=key["repeat_k"],
            payload_repeat=key["payload_repeat"],
            encoding=key["text_encoding"],
        )
        wm_orig_bin = wm_bin_raw
    elif key["mode"] == "id":
        if not key.get("id_input"):
            raise ValueError("Missing id_input in key file.")
        wm_bin_raw = id_to_bin_image(
            key.get("id_input", ""),
            repeat_k=key["repeat_k"],
            payload_repeat=key["payload_repeat"],
        )
        wm_orig_bin = wm_bin_raw
    else:
        wm_raw_path = key.get("wm_raw_path", "")
        wm_raw = cv2.imread(wm_raw_path, cv2.IMREAD_GRAYSCALE) if wm_raw_path else None
        if wm_raw is None:
            raise ValueError(f"Unable to read watermark image: {wm_raw_path}")
        wm_orig_bin = (wm_raw > 127).astype(np.uint8)

    orig_h, orig_w = key["orig_shape"]

    print(f"[*] Affine correction mode: {'ON' if use_affine_correction else 'OFF'}")
    print(f"\n{'Attack':<25} | {'PSNR':<8.2} | {'SSIM':<8.4} | {'NC':<8.4} | {'BER':<8.4}")
    print("-" * 55)

    results_list = []
    attacks = get_attack_scenarios()
    for name, func in attacks:
        att_img = func(host)
        cv2.imwrite(os.path.join(attack_dir, f"attack_{name}.png"), att_img)
        ext_padded = extract_watermark(
            att_img,
            key["safe_des_orig"],
            key["safe_kp_orig"],
            key["alpha"],
            key["pos"],
            key["wm_shape"],
            seed=key["seed"],
            use_affine_correction=use_affine_correction,
        )
        ext = ext_padded[:orig_h, :orig_w]

        if orig_host is not None:
            psnr, ssim, nc, ber = evaluate(orig_host, att_img, wm_orig_bin, ext)
        else:
            psnr, ssim, nc, ber = 0.0, 0.0, 0.0, 1.0

        row = {
            "Attack": name,
            "Alpha": round(float(key["alpha"]), 4),
            "PSNR": round(psnr, 4),
            "SSIM": round(ssim, 4),
            "NC": round(nc, 4),
            "BER": round(ber, 4),
        }

        if key["mode"] == "text":
            extracted_text = bin_image_to_text(
                ext,
                repeat_k=key["repeat_k"],
                payload_repeat=key["payload_repeat"],
                encoding=key["text_encoding"],
            )
            extracted_text_clean = sanitize_excel_text(extracted_text)
            char_acc = char_accuracy(key.get("text_input", ""), extracted_text_clean)
            orig_bits = text_to_payload_bits(key.get("text_input", ""), encoding=key["text_encoding"])
            extracted_bits = payload_bits_from_bin_image_text(
                ext,
                repeat_k=key["repeat_k"],
                payload_repeat=key["payload_repeat"],
            )
            bit_ber = payload_ber(orig_bits, extracted_bits)
            row["ExtractedText"] = extracted_text_clean
            row["CharAcc"] = round(float(char_acc), 4)
            row["PayloadBER"] = round(float(bit_ber), 4)
            print(
                f"{name:<25} | {psnr:<8.2f} | {ssim:<8.4f} | {nc:<8.4f} | {ber:<8.4f} | "
                f"Extracted text: {extracted_text_clean} | CharAcc: {char_acc:.4f} | PayloadBER: {bit_ber:.4f}"
            )
        elif key["mode"] == "id":
            extracted_id = bin_image_to_id(
                ext,
                repeat_k=key["repeat_k"],
                payload_repeat=key["payload_repeat"],
            )
            extracted_id_clean = sanitize_excel_text(extracted_id)
            char_acc = char_accuracy(key.get("id_input", ""), extracted_id_clean)
            orig_bits = id_to_payload_bits(key.get("id_input", ""))
            extracted_bits = payload_bits_from_bin_image_id(
                ext,
                repeat_k=key["repeat_k"],
                payload_repeat=key["payload_repeat"],
            )
            bit_ber = payload_ber(orig_bits, extracted_bits)
            row["ExtractedID"] = extracted_id_clean
            row["CharAcc"] = round(float(char_acc), 4)
            row["PayloadBER"] = round(float(bit_ber), 4)
            print(
                f"{name:<25} | {psnr:<8.2f} | {ssim:<8.4f} | {nc:<8.4f} | {ber:<8.4f} | "
                f"Extracted id: {extracted_id_clean} | CharAcc: {char_acc:.4f} | PayloadBER: {bit_ber:.4f}"
            )
        else:
            print(f"{name:<25} | {psnr:<8.2f} | {ssim:<8.4f} | {nc:<8.4f} | {ber:<8.4f}")

        results_list.append(row)
        cv2.imwrite(os.path.join(recover_dir, f"recovered_{name}.png"), ext * 255)

    df = pd.DataFrame(results_list)
    metrics_path = os.path.join(output_dir, "extract_metrics.xlsx")
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="extract", index=False)

    print("-" * 55)
    print(f"Results were saved to '{output_dir}/'.")
    print(f"  - Attacked images: {attack_dir}")
    print(f"  - Recovered images: {recover_dir}")
    print(f"[*] Extract metrics saved: {metrics_path}")
    return metrics_path