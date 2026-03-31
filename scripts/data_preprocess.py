from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    output_size: tuple[int, int] = (512, 512)
    padding_color: tuple[int, int, int] = (235, 235, 235)
    crop_margin_ratio: float = 0.08
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    morph_kernel_size: int = 9
    min_component_area_ratio: float = 0.01
    hough_threshold: int = 80
    hough_min_line_length_ratio: float = 0.15
    hough_max_line_gap: int = 20
    debug: bool = True


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    return image


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def build_edge_map(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(
        blurred,
        threshold1=cfg.canny_threshold1,
        threshold2=cfg.canny_threshold2,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (cfg.morph_kernel_size, cfg.morph_kernel_size),
    )
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges_closed = cv2.dilate(edges_closed, kernel, iterations=1)
    return edges_closed


def largest_useful_component(
    binary_map: np.ndarray,
    image_shape: tuple[int, int],
    cfg: PreprocessConfig,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

    h, w = image_shape
    min_area = int(h * w * cfg.min_component_area_ratio)

    best_mask = None
    best_score = -1.0

    for label_idx in range(1, num_labels):
        x, y, bw, bh, area = stats[label_idx]

        if area < min_area:
            continue

        aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
        score = area * (1.0 + 0.15 * aspect_ratio)

        component_mask = (labels == label_idx).astype(np.uint8) * 255

        if score > best_score:
            best_score = score
            best_mask = component_mask

    if best_mask is None:
        # fallback: usar todo el mapa binario si no se detectó componente fuerte
        return binary_map.copy()

    return best_mask


def estimate_angle_hough(
    binary_mask: np.ndarray,
    cfg: PreprocessConfig,
) -> float | None:
    h, w = binary_mask.shape
    min_line_length = int(max(h, w) * cfg.hough_min_line_length_ratio)

    lines = cv2.HoughLinesP(
        binary_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=cfg.hough_max_line_gap,
    )

    if lines is None or len(lines) == 0:
        return None

    angles = []
    lengths = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        if length < 1:
            continue

        angle = math.degrees(math.atan2(dy, dx))

        # normalizar a rango [-90, 90)
        if angle >= 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        angles.append(angle)
        lengths.append(length)

    if not angles:
        return None

    angles_array = np.array(angles, dtype=np.float32)
    lengths_array = np.array(lengths, dtype=np.float32)

    # promedio ponderado por longitud
    weighted_angle = float(np.average(angles_array, weights=lengths_array))
    return weighted_angle


def estimate_angle_pca(binary_mask: np.ndarray) -> float | None:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) < 20:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors, _ = cv2.PCACompute2(points, mean=None)

    vx, vy = eigenvectors[0]
    angle = math.degrees(math.atan2(vy, vx))

    if angle >= 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return float(angle)


def estimate_dominant_angle(binary_mask: np.ndarray, cfg: PreprocessConfig) -> float:
    angle_hough = estimate_angle_hough(binary_mask, cfg)
    angle_pca = estimate_angle_pca(binary_mask)

    if angle_hough is not None and angle_pca is not None:
        # combinación simple
        return 0.7 * angle_hough + 0.3 * angle_pca

    if angle_hough is not None:
        return angle_hough

    if angle_pca is not None:
        return angle_pca

    return 0.0


def rotate_image_keep_bounds(image: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])

    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(235, 235, 235),
    )
    return rotated, rotation_matrix


def transform_mask(mask: np.ndarray, rotation_matrix: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    transformed = cv2.warpAffine(
        mask,
        rotation_matrix,
        (out_shape[1], out_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return transformed


def bounding_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape
        return 0, 0, w, h

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def expand_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    img_h, img_w = image_shape

    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    return x1, y1, x2 - x1, y2 - y1


def crop_image(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return image[y:y + h, x:x + w]


def resize_and_pad(
    image: np.ndarray,
    output_size: tuple[int, int],
    padding_color: tuple[int, int, int],
) -> np.ndarray:
    target_h, target_w = output_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def draw_box(image: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> np.ndarray:
    x, y, w, h = box
    vis = image.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
    return vis


def preprocess_image(
    image_bgr: np.ndarray,
    cfg: PreprocessConfig,
) -> dict[str, np.ndarray | float | tuple[int, int, int, int]]:
    gray = to_gray(image_bgr)
    edge_map = build_edge_map(gray, cfg)

    initial_mask = largest_useful_component(edge_map, gray.shape, cfg)
    initial_box = expand_box(
        bounding_box_from_mask(initial_mask),
        gray.shape,
        cfg.crop_margin_ratio,
    )

    angle = estimate_dominant_angle(initial_mask, cfg)

    # Rotamos para alinear el eje principal horizontalmente.
    rotated_image, rot_matrix = rotate_image_keep_bounds(image_bgr, -angle)
    rotated_mask = transform_mask(initial_mask, rot_matrix, rotated_image.shape[:2])

    final_mask = largest_useful_component(rotated_mask, rotated_mask.shape, cfg)
    final_box = expand_box(
        bounding_box_from_mask(final_mask),
        rotated_image.shape[:2],
        cfg.crop_margin_ratio,
    )

    cropped = crop_image(rotated_image, final_box)
    final_preprocessed = resize_and_pad(cropped, cfg.output_size, cfg.padding_color)

    result: dict[str, np.ndarray | float | tuple[int, int, int, int]] = {
        "gray": gray,
        "edge_map": edge_map,
        "initial_mask": initial_mask,
        "initial_box_vis": draw_box(image_bgr, initial_box, (0, 255, 0)),
        "angle_deg": angle,
        "rotated_image": rotated_image,
        "rotated_mask": rotated_mask,
        "final_box_vis": draw_box(rotated_image, final_box, (0, 0, 255)),
        "cropped": cropped,
        "final_preprocessed": final_preprocessed,
        "initial_box": initial_box,
        "final_box": final_box,
    }
    return result


def save_debug_outputs(
    outputs: dict[str, np.ndarray | float | tuple[int, int, int, int]],
    out_dir: str | Path,
    stem: str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_keys = [
        "gray",
        "edge_map",
        "initial_mask",
        "initial_box_vis",
        "rotated_image",
        "rotated_mask",
        "final_box_vis",
        "cropped",
        "final_preprocessed",
    ]

    for key in image_keys:
        value = outputs[key]
        if not isinstance(value, np.ndarray):
            continue

        out_path = out_dir / f"{stem}_{key}.png"
        cv2.imwrite(str(out_path), value)

    angle = outputs["angle_deg"]
    with open(out_dir / f"{stem}_meta.txt", "w", encoding="utf-8") as f:
        f.write(f"angle_deg: {angle}\n")
        f.write(f"initial_box: {outputs['initial_box']}\n")
        f.write(f"final_box: {outputs['final_box']}\n")


def process_one_image(
    image_path: str | Path,
    out_dir: str | Path,
    cfg: PreprocessConfig | None = None,
) -> dict[str, np.ndarray | float | tuple[int, int, int, int]]:
    if cfg is None:
        cfg = PreprocessConfig()

    image = load_image(image_path)
    outputs = preprocess_image(image, cfg)
    save_debug_outputs(outputs, out_dir, Path(image_path).stem)
    return outputs


if __name__ == "__main__":
    cfg = PreprocessConfig(
        output_size=(512, 512),
        padding_color=(235, 235, 235),
        crop_margin_ratio=0.10,
        debug=True,
    )

    image_path = "C:\Proyectos\KINECTRICS\Anomaly-Detection\data_raw_jpg\clean_v1\IMG_0667.jpg"
    out_dir = "debug_preprocess"

    outputs = process_one_image(image_path, out_dir, cfg)
    print("Ángulo estimado:", outputs["angle_deg"])