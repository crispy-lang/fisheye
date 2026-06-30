"""独立几何回归测试：验证 PCA 平面拟合可处理普通、陡峭和竖直平面。"""

from __future__ import annotations

import numpy as np


def pca_normal(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, np.argmin(values)]
    return normal / np.linalg.norm(normal)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1.0, 1.0))))


def sample_plane(normal: np.ndarray, point: np.ndarray, count: int = 1000) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(helper @ normal) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    axis_1 = np.cross(normal, helper)
    axis_1 /= np.linalg.norm(axis_1)
    axis_2 = np.cross(normal, axis_1)
    rng = np.random.default_rng(2026)
    uv = rng.uniform(-1.0, 1.0, size=(count, 2))
    return point + uv[:, :1] * axis_1 + uv[:, 1:] * axis_2


def main() -> None:
    tests = {
        "front": np.array([0.0, 0.0, -1.0]),
        "slanted": np.array([0.3, -0.2, -1.0]),
        "steep": np.array([1.0, 0.0, -0.1]),
        "vertical": np.array([1.0, 0.0, 0.0]),
    }
    for name, expected in tests.items():
        points = sample_plane(expected, np.array([0.0, 0.0, 5.0]))
        estimated = pca_normal(points)
        error = angle_deg(estimated, expected)
        print(f"{name:8s}: angular error = {error:.10f} deg")
        if error > 1e-6:
            raise SystemExit(f"PCA regression failed for {name}")
    print("All PCA plane-orientation tests passed.")


if __name__ == "__main__":
    main()
