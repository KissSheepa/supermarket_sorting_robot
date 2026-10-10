#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO backend used by the participant baseline."""

import os
from pathlib import Path

import numpy as np


class YoloBackend:
    # 与 gen_dataset.py 的 CLASSES / data.yaml names 保持 class_id 对齐
    CLASS_NAMES = [
        "sanmingzhi", "heweidao", "shupian", "zhijin",
        "maidong", "kouxiangtang", "pingguo", "chengzi", "kele",
    ]

    # 已知不兼容组合: cuDNN 9.2.x + PyTorch 2.7.x(+cu128) 会在部分 conv/线性层
    # 抛出 CUDNN_STATUS_NOT_INITIALIZED。检测到该组合时自动回退到原生 CUDA 卷积,
    # 避免依赖团队手工设置环境变量。
    _BAD_CUDNN_MAJOR = 9
    _BAD_CUDNN_MINOR_MIN = 2
    _BAD_CUDNN_MINOR_MAX = 2

    def __init__(self, weights: Path, confidence: float = 0.65, device: str = "auto"):
        self.confidence = confidence
        self.model = None
        self.device = device

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")

        import torch
        from ultralytics import YOLO

        self._apply_cudnn_guard(torch)

        selected_device = self._select_device(torch, device)
        original_load = torch.load

        def compatible_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = compatible_load
        try:
            self.model = YOLO(str(weights)).to(selected_device)
            self.model.model.eval()
        finally:
            torch.load = original_load

        print(f"[YoloBackend] loaded {weights} on {selected_device}")

    @staticmethod
    def _apply_cudnn_guard(torch):
        """cuDNN 与 PyTorch 版本不兼容时自动禁用 cuDNN(回退原生 CUDA conv)。"""
        disable = False
        reason = ""

        # 1) 显式开关(外部可强制覆盖)
        env = os.environ.get("SUPERMARKET_DISABLE_CUDNN")
        if env is not None:
            if env in ("1", "true", "True", "yes", "on"):
                disable = True
                reason = "SUPERMARKET_DISABLE_CUDNN is enabled"
        else:
            # 2) 自动检测已知不兼容的 cuDNN 版本
            try:
                cudnn_ver = torch.backends.cudnn.version()
                if cudnn_ver is not None:
                    # 常规编码: major*10000 + minor*1000 + patch*10
                    # (例如 9.2.0 -> 92000); 兼容其它编码回退到字符串判断
                    major = int(cudnn_ver) // 10000
                    minor = (int(cudnn_ver) // 1000) % 10
                    bad_major = major == YoloBackend._BAD_CUDNN_MAJOR
                    bad_minor = YoloBackend._BAD_CUDNN_MINOR_MIN <= minor <= YoloBackend._BAD_CUDNN_MINOR_MAX
                    if bad_major and bad_minor:
                        disable = True
                        reason = (f"auto-detected incompatible cuDNN {cudnn_ver} "
                                  f"(torch {torch.__version__}, cuda {getattr(torch.version, 'cuda', '?')})")
            except Exception:
                pass

        if disable:
            torch.backends.cudnn.enabled = False
            os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
            print(f"[YoloBackend] cuDNN disabled ({reason}); using native CUDA conv")

    @staticmethod
    def _select_device(torch, requested: str):
        requested = requested.lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested not in {"auto", "cuda"} and not requested.startswith("cuda:"):
            raise ValueError("YOLO device must be auto, cpu, cuda, or cuda:N")
        if not torch.cuda.is_available():
            if requested != "auto":
                raise RuntimeError(
                    f"CUDA device '{requested}' was requested but CUDA is unavailable")
            print("[YoloBackend] CUDA unavailable; using CPU")
            return torch.device("cpu")

        if requested.startswith("cuda:"):
            try:
                device_index = int(requested.split(":", 1)[1])
            except ValueError:
                raise ValueError(f"invalid CUDA device index in '{requested}'")
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device '{requested}' requested but only "
                    f"{torch.cuda.device_count()} device(s) are visible")
            selected_device = torch.device(f"cuda:{device_index}")
        else:
            selected_device = torch.device("cuda:0")
            device_index = 0

        major, minor = torch.cuda.get_device_capability(device_index)
        capability = major * 10 + minor
        supported = [
            int(arch[3:])
            for arch in torch.cuda.get_arch_list()
            if arch.startswith("sm_")
        ]
        compatible = any(
            arch // 10 == major and arch % 10 <= minor for arch in supported
        )
        if compatible:
            return selected_device
        if requested == "cuda":
            raise RuntimeError(
                f"GPU sm_{capability} is unsupported by this PyTorch build: {supported}")
        print(
            f"[YoloBackend] GPU sm_{capability} is unsupported by this PyTorch "
            f"build ({supported}); using CPU")
        return torch.device("cpu")

    def detect(self, rgb: np.ndarray) -> list[dict]:
        # 每次推理前再确认一次 cuDNN 状态(防止其它模块在别处重新打开 cuDNN)
        import torch
        if os.environ.get("SUPERMARKET_DISABLE_CUDNN") in ("1", "true", "True", "yes", "on"):
            torch.backends.cudnn.enabled = False

        results = self.model(rgb, verbose=False)[0]
        detections = []
        for box in results.boxes:
            confidence = float(box.conf.item())
            if confidence < self.confidence:
                continue
            class_id = int(box.cls.item())
            if class_id >= len(self.CLASS_NAMES):
                continue
            x0, y0, x1, y1 = map(int, box.xyxy[0].cpu().numpy())
            detections.append(
                {
                    "class": self.CLASS_NAMES[class_id],
                    "x": (x0 + x1) // 2,
                    "y": (y0 + y1) // 2,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "conf": confidence,
                }
            )
        return detections
