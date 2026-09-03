from pathlib import Path

import numpy as np

from ultralytics.data.dataset import YOLODataset


class HardCaseYOLODataset(YOLODataset):
    """
    YOLODataset with hard-case sample weights.

    hard_case_file:
        每行一个 hard case 图片路径。

    hard_case_weight:
        hard case 相对于普通样本的采样权重。
    """

    def __init__(
            self,
            *args,
            hard_case_file=None,
            hard_case_weight=1.0,
            **kwargs,
    ):
        self.hard_case_file = hard_case_file
        self.hard_case_weight = float(hard_case_weight)

        if self.hard_case_weight <= 0:
            raise ValueError(
                f"hard_case_weight must be > 0, "
                f"got {self.hard_case_weight}"
            )

        super().__init__(*args, **kwargs)

        self.sample_weights = self.build_sample_weights()

    @staticmethod
    def normalize_path(path):
        """
        Normalize image path for robust matching.
        """
        return str(Path(path).expanduser().resolve())

    def build_sample_weights(self):
        """
        Build sample weights.

        Normal sample:
            1.0

        Hard case:
            hard_case_weight
        """

        weights = np.ones(
            len(self.im_files),
            dtype=np.float64,
        )

        if not self.hard_case_file:
            return weights

        hard_case_path = Path(
            self.hard_case_file
        ).expanduser()

        if not hard_case_path.exists():
            raise FileNotFoundError(
                f"Hard case file not found: "
                f"{hard_case_path}"
            )

        # -------------------------------------------------
        # Load hard case paths
        # -------------------------------------------------

        hard_cases = set()

        with open(
                hard_case_path,
                "r",
                encoding="utf-8",
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                hard_cases.add(
                    self.normalize_path(line)
                )

        # -------------------------------------------------
        # Match dataset images
        # -------------------------------------------------

        self.hard_case_indices = set()

        for i, image_path in enumerate(
                self.im_files
        ):

            normalized_path = (
                self.normalize_path(image_path)
            )

            if normalized_path in hard_cases:
                weights[i] = (
                    self.hard_case_weight
                )

                self.hard_case_indices.add(i)

        hard_count = len(
            self.hard_case_indices
        )

        total_count = len(weights)

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        original_ratio = (
            hard_count / total_count
            if total_count > 0
            else 0
        )

        weighted_ratio = (
            hard_count * self.hard_case_weight
            /
            (
                    (total_count - hard_count)
                    +
                    hard_count * self.hard_case_weight
            )
            if total_count > 0
            else 0
        )

        print(
            f"[HardCase] "
            f"total={total_count}, "
            f"matched={hard_count}, "
            f"original_ratio="
            f"{original_ratio:.4f}, "
            f"weight={self.hard_case_weight:.2f}, "
            f"expected_ratio="
            f"{weighted_ratio:.4f}"
        )

        return weights
