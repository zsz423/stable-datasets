"""German Traffic Sign Recognition Benchmark, official final train/test splits."""

import csv
import io
import zipfile
from pathlib import PurePosixPath

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


_BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"


class GTSRB(BaseDatasetBuilder):
    """43 traffic-sign classes using full, uncropped RGB images.

    Uses GTSRB_Final_Training_Images.zip (39,209 images) and the final
    test set (12,630 images). Labels preserve the official IDs 0 through 42.
    Images retain their original sizes. Resize and normalize at training time.
    No validation split is invented. The base builder prepares both splits
    on the first cache miss, even when a single split is requested.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage=_BASE_URL + "published-archive.html",
        assets={
            "train_images": DownloadInfo(url=_BASE_URL + "GTSRB_Final_Training_Images.zip"),
            "test_images": DownloadInfo(
                url=_BASE_URL + "GTSRB_Final_Test_Images.zip",
                checksum="md5:c7e4e6327067d32654124b0fe9e82185",
            ),
            "test_labels": DownloadInfo(
                url=_BASE_URL + "GTSRB_Final_Test_GT.zip",
                checksum="md5:fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            ),
        },
        citation="""@article{stallkamp2012man,
  title={Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition},
  author={Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  journal={Neural Networks},
  volume={32},
  pages={323--332},
  year={2012}
}""",
    )

    def _info(self):
        return DatasetInfo(
            description="GTSRB: 43 German traffic-sign classes with official final train/test splits.",
            features=Features({"image": ImageFeature(), "label": ClassLabel(names=[str(i) for i in range(43)])}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self):
        # Asset names are download resources, not dataset split names.
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        assets = self._source()["assets"]
        train_images, test_images, test_labels = bulk_download(
            [assets["train_images"], assets["test_images"], assets["test_labels"]],
            dest_folder=self._raw_download_dir,
        )
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_path": train_images, "split": "train"}),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_path": test_images, "split": "test", "labels_path": test_labels},
            ),
        ]

    @staticmethod
    def _read_annotations(image_archive, annotation_archive, csv_name, image_dir):
        with annotation_archive.open(csv_name) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, delimiter=";", skipinitialspace=True)
            if not {"Filename", "ClassId"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing Filename or ClassId column in {csv_name}")
            seen = set()
            for row in reader:
                filename = row["Filename"]
                if not filename or PurePosixPath(filename).name != filename:
                    raise ValueError(f"Invalid image filename in {csv_name}: {filename!r}")
                image_name = str(image_dir / filename)
                if image_name in seen:
                    raise ValueError(f"Duplicate annotation for {image_name}")
                seen.add(image_name)
                label = int(row["ClassId"])
                if not 0 <= label < 43:
                    raise ValueError(f"Invalid GTSRB class ID: {label}")
                yield image_name, {"image": image_archive.read(image_name), "label": label}

    def _generate_examples(self, data_path, split, labels_path=None):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown GTSRB split: {split!r}")
        if split == "test" and labels_path is None:
            raise ValueError("The test split requires the ground-truth archive.")
        with zipfile.ZipFile(data_path) as images:
            if split == "train":
                csv_names = sorted(
                    name for name in images.namelist()
                    if name.startswith("GTSRB/Final_Training/Images/")
                    and PurePosixPath(name).name.startswith("GT-") and name.endswith(".csv")
                )
                if not csv_names:
                    raise ValueError("No training annotations found in GTSRB_Final_Training_Images.zip")
                for csv_name in csv_names:
                    yield from self._read_annotations(images, images, csv_name, PurePosixPath(csv_name).parent)
            else:
                with zipfile.ZipFile(labels_path) as labels:
                    yield from self._read_annotations(
                        images, labels, "GT-final_test.csv", PurePosixPath("GTSRB/Final_Test/Images")
                    )
