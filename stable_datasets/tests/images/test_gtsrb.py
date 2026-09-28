"""GTSRB tests using tiny local archives and the real dataset."""

import io
import zipfile

import pytest
from PIL import Image

from stable_datasets.images import GTSRB


def _ppm(color):
    buffer = io.BytesIO()
    Image.new("RGB", (7, 5), color=color).save(buffer, format="PPM")
    return buffer.getvalue()


@pytest.fixture
def archives(tmp_path):
    train = tmp_path / "train.zip"
    test = tmp_path / "test.zip"
    labels = tmp_path / "labels.zip"

    with zipfile.ZipFile(train, "w") as archive:
        for label, color in [(0, "red"), (42, "blue")]:
            directory = f"GTSRB/Final_Training/Images/{label:05d}"
            archive.writestr(
                f"{directory}/00000_00000.ppm",
                _ppm(color),
            )
            archive.writestr(
                f"{directory}/GT-{label:05d}.csv",
                f"Filename;ClassId\n00000_00000.ppm;{label}\n",
            )

    with zipfile.ZipFile(test, "w") as archive:
        archive.writestr(
            "GTSRB/Final_Test/Images/00000.ppm",
            _ppm("green"),
        )
        archive.writestr(
            "GTSRB/Final_Test/Images/00001.ppm",
            _ppm("yellow"),
        )

    with zipfile.ZipFile(labels, "w") as archive:
        # Reversed order ensures labels are joined by filename, not ZIP position.
        archive.writestr(
            "GT-final_test.csv",
            "Filename;ClassId\n00001.ppm;42\n00000.ppm;0\n",
        )

    return train, test, labels


def _builder():
    # Bypass the automatic download/cache behavior in BaseDatasetBuilder.__new__.
    builder = object.__new__(GTSRB)
    GTSRB.__init__(builder)
    return builder


def test_train_samples(archives):
    rows = list(_builder()._generate_examples(archives[0], "train"))

    assert len(rows) == 2
    # Identical basenames in different classes must have distinct sample keys.
    assert len({key for key, _ in rows}) == 2
    assert [sample["label"] for _, sample in rows] == [0, 42]

    for (_, sample), color in zip(rows, [(255, 0, 0), (0, 0, 255)]):
        with Image.open(io.BytesIO(sample["image"])) as image:
            assert image.size == (7, 5)
            assert image.getpixel((0, 0)) == color


def test_test_labels_follow_filenames(archives):
    rows = dict(
        _builder()._generate_examples(
            archives[1],
            "test",
            archives[2],
        )
    )

    sample = rows["GTSRB/Final_Test/Images/00001.ppm"]
    assert sample["label"] == 42

    with Image.open(io.BytesIO(sample["image"])) as image:
        assert image.getpixel((0, 0)) == (255, 255, 0)

    assert rows["GTSRB/Final_Test/Images/00000.ppm"]["label"] == 0


@pytest.mark.parametrize(
    "csv_text,match",
    [
        ("Filename;Other\n00000.ppm;0\n", "Missing"),
        ("Filename;ClassId\n00000.ppm;43\n", "Invalid GTSRB"),
        ("Filename;ClassId\n00000.ppm;-1\n", "Invalid GTSRB"),
        (
            "Filename;ClassId\n00000.ppm;0\n00000.ppm;0\n",
            "Duplicate",
        ),
    ],
)
def test_bad_annotations(archives, tmp_path, csv_text, match):
    labels = tmp_path / "bad_labels.zip"

    with zipfile.ZipFile(labels, "w") as archive:
        archive.writestr("GT-final_test.csv", csv_text)

    with pytest.raises(ValueError, match=match):
        list(
            _builder()._generate_examples(
                archives[1],
                "test",
                labels,
            )
        )


def test_missing_image(archives, tmp_path):
    labels = tmp_path / "missing_image.zip"

    with zipfile.ZipFile(labels, "w") as archive:
        archive.writestr(
            "GT-final_test.csv",
            "Filename;ClassId\nmissing.ppm;0\n",
        )

    with pytest.raises(KeyError):
        list(
            _builder()._generate_examples(
                archives[1],
                "test",
                labels,
            )
        )


def test_invalid_split(archives):
    with pytest.raises(ValueError, match="Unknown GTSRB"):
        list(_builder()._generate_examples(archives[0], "validation"))


def test_test_requires_labels(archives):
    with pytest.raises(ValueError, match="ground-truth"):
        list(_builder()._generate_examples(archives[1], "test"))


def test_loading_and_cache_reuse(archives, tmp_path, monkeypatch):
    calls = []

    def fake_download(specs, dest_folder):
        calls.append(dest_folder)
        assert len(specs) == 3
        return list(archives)

    monkeypatch.setattr(
        "stable_datasets.images.gtsrb.bulk_download",
        fake_download,
    )

    kwargs = {
        "download_dir": tmp_path / "raw",
        "processed_cache_dir": tmp_path / "processed",
    }
    datasets = GTSRB(**kwargs)

    assert set(datasets.keys()) == {"train", "test"}
    assert len(datasets["train"]) == len(datasets["test"]) == 2

    sample = datasets["train"][0]
    assert set(sample) == {"image", "label"}
    assert isinstance(sample["image"], Image.Image)
    assert isinstance(sample["label"], int)
    assert calls == [kwargs["download_dir"]]

    def no_download(*args, **kwargs):
        pytest.fail("Loading a fully cached dataset must not download again")

    monkeypatch.setattr(
        "stable_datasets.images.gtsrb.bulk_download",
        no_download,
    )

    reopened = GTSRB(split="test", **kwargs)
    assert len(reopened) == 2
    assert reopened[0]["label"] == 42

    with pytest.raises(ValueError, match="not found"):
        GTSRB(split="validation", **kwargs)


def test_gtsrb_metadata():
    builder = _builder()
    info = builder.info

    assert str(GTSRB.VERSION) == "1.0.0"
    assert info.description
    assert info.homepage == GTSRB.SOURCE["homepage"]
    assert info.citation == GTSRB.SOURCE["citation"]
    assert info.supervised_keys == ("image", "label")
    assert set(info.features) == {"image", "label"}
    assert info.features["label"].names == [str(i) for i in range(43)]
    assert builder._candidate_splits() == ["train", "test"]


def test_gtsrb_real_data():
    """Load real GTSRB data, downloading it if it is not already cached."""
    datasets = GTSRB()

    for split, expected_count in [
        ("train", 39209),
        ("test", 12630),
    ]:
        dataset = datasets[split]
        assert len(dataset) == expected_count, (
            f"{split}: expected {expected_count} samples, got {len(dataset)}"
        )

        counts = [0] * 43

        for index, sample in enumerate(dataset):
            context = f"{split}[{index}]"

            assert set(sample) == {"image", "label"}, (
                f"{context}: unexpected fields {set(sample)}"
            )

            image = sample["image"]
            assert isinstance(image, Image.Image), (
                f"{context}: expected PIL image, got {type(image)}"
            )
            assert image.mode == "RGB", (
                f"{context}: expected RGB image, got {image.mode}"
            )
            image.load()  # Force decoding to detect broken images.

            label = sample["label"]
            assert isinstance(label, int), (
                f"{context}: expected integer label, got {type(label)}"
            )
            assert 0 <= label < 43, (
                f"{context}: label {label} is outside [0, 42]"
            )
            counts[label] += 1

        missing_classes = [
            label for label, count in enumerate(counts) if count == 0
        ]
        assert not missing_classes, (
            f"{split}: missing classes {missing_classes}"
        )