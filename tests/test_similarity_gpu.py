import pytest
from parallel_rdkit import FingerprintParams, cross_similarity


def test_gpu_rejects_unsupported_configuration_without_cuda():
    with pytest.raises(ValueError, match="Morgan"):
        cross_similarity(["CCO"], ["CCO"], backend="gpu",
                         fp_params=FingerprintParams(fp_type="rdkit"))


def test_gpu_missing_runtime_is_explicit():
    try:
        import nvmolkit  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            cross_similarity(["CCO"], ["CCO"], backend="gpu")


def test_gpu_dispatch_rejects_truthy_overwrite_before_call(tmp_path, monkeypatch):
    path = tmp_path / "existing.npy"
    path.write_bytes(b"original")
    called = []

    def fake_gpu(*args, **kwargs):
        called.append(kwargs)
        raise AssertionError("GPU backend must not be called")

    import parallel_rdkit._similarity_gpu as gpu
    monkeypatch.setattr(gpu, "cross_similarity_gpu", fake_gpu)
    with pytest.raises(TypeError, match="overwrite"):
        cross_similarity(["CCO"], ["CCO"], backend="gpu",
                         output_path=path, overwrite="yes")
    assert path.read_bytes() == b"original"
    assert called == []


def test_gpu_dispatch_normalizes_numpy_boolean(monkeypatch):
    seen = {}

    def fake_gpu(left, right, **kwargs):
        seen.update(kwargs)
        return "mock-result"

    import parallel_rdkit._similarity_gpu as gpu
    monkeypatch.setattr(gpu, "cross_similarity_gpu", fake_gpu)
    assert cross_similarity(["CCO"], ["CCN"], backend="gpu",
                            overwrite=False) == "mock-result"
    assert seen["overwrite"] is False
