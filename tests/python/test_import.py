import cfd
import cfd_core


def test_import_and_version() -> None:
    assert isinstance(cfd.__version__, str)
    assert cfd.__version__
    assert isinstance(cfd_core.version(), str)


def test_hello() -> None:
    assert cfd_core.hello() == "cfd_core bindings ok"