from conftest import load_script
from epi_imgproc import presentation, techniques


def test_select_evenly():
    assert presentation.select_evenly(list(range(11)), 3) == [0, 5, 10]
    assert presentation.select_evenly([1, 2], 3) == [1, 2]
    assert presentation.select_evenly([], 3) == []
    assert presentation.select_evenly([7, 8, 9], 1) == [7]


def test_make_figures(tmp_path):
    written = presentation.make_figures([("cena", presentation.synthetic_scene())], tmp_path)
    techniques_shown = len(techniques.TECHNIQUES) - 1
    assert len(written) == 2 * techniques_shown + 2
    assert all(path.exists() and path.read_bytes().startswith(b"\x89PNG") for path in written)
    assert (tmp_path / "espectros" / "fft_lowpass" / "cena.png").exists()
    assert (tmp_path / "espectros" / "filtros" / "fft_high_emphasis_H.png").exists()


def test_copy_results_table(tmp_path):
    missing = presentation.copy_results_table(tmp_path / "nope.md", tmp_path / "out")
    assert "não rodou" in missing.read_text()
    source = tmp_path / "results.md"
    source.write_text("| técnica |\n")
    assert presentation.copy_results_table(source, tmp_path / "out2").read_text() == "| técnica |\n"


def test_presentation_script_synthetic(tmp_path):
    script = load_script("make_presentation")
    assert script.main(["--synthetic", "--count", "1", "--out-dir", str(tmp_path), "--results", str(tmp_path / "x.md")]) == 0
    assert (tmp_path / "sintetico" / "LEIA-ME.md").exists()
    assert (tmp_path / "sintetico" / "antes-depois" / "clahe" / "cena-sintetica-0.png").exists()


def test_presentation_script_without_data(tmp_path):
    script = load_script("make_presentation")
    assert script.main(["--data-root", str(tmp_path), "--out-dir", str(tmp_path / "out")]) == 2
