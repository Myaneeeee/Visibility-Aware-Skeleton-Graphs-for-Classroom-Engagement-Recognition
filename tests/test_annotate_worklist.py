import annotate


def test_load_worklist_ids_from_file(tmp_path):
    worklist_file = tmp_path / "worklist.txt"
    worklist_file.write_text("vidA_id_1_f0_f89\n\nvidB_id_2_f0_f89\nvidC_id_3_f0_f89\n")

    ids = annotate.load_worklist_ids(str(worklist_file))

    assert ids == {"vidA_id_1_f0_f89", "vidB_id_2_f0_f89", "vidC_id_3_f0_f89"}


def test_load_worklist_ids_none_returns_none():
    assert annotate.load_worklist_ids(None) is None
