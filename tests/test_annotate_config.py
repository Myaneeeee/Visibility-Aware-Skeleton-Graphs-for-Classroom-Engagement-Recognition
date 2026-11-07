import annotate

def test_resolve_paths_overrides():
    u, v, a = annotate.resolve_paths("/tmp/windows", "/tmp/videos", "/tmp/annotations.csv")
    assert u == "/tmp/windows"
    assert v == "/tmp/videos"
    assert a == "/tmp/annotations.csv"

def test_resolve_paths_defaults_nonempty():
    u, v, a = annotate.resolve_paths(None, None, None)
    assert u and v and a
