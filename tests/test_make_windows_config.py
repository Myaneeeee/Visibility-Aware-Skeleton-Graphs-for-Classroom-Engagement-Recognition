import inspect
import make_windows

def test_process_accepts_dir_overrides():
    sig = inspect.signature(make_windows.process_jsonl_to_tensors)
    assert "input_tracks_dir" in sig.parameters
    assert "output_tensor_dir" in sig.parameters
    assert sig.parameters["input_tracks_dir"].default is None
