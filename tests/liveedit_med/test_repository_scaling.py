from methods.liveedit_med.routing_attribution import stable_repository


def test_repository_is_nested_and_target_first():
    rows = [{"record_id": str(i), "selection_hash": str(10 - i)} for i in range(8)]
    small = stable_repository(rows, "3", 4)
    large = stable_repository(rows, "3", 8)
    assert small[0]["record_id"] == "3"
    assert [x["record_id"] for x in small] == [x["record_id"] for x in large[:4]]
