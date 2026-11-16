import numpy as np
from ballq.colours import classify_kits, sample_patch_lab, shirt_point


def test_shirt_point_is_above_the_feet():
    # identity-like homography: pitch (0..1) -> pixels (x*1000+100, y*600+100); a point 1 m up the image moves up (smaller y)
    H = np.array([[1000, 0, 100], [0, 600, 100], [0, 0, 1]], float)
    fx, fy = shirt_point(H, 0.5, 0.5, shirt_height_m=1.0, pitch=(105.3, 68.0), frame_h=1080)
    assert abs(fx - 600) < 1 and fy < 400


def test_sample_patch_lab_reads_colour():
    img = np.zeros((100, 100, 3), np.uint8); img[:] = (255, 255, 255)        # white in BGR
    L, a, b = sample_patch_lab(img, 50, 50, 10)
    assert L > 240 and abs(a - 128) < 3 and abs(b - 128) < 3


def test_classify_kits_finds_the_white_team_and_a_mislabel():
    # per slot medians (L, a, b): home dark red-ish, away white; slot 5 is labelled A but looks like home
    slots = {"1": (60, 170, 150), "2": (62, 168, 152), "3": (58, 172, 149), "4": (235, 128, 128), "5": (61, 169, 151), "6": (238, 129, 127)}
    team = {"1": "H", "2": "H", "3": "H", "4": "A", "5": "A", "6": "A"}
    out = classify_kits(slots, team, white_lightness_min=170, white_chroma_max=25)
    assert out["white_team"] == "A"
    assert out["slots"]["5"]["mislabel"] is True
    assert out["slots"]["4"]["mislabel"] is False


def test_kitdb_file_lookup_and_resolution(tmp_path):
    from ballq.kitdb import FileKitDB, lookup_kits, resolve_kits, hex_to_lab
    p = tmp_path / "kits.yaml"
    p.write_text("Alpha FC: {home: '#ffffff', away: '#000080'}\nBeta: {home: '#c00000', away: '#f0f0f0'}\n")
    db = FileKitDB(str(p))
    assert db.lookup("alpha fc")["home"] == "#ffffff" and db.lookup("Alpha")["away"] == "#000080"
    assert db.lookup("Nobody") == {}
    kits = lookup_kits({"H": "Alpha FC", "A": "Beta"}, [db])
    # video measured: H bright grey, A red -> video decides, and each team's kit is named
    res = resolve_kits(kits, {"H": hex_to_lab("#f8f8f8"), "A": hex_to_lab("#b00000")}, 170, 25)
    assert res["white_team"] == "H" and res["source"] == "video" and res["kit_used"] == {"H": "home", "A": "home"}
    # no video: database alone, home kit for H, away kit for A -> both light, so undecided
    res2 = resolve_kits(kits, {"H": None, "A": None}, 170, 25)
    assert res2["white_team"] is None and res2["source"] == "none"
    # no video, only H listed -> database picks H
    res3 = resolve_kits({"H": kits["H"], "A": {}}, {"H": None, "A": None}, 170, 25)
    assert res3["white_team"] == "H" and res3["source"] == "database"


def test_third_kit_flag():
    from ballq.colours import classify_kits
    lab = {"1": (200, 128, 128), "2": (205, 128, 130), "3": (60, 150, 150), "4": (65, 148, 152), "ref": (120, 90, 200)}
    team = {"1": "H", "2": "H", "3": "A", "4": "A", "ref": "A"}
    res = classify_kits(lab, team, 170, 25)
    assert res["white_team"] == "H" and res["slots"]["ref"]["third"] and not res["slots"]["3"]["third"]
