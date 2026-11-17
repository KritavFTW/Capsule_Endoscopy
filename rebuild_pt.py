import zipfile, pathlib

src = pathlib.Path("capsule_best_dir/best")
with zipfile.ZipFile("capsule_best.pt", "w", zipfile.ZIP_STORED) as z:
    for p in sorted(src.rglob("*")):
        if p.is_file():
            z.write(p, "best/" + p.relative_to(src).as_posix())
print("done")