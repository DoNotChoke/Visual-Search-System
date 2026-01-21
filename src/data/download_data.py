import os
import zipfile
import gdown

file_id = "1TclrpQOF_ullUP99wk_gjGN8pKvtErG8"
out_zip = "Stanford_Online_Products.zip"
out_dir = "sop"

# 1) Download
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, out_zip, quiet=False)

# 2) Unzip (cross-platform)
os.makedirs(out_dir, exist_ok=True)
with zipfile.ZipFile(out_zip, "r") as zf:
    zf.extractall(out_dir)

# 3) List vài dòng đầu
print("Extracted to:", os.path.abspath(out_dir))
print("Top-level files/dirs:", sorted(os.listdir(out_dir))[:20])
