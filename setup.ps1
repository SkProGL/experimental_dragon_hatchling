$data = "tinystories"
$arg = if ([string]::IsNullOrWhiteSpace($args[0])) { "infer" } else { $args[0] }

if (-not (Test-Path -Path $data)) {
	uv pip install -r requirements.txt
	uv pip install --pre torch torchvision --index-url "https://download.pytorch.org/whl/nightly/cu128"
	echo "[rclone] downloading data"
	New-Item -ItemType Directory -Path $data -Force
	C:\Users\g2-leonovs\repo\remote-main\rclone\rclone.exe copy gdrive:datasets/$data factual_models/$data
}
echo "[python] initializing model"
# explorer.exe factual_models;
python start.py $arg
C:\Users\g2-leonovs\repo\remote-main\rclone\rclone.exe copy results gdrive:results
timeout /t 60; shutdown /l
