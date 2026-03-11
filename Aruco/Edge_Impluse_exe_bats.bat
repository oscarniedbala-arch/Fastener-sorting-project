static const char* WIFI_SSID = "VM4914218";
static const char* WIFI_PASS = "Hk9ygmqs6dvd";

--api-key= "ei_8543ef8d7a54f00f86a9cec5cff5fb6cca0dcd4a163c51490e249dcacb36b3af"


set EI_API_KEY=ei_c78bd9c75b27cc05903bdca79353f11a46903a0e6fe6202cac87c85f09429fe6
py collect_and_upload_ei_cv.py --esp32 http://192.168.0.202 --label M6 --count 10 --interval 1 --category testing --debug-every 1


cd C:\Users\ON
.\ei-venv\Scripts\activate

set EI_API_KEY=ei_c78bd9c75b27cc05903bdca79353f11a46903a0e6fe6202cac87c85f09429fe6
python .\collect_and_upload_ei_cv.py --esp32 http://192.168.0.202 --label M6 --count 10 --interval 1 --category testing --debug-every 1


C:\Users\ON\ei-venv\Scripts\python.exe .\collect_and_upload_ei_cv.py --esp32 http://192.168.0.202 --label M6 --count 10 --interval 1 --category testing --debug-every 1 --pad-frac 0.7


camera is 7.5/8cm off ground

"To run the training data collection, use the following command:"

py -3.12 -m venv ei-venv
.\ei-venv\Scripts\activate
python -m pip install -U pip setuptools wheel


"Newest command for training data collection"
python .\collect_and_upload_ei_cv.py --esp32 http://192.168.0.202 --label M6 --count 100 --interval 2 --category testing --pad-frac 0.8 --target-px-per-mm 7.7 --reject-skin --reject-edge



python .\measure_screw_grid_v2.py --url http://192.168.0.202/capture.jpg --show --debug --s-max 60 --v-min 30 --v-max 230
