import numpy as np  # 匯入 numpy，用來做數值與陣列處理
import pandas as pd  # 匯入 pandas，用來建立和儲存表格資料
#from openpyxl import Workbook  # 匯入 openpyxl，可以儲存為 xlsx 格式
#from PIL import Image  # 匯入 PIL 的 Image 模組（如果要存圖片會用到）

# 設定參數
num_images = 10  # 要產生的影像數量
image_size = (100, 100)  # 每張影像的大小（寬、高）

# 建立空的統計資料表格（字典）
statistics = {
    "Image": [],  # 影像名稱
    "Max": [],    # 最大值
    "Min": [],    # 最小值
    "Mean": [],   # 平均值
    "Std": []     # 標準差
}

# 產生影像並進行統計
for i in range(num_images):  # 重複執行 10 次（從 0 到 9）
    image_rgb = np.random.randint(0, 256, (image_size[0], image_size[1], 3), dtype=np.uint8)  
    # 生成一張隨機的 RGB 彩色影像（值在 0~255，形狀為 高x寬x3通道）

    image_gray = np.mean(image_rgb, axis=2)  
    # 將 RGB 影像轉為灰階（對第3軸的 R/G/B 值取平均）

    max_val = np.max(image_gray)  # 取得灰階影像的最大值
    min_val = np.min(image_gray)  # 取得最小值
    mean_val = np.mean(image_gray)  # 取得平均亮度
    std_val = np.std(image_gray)  # 取得亮度的標準差

    # 將這張影像的統計結果加入表格中
    statistics["Image"].append(f"Image_{i+1}")  # 命名為 Image_1, Image_2,...
    statistics["Max"].append(round(max_val, 2))  # 最大值（保留2位小數）
    statistics["Min"].append(round(min_val, 2))  # 最小值
    statistics["Mean"].append(round(mean_val, 2))  # 平均值
    statistics["Std"].append(round(std_val, 2))  # 標準差

# 將統計資料轉為 pandas 的 DataFrame 表格格式
df = pd.DataFrame(statistics)

# 將表格儲存為 Excel（.xlsx）檔案
df.to_excel("images_statistics.xlsx", index=False)  # index=False 表示不輸出索引欄位

print("成功儲存為 images_statistics.xlsx ")  # 顯示成功訊息
