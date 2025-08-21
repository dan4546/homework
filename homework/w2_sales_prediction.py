# 匯入所需的套件
import numpy as np                      # 用於數值與陣列處理
import pandas as pd                    # 用於資料讀取與處理
import matplotlib.pyplot as plt        # 用於資料視覺化
from sklearn.preprocessing import PolynomialFeatures  # 建立多項式特徵
from sklearn.linear_model import LinearRegression     # 線性回歸模型

# 讀取銷售資料集（CSV 檔）
yearly_sales_data = pd.read_csv(r"C:\Users\User\Desktop\w2\salesdata.csv")   #要注意是csv檔
# my code begins
# 準備特徵與標籤
# 假設資料集裡有 'DayOfWeek' (1=Monday, ... ,7=Sunday) 和 'Sales'
X = yearly_sales_data[['DayOfWeek']]   # 特徵（星期幾）
y = yearly_sales_data['Sales']         # 標籤（銷售金額）

# 建立多項式特徵轉換器 (例如三次多項式)
poly = PolynomialFeatures(degree=3)    # 可以調整 degree 看擬合效果
X_poly = poly.fit_transform(X)

# 建立並訓練線性回歸模型
model = LinearRegression()
model.fit(X_poly, y)   # my code ends

# 預測一週（星期一到星期日）的銷售金額
# 計算每星期幾的實際平均銷售金額
X_week = np.arange(1, 8).reshape(-1, 1)        # 建立 1~7（代表星期一到日）的一維列向量
X_week_poly = poly.transform(X_week)          # 套用相同的多項式轉換器
y_week_pred = model.predict(X_week_poly)      # 使用訓練好的模型進行預測

# 計算每星期幾的實際平均銷售金額
weekly_sales_avg = yearly_sales_data.groupby('DayOfWeek')['Sales'].mean().reset_index()

# 繪圖：實際平均 vs 預測銷售金額（按星期幾）
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales_avg['DayOfWeek'], weekly_sales_avg['Sales'], 'bo-', label='Average Actual Sales')  # 藍點線：實際平均
plt.plot(X_week, y_week_pred, 'ro-', label='Predicted Sales')# 紅點線：預測結果

# 圖表標題
plt.title('Average Actual vs Predicted Sales by Day of the Week')  

plt.xlabel('Day of the Week')                                       # X 軸：星期幾
plt.xticks(np.arange(1, 8), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])  # 對應的星期文字
plt.ylabel('Sales')                                                 # Y 軸：銷售金額

plt.legend()                                                        # 顯示圖例
plt.show()                                                          # 顯示圖形