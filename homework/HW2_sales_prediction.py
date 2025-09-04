# 匯入所需的套件
import numpy as np                      # 用於數值與陣列處理
import pandas as pd                    # 用於資料讀取與處理
import matplotlib.pyplot as plt        # 用於資料視覺化
from sklearn.preprocessing import PolynomialFeatures  # 建立多項式特徵
from sklearn.linear_model import LinearRegression     # 線性回歸模型

# 讀取銷售資料集（CSV 檔）
yearly_sales_data = pd.read_csv(r"C:\Users\danie\OneDrive\Desktop\summer lesson\homework\homework\salesdata.csv")   #要注意是csv檔
# my code begins
# 準備特徵與標籤
# 假設資料集裡有 'DayOfWeek' (1=Monday, ... ,7=Sunday) 和 'Sales'
X = yearly_sales_data[['DayOfWeek']]   # 特徵（星期幾）兩個括號是因為這樣會回傳 DataFrame（表格型態），即使只有一個欄位，也會保持二維結構
y = yearly_sales_data['Sales']         # 標籤（銷售金額）一個括號是因為這樣會回傳 Series（一維向量）

# 建立多項式特徵轉換器(1次多項式)
poly_1 = PolynomialFeatures(degree=1)    # 可以調整 degree 看擬合效果
X_poly_1 = poly_1.fit_transform(X)
model_1 = LinearRegression()
model_1.fit(X_poly_1, y)

# 建立多項式特徵轉換器 (三次多項式)
poly_3 = PolynomialFeatures(degree=3)    # 可以調整 degree 看擬合效果
X_poly_3 = poly_3.fit_transform(X)
model_3 = LinearRegression()
model_3.fit(X_poly_3, y)   # my code ends

# 預測一週（星期一到星期日）的銷售金額
# 計算每星期幾的實際平均銷售金額
X_week = np.arange(1, 8).reshape(-1, 1)        # 建立 1~7（代表星期一到日）的一維列向量
                                               # reshape 用來改變陣列的形狀，-1 的意思是「自動計算這個維度應該有多少」，1 表示「只有 1 欄」
# 1次
X_week_poly_1 = poly_1.transform(X_week)
y_week_pred_1 = model_1.predict(X_week_poly_1)

# 3次
X_week_poly_3 = poly_3.transform(X_week)          # 套用相同的多項式轉換器
y_week_pred_3 = model_3.predict(X_week_poly_3)      # 使用訓練好的模型進行預測

# 計算每星期幾的實際平均銷售金額
weekly_sales_avg = yearly_sales_data.groupby('DayOfWeek')['Sales'].mean().reset_index()

# 繪圖：實際平均 vs 預測銷售金額（按星期幾）
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales_avg['DayOfWeek'], weekly_sales_avg['Sales'], 'bo-', label='Average Actual Sales')  # 藍點線：實際平均
plt.plot(X_week, y_week_pred_1, 'g^-', label='Predicted Sales (degree=1)')   # 綠點線：1次預測結果
plt.plot(X_week, y_week_pred_3, 'ro-', label='Predicted Sales (degree=3)')   # 紅點線：1次預測結果

# 圖表標題
plt.title('Average Actual vs Predicted Sales by Day of the Week')  

plt.xlabel('Day of the Week')  # X 軸：星期幾
plt.xticks(np.arange(1, 8), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])  # 對應的星期文字
plt.ylabel('Sales')     # Y 軸：銷售金額

plt.legend()      # 顯示圖例
plt.show()        # 顯示圖形