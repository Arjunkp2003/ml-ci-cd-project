import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# -------------------------------
# 1️⃣ Target distribution (Histogram)
# -------------------------------
plt.figure()
df["MedHouseVal"].hist(bins=30)
plt.xlabel("Median House Value")
plt.ylabel("Count")
plt.title("Target Distribution")
plt.savefig("eda/target_distribution.png")
plt.close()

# -------------------------------
# 2️⃣ Feature distributions
# -------------------------------
features = ["MedInc", "HouseAge", "AveRooms", "Population"]

for col in features:
    plt.figure()
    df[col].hist(bins=30)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.savefig(f"eda/{col}_distribution.png")
    plt.close()

# -------------------------------
# 3️⃣ Boxplot (Outlier detection)
# -------------------------------
plt.figure()
df[features].boxplot()
plt.title("Boxplot of Selected Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda/boxplot_features.png")
plt.close()

# -------------------------------
# 4️⃣ Scatter plot (Feature vs Target)
# -------------------------------
plt.figure()
plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.3)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Income vs House Value")
plt.savefig("eda/scatter_income_target.png")
plt.close()

# -------------------------------
# 5️⃣ Correlation Heatmap
# -------------------------------
plt.figure()
corr = df.corr()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.savefig("eda/correlation_heatmap.png")
plt.close()

# -------------------------------
# 6️⃣ Basic statistics (logged)
# -------------------------------
stats = df.describe()
stats.to_csv("eda/statistics_summary.csv")

print("✅ EDA completed: plots and statistics saved")
