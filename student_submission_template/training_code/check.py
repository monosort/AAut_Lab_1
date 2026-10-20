import pandas as pd
raw = pd.read_csv("chlorophyll_student_2015_2023.csv", parse_dates=["date"])
tr = raw[raw["date"] < "2023-01-01"]["chlorophyll_a_mg_m3"]
print(tr.quantile([0.9, 0.99, 0.995, 0.999]))
print(tr.sort_values(ascending=False).head(20))