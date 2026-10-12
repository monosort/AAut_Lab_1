import csv
from datetime import datetime
import matplotlib.pyplot as plt  # Plot data

with open("data/chlorophyll_student_2015_2023.csv", newline="", encoding="utf-8") as csv_file:
	reader = csv.DictReader(csv_file)
	column_names = reader.fieldnames
	rows = []
	for row in reader:
		if len(rows) == 365:
			break
		rows.append(row)

# Use the first column as x and the next two columns as y
x_column = column_names[0]
first_column = column_names[1]
second_column = column_names[11]
x_values = [datetime.strptime(row[x_column], "%Y-%m-%d") for row in rows]
month_marks = [date for date in x_values if date.day == 1]
first_values = [float(row[first_column]) for row in rows]
second_values = [float(row[second_column]) for row in rows]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x_values, first_values, label=first_column)
axes[0].set_xlabel(x_column)
axes[0].set_ylabel("Value")
axes[0].legend()
axes[0].grid(True)
for date in month_marks:
	axes[0].axvline(date, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

axes[1].plot(x_values, second_values, label=second_column)
axes[1].set_xlabel(x_column)
axes[1].set_ylabel("Value")
axes[1].legend()
axes[1].grid(True)
for date in month_marks:
	axes[1].axvline(date, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

fig.tight_layout()
plt.show()
