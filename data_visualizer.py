import argparse
import csv
from datetime import datetime
import matplotlib.pyplot as plt  # Plot data

#TO DO:
#make file reader function 
#add argument for lag correlation calculator
#finish correlation_plot function

def correlation_plot(feature,lag):
	#first we open the file and read the days specified or the default
	with open("data/chlorophyll_student_2015_2023.csv", newline="", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		column_names = reader.fieldnames
		rows = []
		for row in reader:
			if len(rows) == n_days:
				break
			rows.append(row)
			print(row)

def lag_correlation_calc(feature):
	with open("data/chlorophyll_student_2015_2023.csv", newline="", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		column_names = reader.fieldnames
		rows = []
		for row in reader:
			if len(rows) == n_days: 
				break
			rows.append(row)
			print(row)

	for lag in range(15):  # 0 through 14
		past_temperature = rows["sst_c"].shift(lag)
		r = past_temperature.corr(df["chlorophyll_a_mg_m3"])
		print(f"Lag {lag}: correlation = {r:.3f}")

def plot_data(n_days):
	"""plots all the data side by side with chlorophyl levels to get a sense of how they relate to each feature"""
	
	#first we open the file and read the days specified or the default
	with open("data/chlorophyll_student_2015_2023.csv", newline="", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		column_names = reader.fieldnames
		rows = []
		for row in reader:
			if len(rows) == n_days:
				break
			rows.append(row)
			print(row)
	x_column = column_names[0]
	first_column = column_names[1]
	# Use the first column as x and the next two columns as y
	for counter in range(2, len(column_names)):
		second_column = column_names[counter]
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

		axes[1].plot(x_values, second_values, label=second_column,color="red")
		axes[1].set_xlabel(x_column)
		axes[1].set_ylabel("Value")
		axes[1].legend()
		axes[1].grid(True)
		for date in month_marks:
			axes[1].axvline(date, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

		fig.tight_layout()
	plt.show()

def main():
	parser = argparse.ArgumentParser(description="Data Visualizer")
	parser.add_argument("--plot_data", action="store_true")
	parser.add_argument("--days",type=int,default=365)
	parser.add_argument("--correlation_plot", action="store_true",)
	parser.add_argument("--feature",type=str)
	parser.add_argument("--lag", type=int, default=1)

	if parser.parse_args().plot_data:
		plot_data(parser.parse_args().days)

	if parser.parse_args().correlation_plot:
		if parser.parse_args().feature == None:
			parser.error("Must specify which feature to carry out correlation plot.")
		elif parser.parse_args().lag > 20:
			parser.error("Maximum lag allowed is 20.")

		correlation_plot(parser.parse_args().feature,parser.parse_args().lag)
	

if __name__ == "__main__":
    main()