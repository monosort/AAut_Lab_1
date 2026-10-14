import argparse
import csv
from datetime import datetime
import matplotlib.pyplot as plt  # Plot data
import pandas as pd

#TO DO:
#finish report maker
# maybe add a option to check up to a number of lag 
# think of any other ways in which we could visualise data to find lag and cycle relationships

def correlation_plot(rows,column_names,feature,Report):
	"""Plots correlation values for each lag"""

	Corr = lag_correlation_calc(rows,column_names,feature,Report)

	if Report==True:

		for column, result in zip(column_names[2:], Corr):
			fig, ax = plt.subplots(figsize=(8, 5))

			ax.plot(result, marker="o", markersize=3)
			ax.set_xlabel("Lag (days)")
			ax.set_ylabel("Correlation")
			ax.set_title(f"{column} versus today's chlorophyll")
			ax.axhline(0, color="gray", linestyle="--")
			ax.set_ylim(-1, 1)
			ax.grid(True)
			fig.tight_layout()

		plt.show()
	else:
		plt.plot(Corr, marker="o")
		plt.xlabel("Index")
		plt.ylabel("Value")
		plt.grid(True)
		plt.show()

def lag_correlation_calc(rows,column_names,feature,Report):
	if Report==True:
		Corr = []
		for column in column_names[2:]:
			feature_values=pd.Series([float(row[column])for row in rows])
			chloro_values = pd.Series([float(row["chlorophyll_a_mg_m3"])for row in rows])
			print(feature_values)
			print(chloro_values)

			Correlation=[]
			for lag in range(93):  # 0 through 14
				past_feature = feature_values.shift(lag)
				r = past_feature.corr(chloro_values)
				Correlation.append(r)
				print(f"feature {column} with lag {lag}")
				print(f"correlation = {r:.3f}")

			Corr.append(Correlation)
		return Corr
	else:
		feature_values=pd.Series([float(row[feature])for row in rows])
		chloro_values = pd.Series([float(row["chlorophyll_a_mg_m3"])for row in rows])
		print(feature_values)
		print(chloro_values)

		Correlation=[]
		for lag in range(93):  # 0 through 14
			past_feature = feature_values.shift(lag)
			r = past_feature.corr(chloro_values)
			Correlation.append(r)
			print(f"feature {feature} with lag {lag}")
			print(f"correlation = {r:.3f}")
		return Correlation


def plot_data(rows,column_names):
	"""plots all the data side by side with chlorophyl levels to get a sense of how they relate to each feature"""
	
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
	parser.add_argument("--pdfReport",action="store_true")

	#open file once and read the values then use for rest of program
	with open("data/chlorophyll_student_2015_2023.csv", newline="", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		column_names = reader.fieldnames
		rows = []
		for row in reader:
			if len(rows) == parser.parse_args().days: 
				break
			rows.append(row)
			print(row)

	if parser.parse_args().pdfReport:
		plot_data(rows,column_names)
		correlation_plot(rows,column_names,parser.parse_args().feature,parser.parse_args().pdfReport)

	if parser.parse_args().plot_data:
		plot_data(rows,column_names)

	if parser.parse_args().correlation_plot:
		if parser.parse_args().feature == None:
			parser.error("Must specify which feature to carry out correlation plot.")
		elif parser.parse_args().lag > 20:
			parser.error("Maximum lag allowed is 20.")

		correlation_plot(rows,column_names,parser.parse_args().feature,parser.parse_args().pdfReport)

	

if __name__ == "__main__":
    main()