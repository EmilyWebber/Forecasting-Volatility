import pandas as pd
import csv

# read in the sp data
def stream_rows(p):
	rt = []
	with open(p, "rU") as f:
		reader = csv.reader(f)
		header = next(reader)
		for row in reader:

			try:
				# grab the date and vix from the spreadsheet, and convert to a float
				l = [row[0], float(row[5])]

				# add the new row to the array
				rt.insert(0, l)

			except:
				print "Empty row here"

	for each in rt:
		if len(each) != 2:
			print each

	# transform into a data frame
	df = pd.DataFrame(rt, columns=["Date", "Adj Close_SP"])
	return df

# read the vix data into a 1D list of data points
def read_vix(p):
	with open(p, "rU") as f:
		reader = csv.reader(f)
		header = next(reader)
		rt = []
		for row in reader:
			l = [row[0], float(row[6])]
			rt.append(l)
	df = pd.DataFrame(rt, columns = ["Date", "Adj Close_VIX"])
	return df

def check_length(sp, vix):
	print "Length of sp is {}".format(len(sp))
	print "Length of vix is {}".format(len(vix))
	df.set_index(["Date"])

def join(sp, vix):
	sp = sp.set_index(["Date"])
	vix = vix.set_index(["Date"])
	# sp.join(vix, on = "Date")
	rt = pd.concat([sp, vix], axis=1, join_axes=[sp.index])
	return rt


def get_merged_data(p1, p2):
	sp = stream_rows(p1)

	vix = read_vix(p2)

	# check_length(sp, vix)

	return join(sp, vix)


def get_ordered_vix():
	sp = stream_rows("../Data/sp_2005_2016.csv")

	vix = read_vix("../Data/vix_2005_2016.csv")

	# check_length(sp, vix)

	df = join(sp, vix)

	return df["Adj Close_VIX"]


if __name__ == "__main__":
	sp = stream_rows("../Data/sp_2005_2016.csv")

	vix = read_vix("../Data/vix_2005_2016.csv")

	# check_length(sp, vix)

	df = join(sp, vix)

	print df["Adj Close_VIX"]