import pandas as pd
import csv

def stream_rows(p):
	rt = []
	with open(p, "rU") as f:
		reader = csv.reader(f)
		header = next(reader)
		for row in reader:
			try:
				l = [float(i) for i in row[1:]]
				l.insert(0, row[0])
			except:
				print "Error in sp data, skipping this row"
			rt.insert(0, l)
		rt.insert(0, header)
	df = pd.DataFrame(rt, columns=["Date", "Open_SP", "High_SP", "Low_SP", "Close_SP", "Adj Close_SP", "Volume_SP"])
	return df

def read_vix(p):
	with open(p, "rU") as f:
		reader = csv.reader(f)
		header = next(reader)
		rt = [header]
		for row in reader:
			l = [float(i) for i in row[1:]]
			l.insert(0, row[0])
			rt.append(l)
	df = pd.DataFrame(rt, columns = ["Date", "Open_VIX", "High_VIX", "Low_VIX", "Close_VIX", "Volume_VIX", "Adj Close_VIX"])
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



if __name__ == "__main__":
	sp = stream_rows("../Data/sp_2005_2016.csv")

	vix = read_vix("../Data/vix_2005_2016.csv")

	# check_length(sp, vix)

	df = join(sp, vix)