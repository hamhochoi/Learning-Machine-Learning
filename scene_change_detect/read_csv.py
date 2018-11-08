import csv


def find_ad_list(file_name="myscene.csv"):
	with open(file_name, "rb") as f:
		reader = csv.reader(f)
		
		hours_list   = []
		minutes_list = []
		seconds_list = []

		for row in reader:
			# print (row[2].split(':'))
			hours, minutes, seconds = map(float, row[2].split(':'))
			# print (hours, minutes, seconds)
			# print (row[2])	
			hours_list.append(hours)
			minutes_list.append(minutes)
			seconds_list.append(seconds)

		high_cut_rate_list = []
		max_minutes = int(max(minutes_list))
		
		for i in range(max_minutes+1):
			cut_rate = minutes_list.count(i)	
			if (cut_rate > 20):
				high_cut_rate_list.append(i)

		# print (high_cut_rate_list)

		ad_list = []
		for i in range(len(high_cut_rate_list)):
			if (i==0):
				if (high_cut_rate_list[i] == high_cut_rate_list[i+1]-1):
					ad_list.append(high_cut_rate_list[i])
				elif (minutes_list.count(high_cut_rate_list[i]) > 30):
					ad_list.append(high_cut_rate_list[i])

			if (i==len(high_cut_rate_list)-1):
				if (high_cut_rate_list[i] == high_cut_rate_list[i-1]-1):
					ad_list.append(high_cut_rate_list[i])
				elif (minutes_list.count(high_cut_rate_list[i]) > 30):
					ad_list.append(high_cut_rate_list[i])

			if (i < len(high_cut_rate_list)-1):
				if (high_cut_rate_list[i] == high_cut_rate_list[i+1]-1):
					ad_list.append(high_cut_rate_list[i])
				elif (high_cut_rate_list[i] == high_cut_rate_list[i-1]+1):
					ad_list.append(high_cut_rate_list[i])
				elif (minutes_list.count(high_cut_rate_list[i]) > 30):
					ad_list.append(high_cut_rate_list[i])

		# print (ad_list)
		return ad_list


# ad_list = find_ad_list("myscene.csv")
# print (ad_list)