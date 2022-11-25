import sys

date = sys.argv[1]
test_file = "date_setting.py"
file = open(test_file, "w")
file.write("%s" % date)
file.close()
