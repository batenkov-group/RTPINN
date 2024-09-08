import os
import numpy as np
import matplotlib.pyplot as plt
from csv import reader

vza = np.load('vza.npy')
vza_tested = vza[vza > 0]
umu = np.sort(np.cos(vza_tested * np.pi/180))

path = '/data/cloudnn/libRadtran-2.0.4/'
os.chdir(path + '/auto_io_files/Data_For_Dima')
input_file = 'radiance_database_dima09182022, 14:00:03470nm.csv'
# open file in read mode
with open(input_file, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        # avg_rad = [float(x) for x in last_line.split(',')]
        run_ind = float(row[0])
        avg_rad = [float(x) for x in row[1:]]
        # print(last_line)
        plt.plot(umu, avg_rad)

f1 = open(input_file, "r")
for ii in range(1,3):
    last_line = f1.readlines()[ii]
    avg_rad = [float(x) for x in last_line.split(',')]
    run_ind = avg_rad[0]
    avg_rad = avg_rad[1:]
    print(last_line)
    plt.plot(umu, avg_rad)

f1.close()
plt.xlabel('umu [cos(theta)]')
plt.ylabel('I')
plt.show()
