# coding: utf-8
import csv

'''
Write read append status
'r': read
'w': write
'a': append
'r+' == r+w (read-write, IOError if file does not exist)
'w+' == w+r (read-write, file created if it doesn't exist)
'a+' == a+r (appendable and writable, file is created if it doesn't exist)
Correspondingly, if it's a binary file, just add a b to both and it's OK:
'rb' 'wb' 'ab' 'rb+' 'wb+' 'ab+'
'''


# read csv file, return list
def read_csv(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


# write csv file, return nothing
def write_csv(file_name, data, mode='w+'):
    with open(file_name, mode, newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)


# dict writ into csv file, return nothing
def write_csv_dict(file_name, data, mode='w+'):
    with open(file_name, mode, newline="\n", encoding='utf-8') as f:
        writer = csv.DictWriter(f, data[0].keys())
        # writer.writeheader()
        writer.writerows(data)


# dict read from csv file, return list
def read_csv_dict(file_name, mode='r'):
    with open(file_name, mode, newline="\n", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data