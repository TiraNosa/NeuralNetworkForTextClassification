import csv

# This will create and write into a new CSV
def write_csv(sent_list, label_list, out_path):
    filewriter = csv.writer(open(out_path, "w+", newline='',encoding="utf-8"))
    count = 0
    header = ['id', 'sentence', 'label']
    filewriter.writerow(i for i in header)
    for ((id, sent), label) in zip(sent_list, label_list):
        filewriter.writerow([id, sent, label])

# This reads CSV a given CSV and stores the data in a list
def read_csv(data_path):
    file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
    sent_list = []
    next(file_reader)
    for row in file_reader:
        id = row[0]
        sent = row[1]

        sent_list.append((id, sent))
    return sent_list

