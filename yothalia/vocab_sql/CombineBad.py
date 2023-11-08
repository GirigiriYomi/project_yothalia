"""
This is file offer a function that combine multi txt files in word directory into one txt file
"""
import os

def update_badwords(word_dir='./badword_files',encode='utf-8'):
    words = []

    # read all words from all txt files in word directory
    for i in os.scandir(word_dir):
        if not i.is_file() or not i.name.endswith("txt"):
            continue
        
        print(i.name)
        with open(os.path.join(word_dir,i.name),'r',encoding=encode,errors='ignore') as file:
            for word_line in file:
                if word_line == '' or word_line == '\n':
                    continue
                words.append(word_line.strip())
    
    # reduce repeated words
    words = list(set(words))

    # write all words to a combined file
    with open('./badwords_all.txt','w+',encoding=encode) as file:
        for i in words:
            file.write(i+'\n')

# nor working
def ANSI_to_UTF8(in_file,out_file):
    source_encoding = 'cp1252'

    # Open the source file with the source encoding and read the content
    with open(in_file, 'r', encoding=source_encoding) as source_file:
        content = source_file.read()

    # Open the target file in write mode with UTF-8 encoding and write the content
    with open(out_file, 'w+', encoding='utf-8') as target_file:
        target_file.write(content)

# TODO didn't check for Chinese word
if __name__ == '__main__':
    update_badwords()
    # 16768 lines when finish
    # 10609 when reduce duplicate words
        