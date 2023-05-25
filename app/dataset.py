import glob
import itertools
from multiprocessing import Pool
import os
from pathlib import Path
import pickle
import re
import tarfile
from typing import List
import zipfile
import urllib.request
from lxml import etree
from nltk.tokenize import sent_tokenize, word_tokenize
import tqdm


def join(array: List[str], extra_regex_chars=r'.,?!:;()') -> List[str]:
    joined = " ".join(array)
    for s in extra_regex_chars:
        if s in "([":
            old = s + " "
        else:
            old = " " + s
        joined = joined.replace(old, s)
    return joined

def download_and_extract_NKJP(url, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = "dataset.tar.gz"
    downloaded_file = os.path.join(output_folder, filename)
    urllib.request.urlretrieve(url, downloaded_file)
    with tarfile.open(downloaded_file, "r:gz") as tar:
        tar.extractall(output_folder)
    os.remove(downloaded_file)
    
def read_and_tokenize_NKJP(corpus_folder):
    tokenized_sentences = []
    for file_path in glob.glob(f"{corpus_folder}/**/*.xml", recursive=True):
        tree = etree.parse(file_path)
        namespaces = tree.getroot().nsmap.copy()
        namespaces["default"] = namespaces.pop(None)
        abs = tree.xpath("//default:ab", namespaces=namespaces)
        for ab in abs:
            text = " ".join(ab.xpath(".//text()"))
            if text:
                for sentence in sent_tokenize(text, language="polish"):
                    sentence = re.sub(r"[^a-z ąćęłńóśżź]+", "", sentence.lower())
                    token = word_tokenize(sentence, language="polish")
                    tokenized_sentences.append(token)
    return tokenized_sentences

def download_and_extract_WCCRS(url, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = "dataset.zip"
    downloaded_file = os.path.join(output_folder, filename)
    urllib.request.urlretrieve(url, downloaded_file)
    with zipfile.ZipFile(downloaded_file, "r") as zip:
        zip.extractall(output_folder)
    os.remove(downloaded_file)

def read_and_tokenize_WCCRS(corpus_folder):
    tokenized_sentences = []
    for file_path in glob.glob(f"{corpus_folder}/dataset/*.txt"):
        ts = read_and_tokenize_txt(file_path)
        tokenized_sentences.extend(ts)
    return tokenized_sentences

def read_and_tokenize_books(corpus_folder, extra_regex_chars = r''):
    tokenized_sentences = []
    for file_path in glob.glob(f"{corpus_folder}/*.txt"):
        ts = read_and_tokenize_txt(file_path, extra_regex_chars=extra_regex_chars)
        tokenized_sentences.extend(ts)
    return tokenized_sentences

def read_and_tokenize_books_3500(corpus_folder, extra_regex_chars = r''):
    files = list(glob.glob(f"{corpus_folder}/**/*.txt", recursive=True))
    with Pool(12) as pool:
        # ts = read_and_tokenize_txt(file_path, None, 'ignore')
        args = zip(files, itertools.repeat('utf8'), itertools.repeat(None), itertools.repeat(extra_regex_chars))
        ts = pool.starmap(read_and_tokenize_txt, tqdm.tqdm(args, total=len(files)))

    tokenized_sentences = []
    for _ts in ts:
        tokenized_sentences.extend(_ts)
    return tokenized_sentences

def read_and_tokenize_texts(corpus_folder, encoding = 'utf8', errors = None, extra_regex_chars = r''):
    files = list(glob.glob(f"{corpus_folder}/*.txt", recursive=True))
    with Pool(12) as pool:
        args = zip(files, itertools.repeat(encoding), itertools.repeat(errors), itertools.repeat(extra_regex_chars))
        ts = pool.starmap(read_and_tokenize_txt, tqdm.tqdm(args, total=len(files)))

    tokenized_sentences = []
    for _ts in ts:
        tokenized_sentences.extend(_ts)
    return tokenized_sentences

def read_texts(corpus_folder, encoding = 'utf8', errors = None):
    texts: List[str] = []
    for path in glob.glob(os.path.join(corpus_folder, '*.txt')):
        with open(path, 'r', encoding=encoding) as f:
            texts.append(f.read())
    return texts

def read_and_tokenize_txt(txt_path: str, encoding = 'utf8', errors = None, extra_regex_chars = r''):
    try:
        tokenized_sentences = []
        with open(txt_path, 'r', encoding=encoding, errors=errors) as file:
            text = file.read()
            if text:
                for sentence in sent_tokenize(text, language="polish"):
                    sentence = re.sub(r"[^a-z ąćęłńóśżź"+extra_regex_chars+"]+", "", sentence.lower())
                    token = word_tokenize(sentence, language="polish")
                    tokenized_sentences.append(token)
        return tokenized_sentences
    except:
        print('error', txt_path)

def save_tokenized_sentences(tokenized_sentences, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(tokenized_sentences, file)

def load_tokenized_sentences(file_path):
    with open(file_path, "rb") as file:
        tokenized_sentences = pickle.load(file)
    return tokenized_sentences

def convert_encoding(path: str, from_format: str = 'windows-1250', to_format: str = 'utf8'):

    with open(path, 'r', encoding=from_format) as f:
        text = f.read()
    with open(path, 'w', encoding=to_format) as f:
        f.write(text)


if __name__ == '__main__':
    output_dataset_folder = "datasets/pickled"
    if False:
        url = "http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=get&target=NKJP-PodkorpusMilionowy-1.2.tar.gz"
        output_download_folder = "datasets/NKJP-PodkorpusMilionowy-raw"
        os.makedirs(output_dataset_folder)
        download_and_extract_NKJP(url, output_download_folder)
        tokenized_sentences = read_and_tokenize_NKJP(output_download_folder)
        save_tokenized_sentences(tokenized_sentences, output_dataset_folder + '/nkjp_clear.pickle')
    if False:
        url = 'https://clarin-pl.eu/dspace/bitstream/handle/11321/700/dataset_clarin.zip?sequence=1&isAllowed=y'
        output_download_folder = "datasets/WCCRS-raw"
        download_and_extract_WCCRS(url, output_download_folder)
        tokenized_sentences = read_and_tokenize_WCCRS(output_download_folder)
        save_tokenized_sentences(tokenized_sentences, output_dataset_folder + '/wccrs_clear.pickle')
    if False:
        output_download_folder = "datasets/books-raw"
        tokenized_sentences = read_and_tokenize_books(output_download_folder)
        save_tokenized_sentences(tokenized_sentences, output_dataset_folder + '/books_clear.pickle')
    if False:
        output_download_folder = "datasets/books-raw"
        tokenized_sentences = read_and_tokenize_books(output_download_folder)
        save_tokenized_sentences(tokenized_sentences, output_dataset_folder + '/books_clear.pickle')
    if False:
        output_download_folder = "datasets/books-3500/data"
        tokenized_sentences = read_and_tokenize_books_3500(output_download_folder)
        save_tokenized_sentences(tokenized_sentences, output_dataset_folder + '/books_3500_clear.pickle')
    if False:
        output_download_folder = "datasets/books-1000"
        tokenized_sentences = read_and_tokenize_books_3500(output_download_folder)
        save_tokenized_sentences(tokenized_sentences, output_dataset_folder + '/books_1000_clear.pickle')
    if False:
        books_3500_dataset = load_tokenized_sentences(
            "datasets/pickled/books_3500_clear.pickle"
        )
        words = []
        for seq in books_3500_dataset:
            words += seq
        with open('datasets/pickled/books_3500_flat.pickle', 'wb') as f:
            pickle.dump(words, f) 

    if True:
        tokenized_sentences = read_and_tokenize_texts('datasets/bajki-extend', extra_regex_chars=r'')
        save_tokenized_sentences(tokenized_sentences, output_dataset_folder + '/bajki-extebd_clear.pickle')
        words = []
        for sentence in tokenized_sentences:
            words += sentence
        save_tokenized_sentences(words, output_dataset_folder + '/bajki-extebd_flat.pickle')

    if False:
        output_download_folder = "datasets/books-3500"
        tokenized_sentences = read_and_tokenize_books_3500(output_download_folder, extra_regex_chars=r'.,?!:;')
        save_tokenized_sentences(tokenized_sentences, output_dataset_folder + '/books_3500_clear_char.pickle')
        words = []
        for sentence in tokenized_sentences:
            words += sentence
        save_tokenized_sentences(words, output_dataset_folder + '/books_3500_flat_char.pickle')
        
    if False:
        output_download_folder = "datasets/books-raw"
        tokenized_sentences = read_and_tokenize_books(output_download_folder, extra_regex_chars=r'.,?!')
        save_tokenized_sentences(tokenized_sentences, output_dataset_folder + '/books_clear_chars.pickle')
        