from bs4 import BeautifulSoup
import requests
from re import match, search
from csv import writer, field_size_limit
from codecs import open



#field_size_limit(maxsize)

def get_text(node):

    soup = BeautifulSoup(requests.get("http://docs.historyrussia.org" + node).text, 'lxml')
    contents = soup.find_all("script")
    for content in contents:
        if "initDocview" in content.text:
            script = search("", content.text)
            print(script)
    exit(0)
    res = []
    for content in contents:
        ps = soup.find_all("p")
        text, possible_authors = "", []
        for p in ps:
            if "class" in p.attrs:
                if p.attrs["class"] == ["rteright"]: possible_authors.append(p.text)
                elif p.attrs["class"] == ["rtejustify"]: text += p.text + "\n"
                elif p.attrs["class"] == ["rtecenter"]:
                    res.append([text, get_author(possible_authors)])
                    del text, possible_authors
                    text, possible_authors = "", []
            else: possible_authors.append(p.text)
        res.append([text, get_author(possible_authors)])
    while ['', None] in res: res.remove(['', None])
    return res


def get_author(possible_authors):

    for possible_author in possible_authors:
        author = match("([А-Я]\. ){0,2}[А-Я][а-я]+([\. ])?$", possible_author)
        if author: return author.group(0)



soup = BeautifulSoup(requests.get("http://docs.historyrussia.org/ru/indexes/values/1772?per_page=563").text, 'lxml')
divs = soup.find_all("div", class_="name")
links = sum([div.find_all("a") for div in divs], [])
authors = []
texts = []
for link in links:
    authors.append(link.text[:link.text.find(".")])
    texts.append(get_text(link["href"]))
exit(0)
with open("dataset", "w", encoding="utf8") as outp:
    writer = writer(outp, delimiter="\t")
    writer.writerow(["text", "author"])
    for link in links:
        if "допроса" in link.text:
            writer.writerows(get_text(link.attrs["href"]))
