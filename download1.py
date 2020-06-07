from re import match, search, IGNORECASE
from csv import writer
from codecs import open

from bs4 import BeautifulSoup
import requests



def get_text(node):

    soup = BeautifulSoup(requests.get("http://istmat.info" + node).text, 'lxml')
    contents = soup.find_all("div", id="content")
    res = []
    for content in contents:
        ps = soup.find_all("p")
        text, possible_authors = "", []
        for p in ps:
            if "class" in p.attrs:
                if p.attrs["class"] == ["rteright"]: possible_authors.append(p.text)
                elif p.attrs["class"] == ["rtejustify"]:
                    if not match("вопрос", p.text, flags=IGNORECASE):
                        text += p.text[6:] + "\n" if match("ответ", p.text, flags=IGNORECASE) else p.text + "\n"
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
        author = search("([А-Я]\. ){0,2}[А-Я][а-я]+([\. ])?$", possible_author)
        if author: return author.group(0)



def relevant_text_from_site_to_csv(site, dataset):
with open(str(dataset), "w", encoding="utf8") as outp:
    writer = writer(outp, delimiter="\t")
    writer.writerow(["text", "author"])
    for page in range(13):
        soup = BeautifulSoup(requests.get(site + str(page)).text, 'lxml')
        links = soup.find_all("a")
        for link in links:
	        if "допроса" in link.text:
	            writer.writerows(get_text(link.attrs["href"]))

		
		
if __name__=="main":
	relevant_text_from_site_to_csv(site="http://istmat.info/documents?tid_theme=All&tid_area=All&tid_state=All&tid_type=All&tid_tags=&period=&title=%D0%B4%D0%BE%D0%BF%D1%80%D0%BE%D1%81&page=", dataset="dataset1.csv")
