# Prepare input:
# ```bash
# wget https://data.hplt-project.org/one/monotext/deduplicated/be/be_1.jsonl.zst
# mv be_1.jsonl.zst be_1_deduplicated.jsonl.zst
# wget https://data.hplt-project.org/one/monotext/cleaned/be/be_1.jsonl.zst
# mv be_1.jsonl.zst be_1_cleaned.jsonl.zst
# zstdcat be_1_deduplicated.jsonl.zst | jq -r .url | sed -r 's#^https?://##' | cut -d/ -f1 | sort | uniq -c | sort -nr > hplt_be_urls_deduplicated
# zstdcat be_1_cleaned.jsonl.zst | jq -r .url | sed -r 's#^https?://##' | cut -d/ -f1 | sort | uniq -c | sort -nr > hplt_be_urls_cleaned
# ```
# Run:
# $ python3 hplt_get_valid_urls.py
# Output: hplt_be_urls_valid, hplt_be_urls_invalid

# URLs with at least 20 pages in the "deduplicated" version of HPLT Belarusian
# that don't occur in the "cleaned" version but in fact should not be filtered out
URLS_ALLOW = {
    "lyricstranslate.com", "vessoft.by", "be.vessoft.com", "bydobry.com", "post.secondbelarus.com",
    "calendar.secondbelarus.com", "moykahany.ru", "www.moykahany.ru", "pravo.kulichki.com", "mapminsk.by",
    "alivaria.by", "govorim.by", "maksimtank.ru", "kupalle.ru", "wawkalaki.ucoz.ru",
    "www.art-pol.by", "by.tribuna.com", "www.orshanka.by", "mapbelarus.by", "luch.by",
    "belarus-travel.by", "www.narasveta.by", "sosh26.by", "realschule.ru", "rasp.by",
    "dukora.by", "www.matulya.ru", "ekapraekt.by", "streetart.by", "www.natal.by",
    "www.kryga.by", "lyavon.by"
}

# URLs with at least 1000 pages in the "deduplicated" version of HPLT Belarusian
# that also occur in the "cleaned" version but in fact should be filtered out
URLS_DENY = {
    "play.google.com", "by.sgames.org", "www.machineseeker.by", "googleplaystoreapks.com", "platesmania.com",
    "onlineradiobox.com", "be.coinmill.com", "www.ofunnygames.com", "1k-tv.com", "www.goethe-verlag.com",
    "be.wheelsage.org", "by.meteocast.net", "by.meteotrend.com", "www.truckfly.com", "vigodno-advices.org",
    "be.surnameanalysis.com", "newsshopping.news"
}

def read_stat(filename):
    stat = []
    with open(filename) as f:
        for line in f:
            n, url = line.strip().split()
            stat.append((url, int(n)))
    return stat

stat_deduplicated = read_stat("hplt_be_urls_deduplicated")
stat_deduplicated_dct = {k: v for k, v in stat_deduplicated}
assert len(stat_deduplicated_dct) == len(stat_deduplicated)
stat_cleaned = read_stat("hplt_be_urls_cleaned")
stat_cleaned_dct = {k: v for k, v in stat_cleaned}
assert len(stat_cleaned_dct) == len(stat_cleaned)

print(sum(stat_deduplicated_dct.values())) # 1263852
print(sum(stat_cleaned_dct.values())) # 356534
print(sum(v for k, v in stat_deduplicated_dct.items() if (k in stat_cleaned_dct and k not in URLS_DENY) or k in URLS_ALLOW)) # 601956

with open("hplt_be_urls_valid", "w") as f:
    with open("hplt_be_urls_invalid", "w") as g:
        for url, count in stat_deduplicated:
            if (url in stat_cleaned_dct and url not in URLS_DENY) or url in URLS_ALLOW:
                print(url, file=f)
            else:
                print(url, file=g)
