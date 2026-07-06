# Paper Case Studies

## Scope

This is a read-only qualitative analysis of the frozen Format-SFT baseline and the current DAG-IG seed42 main checkpoint. It uses gold labels only for post-hoc categorization, not for training, scoring, or prediction changes.

## Strict Comparison Counts

| split | DAG-IG only strict | Format only strict | both strict | both fail | DAG retrieval gain | DAG retrieval loss |
|---|---:|---:|---:|---:|---:|---:|
| dev | 8 | 2 | 40 | 48 | 8 | 3 |
| test | 5 | 1 | 21 | 37 | 5 | 2 |

## Representative Wins And Losses

### dev

#### DAG-IG strict wins

- `pix2fact_11b37c2b51` gold=`6`
  - question: There is a financial institution close to the the word "Broadway" in the photo. I'm planning to buy a house in Los Angeles. Could you please tell me how many branches of this financial institution within Los Angeles o...
  - Format query=`Bank of America Los Angeles branches` answer=`3` R@5=False strict=False
  - DAG-IG query=`Bank of America branches in Los Angeles offering home loan services` answer=`6` R@5=True strict=True
  - DAG-IG top docs: r1 bankofamerica.com (non-gold); r2 online.citi.com (gold); r3 locators.bankofamerica.com (gold)
- `pix2fact_15ebe54ae6` gold=`9,605.30`
  - question: Identify a country's name in the picture, then answer what is the GDP per Capita of this country in 2020 in USD according to World Bank?
  - Format query=`GDP per Capita of Cuba in 2020` answer=`Cuba` R@5=True strict=False
  - DAG-IG query=`GDP per Capita of Cuba in 2020` answer=`Cuba, 9,605.3 USD` R@5=True strict=True
  - DAG-IG top docs: r1 worldometers.info (gold); r2 datausa.io (gold); r3 data.worldbank.org (gold)
- `pix2fact_522951e822` gold=`TUBDDEDDXXX`
  - question: I saw a photo taken in HK, and there is a bank with red logo on the top of a building. I remembered I need to confirm their BIC to complete a transaction. What is the BIC of this bank in Germany?
  - Format query=`bank with red logo in Hong Kong` answer=`HSBC` R@5=False strict=False
  - DAG-IG query=`HSBC bank in Hong Kong` answer=`TUBDDEDDXXX` R@5=True strict=True
  - DAG-IG top docs: r1 foodle.pro (non-gold); r2 facebook.com (gold); r3 hsbc.com.cn (gold)

#### Format-SFT strict wins

- `pix2fact_8c7e55f90b` gold=`130 Pitt Street, Sydney, NSW 2000`
  - question: I want to buy a blue backpack of the same brand as the man in the picture. I am in Sydney now. Could you tell me where the flagship store of this brand is located in Sydney?
  - Format query=`adidas backpacks sydney` answer=`130 Pitt Street, Sydney, NSW 2000` R@5=True strict=True
  - DAG-IG query=`Adidas backpacks Sydney` answer=`Chatswood Neighbourhood OCS – 1-541B Chatswood Westfield, Anderson St, Chatswood, NSW 2067` R@5=True strict=False
  - DAG-IG top docs: r1 adidas.com.au (gold); r2 adidas.co.uk (gold); r3 adidas.com.au (gold)
- `pix2fact_e354a6dae9` gold=`Albert Kiessling`
  - question: I saw a chocolate brand in the picture with "1906" next to its name. It looks like both a Chinese brand and a foreign brand. Do you know who founded it?
  - Format query=`chocolate brand with 1906 next to its name` answer=`Albert Kiessling` R@5=True strict=True
  - DAG-IG query=`1906 chocolate brand` answer=`Domori` R@5=False strict=False
  - DAG-IG top docs: r1 tasteatlas.com (gold); r2 brandirectory.com (gold); r3 adidas.co.uk (gold)

#### Both retrieve support but answer wrong

- `pix2fact_0e4bdf06ed` gold=`03-5320-7744`
  - question: Identify the city name that appears in the picture. I am an American who just moved to this city and am not very familiar with it yet. I heard that this city has set up a foreign residents consultation center, and I w...
  - Format query=`foreign residents consultation center tokyo` answer=`TOKYO` R@5=True strict=False
  - DAG-IG query=`foreign residents consultation center in Tokyo` answer=`Tokyo` R@5=True strict=False
  - DAG-IG top docs: r1 livejapan.com (gold); r2 livejapan.com (gold); r3 baike.baidu.com (gold)
- `pix2fact_12f5e16804` gold=`40,385.30`
  - question: I am writting a thesis about World GDP, and I am interested in the city with its name in the picture. Could you tell me what the GDP per capita (round to 0.1 USD) of the country where this city is located was in 2024 ...
  - Format query=`Rome GDP per capita 2024 World Bank Group` answer=`Italy` R@5=True strict=False
  - DAG-IG query=`Rome GDP per capita 2024 World Bank Group` answer=`Italy` R@5=True strict=False
  - DAG-IG top docs: r1 data.worldbank.org (gold); r2 worldometers.info (gold); r3 bankofamerica.com (non-gold)
- `pix2fact_472bb0ca63` gold=`46.3`
  - question: Identify the brand of the sports shoes worn by the person in a black and grey striped T-shirt. Could you tell me what the net sales revenue (in USD billion) of this brand was for 2025 to help my analysis? (Keep one de...
  - Format query=`Nike sports shoes 2025 net sales revenue` answer=`Nike` R@5=True strict=False
  - DAG-IG query=`Nike sports shoes 2025 net sales revenue` answer=`Nike` R@5=True strict=False
  - DAG-IG top docs: r1 investors.nike.com (gold); r2 wwd.com (gold); r3 news.samsung.com (gold)

#### Both miss retrieval

- `pix2fact_03f97a81f0` gold=`cjfilms2007@gmail.com`
  - question: Could you please provide the contact email address of this Shanghai advertising company shown in the image?
  - Format query=`Shanghai advertising company` answer=`无法提供图片中广告牌上的公司电子邮件地址。` R@5=False strict=False
  - DAG-IG query=`Shanghai Advertising Company contact email` answer=`comforia-reit.co.jp` R@5=False strict=False
  - DAG-IG top docs: r1 comforia-reit.co.jp (non-gold); r2 49thcoffee.com (gold); r3 p65warnings.ca.gov (gold)
- `pix2fact_06e494ae9c` gold=`War Child`
  - question: The name of the bar listed under “CLUB” is the same as a song by one of my favorite bands. This band often raises money for charities. Have they recently donated to any charities through an album? If so, please let me...
  - Format query=`charity that received money from a band that raised money for charities` answer=`New Balance` R@5=False strict=False
  - DAG-IG query=`CLUB GASPANIC band` answer=`I don't have enough information to determine if the band has recently donated to any charities through an album.` R@5=False strict=False
  - DAG-IG top docs: -
- `pix2fact_1a5cb27263` gold=`115`
  - question: I’ve been thinking about immigrating recently. I’d like to know the capital of the country represented by the flag shown in the image, and its ranking in the 2026 Safety Index by Global Residence Index.
  - Format query=`capital of country represented by flag on pole` answer=`The capital of the country represented by the flag is Ontario, and its ranking in the 2026 Safety Index by Global Res...` R@5=False strict=False
  - DAG-IG query=`capital of country with blue background and red star flag` answer=`The country represented by the flag in the image is Ontario, Canada. The capital of Ontario is Toronto.` R@5=False strict=False
  - DAG-IG top docs: r1 britannica.com (non-gold); r2 worldometers.info (gold); r3 worldometers.info (non-gold)

### test

#### DAG-IG strict wins

- `pix2fact_5a39bbb579` gold=`The Seaside Sleepover`
  - question: As we left the amusement park, I noticed the words on the first sign from the left of the little red cabin, and it reminded me of an English storybook with the exact same title that my dad got me when I was little. Do...
  - Format query=`english storybook with the exact same title as the one mentioned in the image` answer=`Once Upon a Tiger` R@5=False strict=False
  - DAG-IG query=`Candy Floss book by author` answer=`The latest book by the author of 'The Velveteen Rabbit' is 'The Seaside Sleepover' by Jacqueline Wilson.` R@5=True strict=True
  - DAG-IG top docs: r1 untitled (non-gold); r2 cosmopolitanlasvegas.mgmresorts.com (gold); r3 nosycrow.com (non-gold)
- `pix2fact_9c906e4134` gold=`May 15th - May 17th`
  - question: I am walking around this area and remembered there is a very famous festival is held here in May. Identify my loaction from the street sign in the middle, and tell me when exactly is the festival going to take place t...
  - Format query=`street sign in the middle of the image` answer=`May` R@5=False strict=False
  - DAG-IG query=`famous festival held in May in Japan` answer=`May 15th - May 17th 2026` R@5=True strict=True
  - DAG-IG top docs: r1 en.japantravel.com (gold); r2 simonandschuster.com (gold); r3 mofba.org (gold)
- `pix2fact_e10ba14542` gold=`83`
  - question: You can see the supermarket's name on the label in the bottom right of the photo.I want to buy a set of Lego Duplo as a gift for my friend's baby from this supermarket. Could you tell me how many Lego Duplo products a...
  - Format query=`Auchan欧尚` answer=`Not available.` R@5=False strict=False
  - DAG-IG query=`Auchan欧尚 Lego Duplo products in France April 2026` answer=`83` R@5=True strict=True
  - DAG-IG top docs: r1 auchan.fr (gold); r2 harpercollinschildrensbooks.co.uk (non-gold); r3 prezi.com (non-gold)

#### Format-SFT strict wins

- `pix2fact_9ac94cba26` gold=`416-325-1941`
  - question: I am currently writing a research report on regional development in Canada. There are six small flags in the center of the building, and I would like to contact the premier of the province represented by the third fla...
  - Format query=`flags on the building` answer=`416-325-1941` R@5=True strict=True
  - DAG-IG query=`flags in the center of a building in Canada` answer=`0161 234 5000` R@5=False strict=False
  - DAG-IG top docs: r1 harpersbazaar.com (gold); r2 manchester.gov.uk (gold); r3 uspashop.com (gold)

#### Both retrieve support but answer wrong

- `pix2fact_0a647f9c35` gold=`209 East 23rd Street New York`
  - question: The three letters beginning with "S" on a book cover match the abbreviation of an art school in New York City. I have a friend who studies in its main campus and I plan to visit her this weekend. Could you tell me whi...
  - Format query=`SVA art school new york city` answer=`East 23rd Street` R@5=True strict=False
  - DAG-IG query=`SVA art school new york city` answer=`East 23rd Street` R@5=True strict=False
  - DAG-IG top docs: r1 sva.edu (gold); r2 city.yokohama.lg.jp (non-gold); r3 gov.ca.gov (gold)
- `pix2fact_17ffc21ecc` gold=`1 Chome-1-6 Namba, Chuo Ward, Osaka, 542-0076`
  - question: I want to go to the shop with the phone number in the picture. I heard that this restaurant is very good, so I made plans with my friend to go eat at this restaurant this weekend. Could you provide me with the specifi...
  - Format query=`restaurant with phone number 06-6213-1888` answer=`Honke Otako Hozenji Branch` R@5=True strict=False
  - DAG-IG query=`restaurant with phone number 06-6213-1888` answer=`Honke Otako Hozenji Branch` R@5=True strict=False
  - DAG-IG top docs: r1 tabelog.com (gold); r2 tabelog.com (gold); r3 livejapan.com (gold)
- `pix2fact_1d9c01b376` gold=`44.88`
  - question: If I want to purchase the beverage "MOCHA" priced in USD in the image, how much would I need to pay in CNY (based on the exchange rate from the OANDA platform on 2026-03-27, to 2 d.p.)?
  - Format query=`exchange rate from the OANDA platform on 2026-03-27 to USD to CNY` answer=`6.50` R@5=True strict=False
  - DAG-IG query=`exchange rate from the OANDA platform on 2026-03-27 to USD to CNY` answer=`6.50` R@5=True strict=False
  - DAG-IG top docs: r1 oanda.com (gold); r2 oanda.com (gold); r3 powersports.honda.com (gold)

#### Both miss retrieval

- `pix2fact_003d2fe46c` gold=`3,765,878`
  - question: What are the population statistics of the city shown on the sign beneath the red traffic light as of March 1, 2026?
  - Format query=`population statistics of Yokohama City as of March 1, 2026` answer=`The population of Yokohama City as of March 1, 2026, is unknown.` R@5=False strict=False
  - DAG-IG query=`population statistics of Yokohama City as of March 1, 2026` answer=`The population of Yokohama City as of March 1, 2026, is unknown.` R@5=False strict=False
  - DAG-IG top docs: r1 worldometers.info (gold); r2 city.yokohama.lg.jp (non-gold); r3 worldometers.info (non-gold)
- `pix2fact_0d1a5ad6cf` gold=`(702) 862-7800`
  - question: What phone number should I call for information about the museum whose name appears above the Big Bus in the image?
  - Format query=`Madame Tussauds phone number` answer=`03-5806-1181` R@5=False strict=False
  - DAG-IG query=`Madame Tussauds phone number` answer=`03-5806-1181` R@5=False strict=False
  - DAG-IG top docs: r1 livejapan.com (gold); r2 untitled (non-gold); r3 correspondence.premier.gov.on.ca (gold)
- `pix2fact_243d684e7d` gold=`9 days`
  - question: My sister and I have both obtained visas for the country whose flag is shown in the image. We are both Australian. The difference is that I received a 30-day visa, while my sister received a 15-day visa. My sister arr...
  - Format query=`Japanese visa duration` answer=`We can spend 7 days together.` R@5=False strict=False
  - DAG-IG query=`Japanese flag in London` answer=`We can spend 7 days together.` R@5=False strict=False
  - DAG-IG top docs: r1 harpercollinschildrensbooks.co.uk (non-gold); r2 britannica.com (non-gold); r3 worldometers.info (non-gold)

## Node-Credit Link

For the main seed42 reward rollouts, reward AUC is `1.000` for retrieval hit and `0.974` for strict success. Top-ranked samples in each GRPO group have strict success `50.4%`, while bottom-ranked samples have `15.4%`. This supports the paper claim that the node-level DAG-IG reward is discriminative rather than collapsed.

## Paper Use

Use these cases to illustrate that the main gains are usually query/retrieval improvements, while the remaining failures are dominated by retrieval misses and retrieved-evidence answer errors. Do not present these cases as additional test tuning or as evidence of web-search generalization.
