linkages = {

###### Classes present in my database
## 1 - Boronate Ester
'Boronate':                     'Boronate Ester', # Boronate
## 2 - Boroxine
'Boroxine':                     'Boroxine', # Boroxine
## 3 - Borosilicate like
'Borosilicate-tBu':             'Borosilicate', # Borosilicate - C - 3 (CH3)
'Borosilicate-PropanoicAcid':   'Borosilicate', # Borosilicate - CH2 - CH2 - COOH
'Borosilicate-Me':              'Borosilicate', # Borosilicate - CH3
'Borosilicate':                 'Borosilicate', # Borosilicate - H
## 4 - Imine
'Imine':                        'Imine', # C(3,4)-CH=N-C(3,4) + notconnected
'ImineCH2':                     'Imine', # 2C-C-CH=N-C-2H, 1 CURATED + 69 Martin COFs
'ImineNC':                      'Imine', # 2C-C(3)-CH=N-N(2)-C + notconnected, 3 CURATED COFs
## 5 - Hydrazone
'Hydrazone':                    'Hydrazone', # Hydrazone
'HydrazoneH':                   'Hydrazone', # =CH-NH-NH-C(=O)-, 7 CoRE COFs + 2 CURATED COFs
## 6 - Azine
'Azine':                        'Azine', # Azine
'AzineH':                       'Azine', # =CH-NH-NH-CH=, 3 CoRE COFs + 1 CURATED COF
## 8 - Benzobisoxazole
'Benzobisoxazole':              'Benzobisoxazole', # Benzobisoxazole
## 9 - Ketoenamine
'Enamine':                      'Enamine', # C(3,4)=CH-NH-C(3,4) + notconnected
'Ketoenamine':                  'Enamine', # -C(3)(=O(1))-C(3)H(1)=C(3)H(1)-N(3)H(1)-
## 10 - Triazine
'Triazine':                     'Triazine', # Triazine
## 11 - Borazine
'Borazine':                     'Borazine', # Borazine
## 12 - Imide
'Imide':                        'Imide', # Imide
'Imide6':                       'Imide', # Imide

###### Classes frequently present in other (hypothetical) databases
## Amide: -C(=O)-NH- (3 CoRE COFs, 5 CURATED COFs and 6479 Mercado COFs)
'Amide':                        'Amide',
## Olefin: -CH=CH-,
'Olefin':                       'Olefin', # 2 CURATED + 667 Mercado
'Olefin(CN)':                   'Olefin', # 1 CoRE + 8 CURATED
'Olefin-CNterm':                'Olefin', # 3 CURATED
## C-C
'C-C':                          'Carbon-Carbon', # 5 CoRE + 5 CURATED + 18061 Mercado
## Amine: -CH2-NH-
'Amine':                        'Amine', # 5418 Mercado

###### Other
## Benzimidazole
'Benzimidazole':                'Other', # 2 CoRE + 14 CURATED + 16446 own (T-brick)
## Azodioxy
'Azodioxy':                     'Other', # 3x2 identical occurences (NPN-[1-3] in CoRE, 1318[0-2]N3 in CURATED) + 1 CURATED (16340N2)
## Silicate: Si-6O
'Silicate-Li':                  'Other', # 1 occurence (SiCOF-Li in CoRE)
'Silicate-Na':                  'Other', # 3 occurences (SiCOF-5_as in CoRE and 1800[0-1] in CURATED)
## Capped imine
'ImineCo':                      'Other', # Exotic imine capped with Co to form 6-connected cluster, 2 identical occurences (17200C3 in CURATED and COF-112 in CoRE)
'PhenylQuinoline':              'Other', # Something with phenyl rings and a quinoline, 2 CoRE + 5 CURATED
'TetraHydroPyranQuinoline':     'Other', # 1 CURATED
'2106':                         'Other', # 1 CURATED
'2015':                         'Other', # 1 CURATED
'Alpha-AminoNitrile':           'Other', # 5 CURATED
## Pyrimidazole: resembles a bit a capped imine, but the cap is already in the synthesis process, 4 occurences in CURATED
'Pyrimidazole1_LZU-561':        'Other',
'Pyrimidazole2_LZU-562':        'Other',
'Pyrimidazole3_LZU-563':        'Other',
'Pyrimidazole4_LZU-564':        'Other',
## Salen: extended complex, 1 occurence in CoRE (Salen-COF) + 5 occurences in CURATED (1925[0-3]N3 and 17110N2)
'Salen':                        'Other',
'Salen-Zn':                     'Other',
## BP
'BP-cube':                      'Other', # 5 CURATED COFs: 2056[0-4]N3
'BP-rod':                       'Other', # 1 CURATED COF: 20565N3
## Other
'Aminal':                       'Other', # (2C-C)-(H)-C-2(N-2(CH2-)), 1 occurence (19370N2 in CURATED)
'Squaraine':                    'Other', # square of Cs with =O functionalization and NH-link, 2 identical occurences (13110N2 in CURATED, CuP-SQ_COF in CoRE)
'Phosphazene':                  'Other', # P-N-P-N-P-N ring, 2 identical occurences (16490N2 in CURATED, MPCOF in CoRE)
'Spiroborate':                  'Other', # 2O-B-2O, 2 identical occurences (16180C2 in CURATED and ICOF-1 in CoRE)
'Spiroborate-Li':               'Other', # 2O-B-2O (2 Os are linked with Li), 2 identical occurences (16181C2 in CURATED and ICOF-2 in CoRE)
'EnamineN':                     'Other', # exotic 3c-linker, 3x2 identical occurences (1624[0-2]C2 in CURATED and TpTG-[I/Br/Cl] in CoRE)
'ImineTG+':                     'Other', # Exotic 3c-linker, 2 occurences in CURATED (20270C2 and 20271C2)
'C-C(CN)':                      'Other', # -CH2-CH(-C=-N)-, 2 CURATED COFs
'C-N':                          'Other', # 2 occurences (19550N2 in CURATED and POR-COF in CoRE)
'Propargylamine':               'Other', # -NH-CH(-C-=C-Ph)- (substituted amine), 2 CURATED (2049[0-1]N2)
'Furonitrile':                  'Other', # 5-ring with 1 O, and C-=N substituted at other C, 2 CURATED 
'Thiazole':                     'Other', # 2 identical occurences (18111N2 in CURATED and 18111N2 in CoRE)
'Ester':                        'Other', # -C(=O)-O-, 4 CURATED COFs (2047[0-3]N2) 
'Dioxin':                       'Other', # 2 CoRE + 8 CURATED
'Phenazine':                    'Other', # 2 CoRe + 7 CURATED
## Unphysical
'ImineUnphysical':              'Other', # 2C-C(3)-C=N-C(3)-2C + notconnected, 5 CoRE + 2 CURATED COFs
'ImineUnphysical2':             'Other', # 2C-C(3)-C=NH-C(3)-2C + notconnected, 1 CURATED COF
'BOImine':                      'Other' # 2C-C(3)-OH=B-C(3)-2C, 1 CURATED COF
}

