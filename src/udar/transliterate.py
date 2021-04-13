__all__ = ['transliterate']

# scholarly notation
scholar_dict = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
                'ё': 'ë', 'ж': 'ž', 'з': 'z', 'и': 'i', 'й': 'j', 'к': 'k',
                'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
                'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 'c',
                'ч': 'č', 'ш': 'š', 'щ': 'šč', 'ъ': 'ʺ', 'ы': 'y', 'ь': 'ʹ',
                'э': 'è', 'ю': 'ju', 'я': 'ja', 'і': 'i', 'ѳ': 'f', 'ѣ': 'ě',
                'ѵ': 'i', 'є': 'je', 'ѥ': 'je', 'ѕ': 'dz', 'ꙋ': 'u', 'ѡ': 'ô',
                'ѿ': 'ôt', 'ѫ': 'ǫ', 'ѧ': 'ę', 'ѭ': 'jǫ', 'ѩ': 'ję', 'ѯ': 'ks',
                'ѱ': 'ps'}
scholar_dict.update((k.title(), v.title())
                    for k, v in list(scholar_dict.items()))
scholarly_table = str.maketrans(scholar_dict)

# Library of Congress
loc_dict = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
            'ё': 'ë', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'ĭ', 'к': 'k',
            'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
            'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 't͡s',
            'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': 'ʺ', 'ы': 'y', 'ь': 'ʹ',
            'э': 'ė', 'ю': 'i͡u', 'я': 'i͡a', 'і': 'ī', 'ѳ': 'ḟ', 'ѣ': 'i͡e',
            'ѵ': 'ẏ', 'є': 'ē', 'ѥ': 'i͡e', 'ѕ': 'ż', 'ꙋ': 'ū', 'ѡ': 'ō',
            'ѿ': 'ō͡t', 'ѫ': 'ǫ', 'ѧ': 'ę', 'ѭ': 'i͡ǫ', 'ѩ': 'i͡ę', 'ѯ': 'k͡s',
            'ѱ': 'p͡s'}
loc_dict.update((k.title(), v.title()) for k, v in list(loc_dict.items()))
loc_table = str.maketrans(loc_dict)

# ISO 9:1995 or GOST 7.79-2000(A)
iso9_dict = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
             'ё': 'ë', 'ж': 'ž', 'з': 'z', 'и': 'i', 'й': 'j', 'к': 'k',
             'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
             'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c',
             'ч': 'č', 'ш': 'š', 'щ': 'ŝ', 'ъ': 'ʺ', 'ы': 'y', 'ь': 'ʹ',
             'э': 'è', 'ю': 'û', 'я': 'â', 'і': 'ì', 'ѳ': 'f̀', 'ѣ': 'ě',
             'ѵ': 'ỳ', 'ѕ': 'ẑ', 'ѫ': 'ǎ'}
iso9_dict.update((k.title(), v.title()) for k, v in list(iso9_dict.items()))
iso9_table = str.maketrans(iso9_dict)

# ICAO (passport standard as of 2013)
icao_dict = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
             'ё': 'e', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'i', 'к': 'k',
             'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
             'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts',
             'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': 'ie', 'ы': 'y', 'ь': '',
             'э': 'e', 'ю': 'iu', 'я': 'ia'}
icao_dict.update((k.title(), v.title()) for k, v in list(icao_dict.items()))
icao_table = str.maketrans(icao_dict)

system_names = {'scholar': scholarly_table,
                'scholarly': scholarly_table,
                'scientific': scholarly_table,
                'science': scholarly_table,
                'loc': loc_table,
                'lc': loc_table,
                'ala-lc': loc_table,
                'library-of-congress': loc_table,
                'iso9': iso9_table,
                'iso-9': iso9_table,
                'iso 9': iso9_table,
                'iso9:1995': iso9_table,
                'iso-9:1995': iso9_table,
                'iso 9:1995': iso9_table,
                'gost7.79-2000': iso9_table,
                'gost-7.79-2000': iso9_table,
                'gost 7.79-2000': iso9_table,
                'gost7.79': iso9_table,
                'gost-7.79': iso9_table,
                'gost 7.79': iso9_table,
                'icao': icao_table,
                'passport': icao_table,
                'ascii': icao_table,
                }


def transliterate(text, system='scholarly'):
    """Transliterate string using table from `system`.

    `system` can be one of the following:
        scholarly -- International Scholarly System (DEFAULT)
        loc -- American Library Association and Library of Congress
        iso9 -- ISO 9:1995 and GOST 7.79-2000
        icao -- ICAO passport standard (since 2013); all ASCII
    """
    return text.translate(system_names[system.lower()])
