UD_conflicts = {
                # UD_tag: set of conflicting udar tags
                'ADJ': {'N'},
                'ADV': {'A', 'CS', 'Pcle'},
                'CCONJ': {'Pron', 'N', 'Interj', 'Adv', 'CS'},
                'SCONJ': {'Pron', 'N', 'Interj', 'Adv', 'CC'},
                'NOUN': {'A', 'Pr'},
                'PRON': {'N', 'Pred', 'Pcle', 'Det'},
                'ADP': {'Interj', 'N', 'V'},
                'VERB': {'A', 'Pred'},
                'Masc': {'Neu', 'Fem'},
                'Fem': {'Msc'},
                'Neut': {'Msc'},
                'Sing': {'Pl'},
                'Plur': {'Sg'},
                'Nom': {'Acc', 'Gen', 'Loc', 'Dat', 'Ins', 'Voc'},
                'Gen': {'Nom', 'Acc', 'Loc', 'Dat', 'Ins', 'Voc'},
                'Dat': {'Nom', 'Acc', 'Gen', 'Loc', 'Ins', 'Voc'},
                'Acc': {'Nom', 'Gen', 'Loc', 'Dat', 'Ins', 'Voc'},
                'Ins': {'Nom', 'Acc', 'Gen', 'Loc', 'Dat', 'Voc'},
                'Loc': {'Nom', 'Acc', 'Gen', 'Dat', 'Ins', 'Voc'},
                'Voc': {'Nom', 'Acc', 'Gen', 'Loc', 'Dat', 'Ins'},
                # 'gen1': {'Nom', 'Acc', 'Loc', 'Dat', 'Ins', 'Voc'}, ??
                # 'gen2': {'Nom', 'Acc', 'Loc', 'Dat', 'Ins', 'Voc'}, ??
                # 'acc2': {'Nom', 'Gen', 'Loc', 'Dat', 'Ins', 'Voc'}, ??
                # 'loc1': {'Nom', 'Acc', 'Gen', 'Dat', 'Ins', 'Voc'}, ??
                # 'loc2': {'Nom', 'Acc', 'Gen', 'Dat', 'Ins', 'Voc'}, ??
                'Tran': {'IV'},
                'Intr': {'TV'},
                }
