
def menu(options: list, title: str = "Options:", query: str = "Selection:"):
    validSelection = False
    while not validSelection:
        print(title)
        for i, label in enumerate(options):
            print(f"\t{i+1}. {label.title()}")
        sel = input(query)
        try:
            selIdx = int(sel)
            if selIdx - 1 >= 0 and selIdx - 1 < len(options):
                validSelection = True
            else:
                continue
        except:
            continue
    return selIdx - 1
