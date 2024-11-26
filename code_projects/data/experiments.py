EXP = {"LABEL": {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9],
    10: [10],
    11: [11],
    12: [12],
    13: [13],
    14: [14],
    15: [15],
    16: [16],
    17: [17],
    18: [18],
    255: [255],
},
    "CLASS": {
        0: 'Background',
        1: 'Skin',
        2: 'Nose',
        3: 'Right_Eye',
        4: 'Left_Eye',
        5: 'Right_Brow',
        6: 'Left_Brow',
        7: 'Right_Ear',
        8: 'Left_Ear',
        9: 'Mouth_Interior',
        10: 'Top_Lip',
        11: 'Bottom_Lip',
        12: 'Neck',
        13: 'Hair',
        14: 'Beard',
        15: 'Clothing',
        16: 'Glasses',
        17: 'Headwear',
        18: 'Facewear',
        255: "Ignore",
    }
}


def remap_label(label: int):  # -> (int, str):

    """ Returns remapped label id and label name given label and exp id."""

    for k, v in EXP["LABEL"].items():

        if label in v:
            return k, EXP["CLASS"][k]

    raise ValueError("Could not remap label.")
