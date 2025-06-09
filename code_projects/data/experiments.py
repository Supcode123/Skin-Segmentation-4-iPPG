EXP1 = {"LABEL": {
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

EXP2 = {"LABEL": {
    0: [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    1: [1, 2],
    255: [255],
},
    "CLASS": {
        0: "Non-Skin",
        1: "Skin",
        255: "Ignore",
    }
}

EXP_ = {"LABEL": {
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
   # 18: [18],
    255: [255],
},
    "CLASS": {
        0: 'background',
        1: 'skin',
        2: 'nose',
        3: 'eye_g',
        4: 'l_eye',
        5: 'r_eye',
        6: 'l_brow',
        7: 'r_brow',
        8: 'l_ear',
        9: 'r_ear',
        10: 'mouth',
        11: 'u_lip',
        12: 'l_lip',
        13: 'hair',
        14: 'hat',
        15: 'ear_r',
        16: 'neck_l',
        17: 'neck',
        18: 'cloth',
        255: "Ignore",
    }
}


def remap_label(label: int, classes: int):  # -> (int, str):

    """ Returns remapped label id and label name given label and exp id."""
    if classes == 19:
        _exp = EXP1
    elif classes == 2:
        _exp = EXP2

    for k, v in _exp["LABEL"].items():

        if label in v:
            return k, _exp["CLASS"][k]

    raise ValueError("Could not remap label.")
