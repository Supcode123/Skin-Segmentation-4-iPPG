EXP1 = {"LABEL": {
    0: [1],
    1: [2],
    2: [3],
    3: [4],
    4: [5],
    5: [6],
    6: [7],
    7: [8],
    8: [9],
    9: [10],
    10: [11],
    11: [12],
    12: [13],
    13: [14],
    14: [15],
    15: [16],
    16: [17],
    17: [18],
    255: [0,255],
},
    "CLASS": {
        # 0: 'Background',
        0: 'Skin',
        1: 'Nose',
        2: 'Right_Eye',
        3: 'Left_Eye',
        4: 'Right_Brow',
        5: 'Left_Brow',
        6: 'Right_Ear',
        7: 'Left_Ear',
        8: 'Mouth_Interior',
        9: 'Top_Lip',
        10: 'Bottom_Lip',
        11: 'Neck',
        12: 'Hair',
        13: 'Beard',
        14: 'Clothing',
        15: 'Glasses',
        16: 'Headwear',
        17: 'FACEWEAR',
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
    18: [18],
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
EXP2 = {"LABEL": {
    0: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    1: [1, 2],
    255: [0,255],
},
    "CLASS": {
        0: "Non-Skin",
        1: "Skin",
        255: "Ignore",
    }
}

EXP3 = {"LABEL": {
    0: [255],
    1: [1, 2],
    2: [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
},
    "CLASS": {
        0: "Ignore",
        1: "Skin",
        2: "Non-Skin",
    }
}

def remap_label(label: int, classes: int):  # -> (int, str):

    """ Returns remapped label id and label name given label and exp id."""
    if classes == 18 :
        _exp = EXP1
    elif classes == 2 :
        _exp = EXP2

    for k, v in _exp["LABEL"].items():

        if label in v:
            return k, _exp["CLASS"][k]

    raise ValueError("Could not remap label.")
