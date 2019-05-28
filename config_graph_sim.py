
vSize = 5120
ccnt = 64
cdim = 32
hd = 64
ld = 128
k = 16

config_dict = {
    # Graph
    'regular_512d': {
        'useVector': False,
        'encoder': {
            'blocks' : 5,
            'particles_count' : [vSize, 1280, 512, max(256, ccnt * 2), 32],
            'conv_count' : [1, 2, 2, 0, 0],
            'res_count' : [0, 0, 0, 1, 6],
            'kernel_size' : [k, k, k, k, k],
            'bik' : [0, 32, 32, 48, 64],
            'channels' : [hd // 2, 2 * hd // 3, hd, 3 * hd // 2, max(ld, hd * 2)],
        },
        'decoder': {
            'blocks' : 1,
            'pcnt' : [vSize], # particle count
            'generator' : [6], # Generator depth
            'maxLen' : [None],
            'nConv' : [0],
            'nRes' : [0],
            'hdim' : [hd // 3],
            'fdim' : [ld], # dim of features used for folding
            'gen_hdim' : [ld],
            'knnk' : [k // 2],
            'genStruct' : 'concat',
            'genFeatures' : True,
        },
        'simulator': {
            'knnk': k
            'layers': [256]
        },
        'stages': [[0, 0]]
    },
    'regular': {
        'useVector': False,
        'encoder': {
            'blocks' : 5,
            'particles_count' : [vSize, 1280, 512, max(256, ccnt * 2), ccnt],
            'conv_count' : [1, 2, 2, 0, 0],
            'res_count' : [0, 0, 0, 1, 6],
            'kernel_size' : [k, k, k, k, k],
            'bik' : [0, 32, 32, 48, 64],
            'channels' : [hd // 2, 2 * hd // 3, hd, 3 * hd // 2, max(ld, hd * 2)],
        },
        'decoder': {
            'blocks' : 1,
            'pcnt' : [vSize], # particle count
            'generator' : [6], # Generator depth
            'maxLen' : [None],
            'nConv' : [0],
            'nRes' : [0],
            'hdim' : [hd // 3],
            'fdim' : [ld], # dim of features used for folding
            'gen_hdim' : [ld],
            'knnk' : [k // 2],
            'genStruct' : 'concat',
            'genFeatures' : True,
        },
        'simulator': {
            'knnk': k
            'layers': [256]
        },
        'stages': [[0, 0]]
    },
    'shallow': {
        'useVector': False,
        'encoder': {
            'blocks' : 3,
            'particles_count' : [vSize, 512, ccnt],
            'conv_count' : [1, 2, 0],
            'res_count' : [0, 0, 3],
            'kernel_size' : [k - 4, k, k],
            'bik' : [0, 32, 32],
            'channels' : [hd // 2, hd, max(ld, hd * 2)],
        },
        'decoder': {
            'blocks' : 1,
            'pcnt' : [vSize], # particle count
            'generator' : [6], # Generator depth
            'maxLen' : [1.5],
            'nConv' : [0],
            'nRes' : [0],
            'hdim' : [hd // 3],
            'fdim' : [ld], # dim of features used for folding
            'gen_hdim' : [ld],
            'knnk' : [k // 2],
            'genStruct' : 'concat',
            'genFeatures' : True,
        },
        'simulator': {
            'knnk': k
            'layers': [256]
        },
        'stages': [[0, 0]]
    },
    'ultraShallow': {
        'useVector': False,
        'encoder': {
            'blocks' : 2,
            'particles_count' : [vSize, ccnt],
            'conv_count' : [0, 0],
            'res_count' : [0, 0],
            'kernel_size' : [k, k],
            'bik' : [0, 128],
            'channels' : [1, max(ld, hd * 2)],
        },
        'decoder': {
            'blocks' : 1,
            'pcnt' : [vSize], # particle count
            'generator' : [4], # Generator depth
            'maxLen' : [None],
            'nConv' : [0],
            'nRes' : [0],
            'hdim' : [hd // 3],
            'fdim' : [ld], # dim of features used for folding
            'gen_hdim' : [ld],
            'knnk' : [k // 2],
            'genStruct' : 'concat',
            'genFeatures' : True,
        },
        'simulator': {
            'knnk': k
            'layers': [256]
        },
        'stages': [[0, 0]]
    },
}

config = config_dict['shallow']
