
vSize = 5120
ccnt = 64
cdim = 64
hd = 64
ld = 128
k = 16

# Dec
# [fullFC_regular, fullGen_regular] Setup for full generator - fully-connected
# coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
# blocks = 3
# pcnt = [256, 1280, self.gridMaxSize] # particle count
# generator = [4, 4, 4] # Generator depth
# hdim = [hd * 2, hd, hd // 3]
# fdim = [ld, ld, ld // 2] # dim of features used for folding
# gen_hdim = [ld, ld, ld]
# knnk = [_k, _k, _k // 2]

# [fullGen_shallow]
# coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
# blocks = 2
# pcnt = [1280, self.gridMaxSize] # particle count
# generator = [6, 3] # Generator depth
# maxLen = [1.0, 0.2]
# nConv = [2, 0]
# nRes = [2, 0]
# hdim = [self.particle_hidden_dim, self.particle_hidden_dim // 3]
# fdim = [self.particle_latent_dim, self.particle_latent_dim] # dim of features used for folding
# gen_hdim = [self.particle_latent_dim, self.particle_latent_dim]
# knnk = [self.knn_k, self.knn_k // 2]

# Dec-Vector
# 3 stages
# coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, 1
# blocks = 3
# pcnt = [self.cluster_count, 1280, self.gridMaxSize] # particle count
# generator = [6, 6]
# generator = [4, 4, 4] # Generator depth
# maxLen = [None, 1.5, 0.3]
# nConv = [2, 2, 0]
# nRes = [4, 1, 0]
# hdim = [max(self.particle_latent_dim, hd * 2), 2 * self.particle_hidden_dim // 3, self.particle_hidden_dim // 3]
# fdim = [512, self.particle_latent_dim, self.particle_latent_dim // 2] # dim of features used for folding
# gen_hdim = [512, self.particle_latent_dim, self.particle_latent_dim // 2]
# knnk = [self.knn_k, self.knn_k, self.knn_k // 2] 

# coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, 1
# blocks = 1
# pcnt = [self.gridMaxSize] # particle count
# generator = [6] # Generator depth
# maxLen = [None]
# nConv = [0]
# nRes = [0]
# hdim = [self.particle_hidden_dim // 3]
# fdim = [self.particle_latent_dim] # dim of features used for folding
# gen_hdim = [self.particle_latent_dim]
# knnk = [self.knn_k // 2]

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
        },
        'stages': [[0, 0]]
    },
    'regular_512d_mono': {
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
            'mono' : True,
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
        },
        'stages': [[0, 0]]
    },
    'regular_fS': {
        'useVector': False,
        'encoder': {
            'blocks' : 5,
            'particles_count' : [vSize, 1280, 512, max(256, ccnt * 2), ccnt],
            'conv_count' : [1, 2, 2, 0, 0],
            'res_count' : [0, 0, 0, 1, 6],
            'kernel_size' : [k - 8, k - 4, k, k, k],
            'bik' : [0, 32, 32, 48, 64],
            'channels' : [hd // 2, 2 * hd // 3, hd, 3 * hd // 2, max(ld, hd * 2)],
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
            'genStruct' : 'final_selection',
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
            'kernel_size' : [k - 8, k, k],
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
        },
        'stages': [[0, 0]]
    },
    'shallow_fS': {
        'useVector': False,
        'encoder': {
            'blocks' : 3,
            'particles_count' : [vSize, 512, ccnt],
            'conv_count' : [1, 2, 0],
            'res_count' : [0, 0, 3],
            'kernel_size' : [k - 8, k, k],
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
            'genStruct' : 'final_selection',
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
        },
        'stages': [[0, 0]]
    },
    'ultraShallow_fS': {
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
            'genStruct' : 'final_selection',
        },
        'stages': [[0, 0]]
    },

    # Vector
    'vecRegular_noSep': {
        'useVector': True,
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
            'blocks' : 2,
            'pcnt' : [128, vSize], # particle count
            'generator' : [5, 6], # Generator depth
            'maxLen' : [None, None],
            'nConv' : [2, 0],
            'nRes' : [3, 0],
            'hdim' : [max(ld, hd * 2), hd // 3],
            'fdim' : [512, ld], # dim of features used for folding
            'gen_hdim' : [512, ld],
            'knnk' : [k, k],
            'genStruct' : 'concat',
            'genFeatures' : True
        },
        'stages': [[0, 0]]
    },
    'vecRegular_ngF_noSep': {
        'useVector': True,
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
            'blocks' : 2,
            'pcnt' : [128, vSize], # particle count
            'generator' : [5, 6], # Generator depth
            'maxLen' : [None, None],
            'nConv' : [2, 0],
            'nRes' : [3, 0],
            'hdim' : [max(ld, hd * 2), hd // 3],
            'fdim' : [512, ld], # dim of features used for folding
            'gen_hdim' : [512, ld],
            'knnk' : [k, k],
            'genStruct' : 'concat',
            'genFeatures' : False
        },
        'stages': [[0, 0]]
    },
    'vecRegular': {
        'useVector': True,
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
            'blocks' : 2,
            'pcnt' : [ccnt, vSize], # particle count
            'generator' : [5, 6], # Generator depth
            'maxLen' : [None, None],
            'nConv' : [2, 0],
            'nRes' : [3, 0],
            'hdim' : [max(ld, hd * 2), hd // 3],
            'fdim' : [512, ld], # dim of features used for folding
            'gen_hdim' : [512, ld],
            'knnk' : [k, k],
            'genStruct' : 'concat',
            'genFeatures' : True
        },
        'stages': [[5, 1], [0, 0]]
    },
    'vecRegular_fS': {
        'useVector': True,
        'encoder': {
            'blocks' : 5,
            'particles_count' : [vSize, 1280, 512, max(256, ccnt * 2), ccnt],
            'conv_count' : [1, 2, 2, 0, 0],
            'res_count' : [0, 0, 0, 1, 6],
            'kernel_size' : [k - 8, k - 4, k, k, k],
            'bik' : [0, 32, 32, 48, 64],
            'channels' : [hd // 2, 2 * hd // 3, hd, 3 * hd // 2, max(ld, hd * 2)],
        },
        'decoder': {
            'blocks' : 2,
            'pcnt' : [ccnt, vSize], # particle count
            'generator' : [3, 6], # Generator depth
            'maxLen' : [None, None],
            'nConv' : [2, 0],
            'nRes' : [3, 0],
            'hdim' : [max(ld, hd * 2), hd // 3],
            'fdim' : [512, ld], # dim of features used for folding
            'gen_hdim' : [512, ld],
            'knnk' : [k, k // 2],
            'genStruct' : 'final_selection',
        },
        'stages': [[5, 1], [0, 0]]
    },
    'vecSmall': {
        'useVector': True,
        'encoder': {
            'blocks' : 3,
            'particles_count' : [vSize, 512, ccnt],
            'conv_count' : [1, 2, 0],
            'res_count' : [0, 0, 2],
            'kernel_size' : [k, k, k],
            'bik' : [0, 64, 64],
            'channels' : [hd // 2, hd, max(ld, hd * 2)],
        },
        'decoder': {
            'blocks' : 2,
            'pcnt' : [ccnt, vSize], # particle count
            'generator' : [5, 5], # Generator depth
            'maxLen' : [None, None],
            'nConv' : [2, 0],
            'nRes' : [2, 0],
            'hdim' : [max(ld, hd * 2), hd // 3],
            'fdim' : [512, ld], # dim of features used for folding
            'gen_hdim' : [512, ld],
            'knnk' : [k, k],
            'genStruct' : 'concat',
            'genFeatures' : True
        },
        'stages': [[3, 1], [0, 0]]
    },
    'vecSmall_ngF_noSep': {
        'useVector': True,
        'encoder': {
            'blocks' : 3,
            'particles_count' : [vSize, 512, ccnt],
            'conv_count' : [1, 2, 0],
            'res_count' : [0, 0, 4],
            'kernel_size' : [k, k, k],
            'bik' : [0, 64, 64],
            'channels' : [hd // 2, hd, max(ld, hd * 2)],
        },
        'decoder': {
            'blocks' : 2,
            'pcnt' : [ccnt, vSize], # particle count
            'generator' : [5, 5], # Generator depth
            'maxLen' : [None, None],
            'nConv' : [1, 0],
            'nRes' : [1, 0],
            'hdim' : [max(ld, hd * 2), hd // 3],
            'fdim' : [512, ld], # dim of features used for folding
            'gen_hdim' : [512, ld],
            'knnk' : [k, k],
            'genStruct' : 'concat',
            'genFeatures' : False
        },
        'stages': [[0, 0]]
    },
    'vecSmall_noSep': {
        'useVector': True,
        'encoder': {
            'blocks' : 3,
            'particles_count' : [vSize, 512, ccnt],
            'conv_count' : [1, 2, 0],
            'res_count' : [0, 0, 4],
            'kernel_size' : [k, k, k],
            'bik' : [0, 64, 64],
            'channels' : [hd // 2, hd, max(ld, hd * 2)],
        },
        'decoder': {
            'blocks' : 2,
            'pcnt' : [ccnt, vSize], # particle count
            'generator' : [5, 5], # Generator depth
            'maxLen' : [None, None],
            'nConv' : [1, 0],
            'nRes' : [1, 0],
            'hdim' : [max(ld, hd * 2), hd // 3],
            'fdim' : [512, ld], # dim of features used for folding
            'gen_hdim' : [512, ld],
            'knnk' : [k, k],
            'genStruct' : 'concat',
            'genFeatures' : True
        },
        'stages': [[0, 0]]
    },
    'vecSingle': {
        'useVector': True,
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
        },
        'stages': [[0, 0]]
    },
    'vecSingleSmall': {
        'useVector': True,
        'encoder': {
            'blocks' : 3,
            'particles_count' : [vSize, 512, ccnt],
            'conv_count' : [1, 2, 0],
            'res_count' : [0, 0, 2],
            'kernel_size' : [k, k, k],
            'bik' : [0, 64, 64],
            'channels' : [hd // 2, hd, max(ld, hd * 2)],
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
        },
        'stages': [[0, 0]]
    },
    
    'vecUltraShallow': {
        'useVector': True,
        'encoder': {
            'blocks' : 1,
            'particles_count' : [vSize],
            'conv_count' : [2],
            'res_count' : [0],
            'kernel_size' : [k],
            'bik' : [0],
            'channels' : [hd],
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
        },
        'stages': [[0, 0]]
    },
}

config = config_dict['regular_512d_mono']
