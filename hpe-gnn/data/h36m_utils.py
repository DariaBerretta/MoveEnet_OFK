

# Constants
SUBJECTS = {1: 'S1',
            5: 'S5',
            6: 'S6',
            7: 'S7',
            8: 'S8',
            9: 'S9',
            11: 'S11'}
SPLITS = {'train': [1, 5, 6, 7, 8],
          'val': [9, 11]}
CAMERAS = {2: 'cam2',
           4: 'cam4'}

ACTIONS = {0: 'Directions', 1: 'Directions_1', 2: 'Discussion_1', 3: 'Discussion_2', 4: 'Eating', 5: 'Eating_1',
           6: 'Greeting', 7: 'Greeting_2', 8: 'Phoning_2', 9: 'Phoning_3', 10: 'Photo', 11: 'Photo_1', 12: 'Posing',
           13: 'Posing_1', 14: 'Purchases', 15: 'Purchases_1', 16: 'SittingDown', 17: 'SittingDown_1', 18: 'Sitting_1',
           19: 'Smoking', 20: 'Smoking_2', 21: 'Waiting', 22: 'Waiting_1', 23: 'WalkDog', 24: 'WalkDog_1',
           25: 'Walking', 26: 'Walking_1', 27: 'WalkTogether', 28: 'WalkTogether_1', 29: 'Discussion', 30: 'Eating_2',
           31: 'Greeting_1', 32: 'Phoning', 33: 'Phoning_1', 34: 'SittingDown_2', 35: 'Sitting_2', 36: 'Smoking_1',
           37: 'TakingPhoto_1', 38: 'WalkingDog', 39: 'WalkingDog_1', 40: 'Directions_2', 41: 'Discussion_3',
           42: 'Sitting', 43: 'TakingPhoto', 44: 'Photo_2', 45: 'Waiting_2', 46: 'Posing_2', 47: 'Waiting_3',
           48: 'Walking_2', 49: 'WalkTogether_2'}

def name_to_embedding(name):
    components = name.split('_')
    cam = components[0]
    sub = components[1]
    action = '_'.join(components[2:])
    cam_embedding = int(cam[-1])
    sub_embedding = int(sub[1:])
    action_embedding = list(ACTIONS.values()).index(action)
    return [cam_embedding,sub_embedding, action_embedding]

def embedding_to_name(embedding):
    cam = 'cam'+str(embedding[0])
    sub = 'S'+str(embedding[1])
    action = ACTIONS[embedding[2]]
    name = '_'.join([cam, sub, action])
    return name




# import os
#
# data_path = '/home/ggoyal/data/h36m_gamer/gamer'
# filenames = os.listdir(data_path+'/raw')


# actions = {}
# count = 0
# for i, file in enumerate(filenames):
#     components = file.split('_')
#     cam = components[0]
#     sub = components[1]
#     action = '_'.join(components[2:])
#     print(file, sub, cam, action)
#     if action not in actions.values():
#         actions[count] = action
#         count += 1
# print(actions)

# for file in filenames:
#     embed = name_to_embedding(file)
#     retrieved_name = embedding_to_name(embed)
#     print(file, embed, retrieved_name)
#     assert file == retrieved_name

