from copy import deepcopy
from functional import seq
import numpy as np
from common import *
from record import read_record
TILE_TYPE_NUM = 34
TILES = (
    '1m','2m','3m','4m','5m','6m','7m','8m','9m',
    '1s','2s','3s','4s','5s','6s','7s','8s','9s',
    '1p','2p','3p','4p','5p','6p','7p','8p','9p',
    '1z','2z','3z','4z','5z','6z','7z','1f','2f'
)

def get_hi(p):
    return (p>>8)&0xFF

def get_lo(p):
    return p&0xFF

def get_tiles(tile_id: int) -> str:
    return TILES[tile_id//4]

def getTileTypeId(tile_id: int):
    return tile_id//4
def getTileTypeIdPair(tile_id: int):
    """
    将 0~135 的 TileId 转换为 (suit, rank) 的二元组。
    suit 取值范围: 1~5 (1:万, 2:条, 3:饼, 4:风, 5:箭)
    rank 取值范围:
        - 万/条/饼 为 1~9
        - 风       为 1~4 (东南西北)
        - 箭       为 1~3 (中发白)
    """
    if 0 <= tile_id < 36:        # 万
        suit = 1
        rank = tile_id // 4 + 1  # tile_id // 4 对应 0~8, +1 => 1~9
    elif 36 <= tile_id < 72:     # 条
        suit = 2
        rank = (tile_id - 36) // 4 + 1
    elif 72 <= tile_id < 108:    # 饼
        suit = 3
        rank = (tile_id - 72) // 4 + 1
    elif 108 <= tile_id < 124:   # 风
        suit = 4
        rank = (tile_id - 108) // 4 + 1  # 1~4
    else:                        # 箭
        suit = 5
        rank = (tile_id - 124) // 4 + 1  # 1~3
    return (suit, rank)

def get_offer(p):
    return (p>>6)&3

def get_pack_tile(p):
    return (p&0x3F)<<2

def get_offer_tile(p):
    packed_tile = get_pack_tile(p)
    lower_tile = packed_tile - 4 + ((p>>10)&3)
    middle_tile = packed_tile + ((p>>12)&3)
    upper_tile = packed_tile + 4 + ((p>>14)&3)
    return lower_tile, middle_tile, upper_tile

def is_add_kong(p):
    return (p&0x0300) == 0x0300

def translate_fulu(fulu_unit):
    # 将副露中的牌号翻译为可读形式
    if isinstance(fulu_unit, (list, tuple)):
        return [get_tiles(x) for x in fulu_unit]
    else:
        # 不预期出现单一整数fulu
        return [get_tiles(fulu_unit)]

def getOneGame(id)->list:
    try:
        ret = _getOneGame(id)
        #print('getOneGame succ: ', id)
        return ret
    except RemoveFromHandException as err:
        print('getOneGame error:', err)

class RemoveFromHandException(Exception):
    """Custom exception for discard errors in the game."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def removeFromHand (M, game_id, hands, player_index, discard, type):
    game_id=game_id
    player_name=M['p'][player_index]['n']
    discard_tile=get_tiles(discard)
    if discard not in hands[player_index]:
        raise RemoveFromHandException(
            f"DiscardException. extra_msg={type}, game_id={game_id}, player={player_name}, discard={discard_tile}"
        )
    hands[player_index].remove(discard)

def _getOneGame(id)->list:
    """
    根据给定id从数据中读取牌局记录，并返回最终胜利玩家的出牌序列过程。

    返回格式：
    {
      'win': 最终胜利player的id,
      'winner_process': [
         {
           'fulu': 当前时刻该玩家副露牌的列表(已翻译成字符串),
           'hands': 当前时刻该玩家手牌的列表(已翻译成字符串),
           'discard': 本次出牌的牌(字符串)
         },
         ...
      ]
    }
    """
    M = read_record(id)
    if M is None:
        return None
    M = M['script']

    actions = []

    walls = M['w']
    dice = M['d']
    start_id = (36*((4-(dice[0]+dice[1]-1)%4)%4) + 2*(dice[0]+dice[1]+dice[2]+dice[3])) % 143

    # 发牌
    walls = walls[start_id:] + walls[:start_id]
    hands = {0: [], 1: [], 2: [], 3: []}
    players_fulu = {0: [], 1: [], 2: [], 3:[]}
    paihe = {0:[], 1:[], 2:[], 3:[]}
    is_hand_discard_dict = {0:[], 1:[], 2:[], 3:[]}
    huapai = {0:0, 1:0, 2:0, 3:0}

    for _ in range(3):
        for player_index in hands:
            hands[player_index].extend(walls[:4])
            walls = walls[4:]
    for player_index in hands:
        hands[player_index].append(walls[0])
        walls = walls[1:]
    hands[0].append(walls[0])
    walls = walls[1:]

    last_discard = 'to-init-1'
    last_draw = 'to-init-2'
    winner_index = -1

    #print('game id:', id)
    for action in M['a']:
        player_index = action['p']
        action_type = action['a']
        action_detail = action['d']
        #print('action:', action)

        if action_type == 0:
            # 无动作(过)
            continue
        elif action_type == 1:
            # 展示花牌
            draw = get_hi(action_detail)
            discard = get_lo(action_detail)
            #print(action_type, discard, getTileTypeId(discard))
            hands[player_index].append(draw)
            removeFromHand(M, id, hands, player_index, discard, '花')
            huapai[player_index] += 1
            last_draw = 'to-init-flower'
        elif action_type == 2:
            # 弃牌
            discard = get_lo(action_detail)
            is_hand_discard = bool((action_detail>>8)&0xFF)
            is_hand_discard_dict[player_index].append(is_hand_discard)

            state_dict = {
                "action_type": "discard",
                "player": player_index,
                "paihe": deepcopy(paihe),
                "fulu": deepcopy(players_fulu),
                "is_hand_discard": deepcopy(is_hand_discard_dict),
                "hands":deepcopy(hands[player_index]),
                "discard": discard,
            }
            actions.append(state_dict)

            paihe[player_index].append(discard)
            removeFromHand(M, id, hands, player_index, discard, '弃')
            last_discard = discard
        elif action_type == 3:
            # 吃
            offer_id = get_offer(action_detail)
            lower, middle, upper = get_offer_tile(action_detail)
            if offer_id == 1:
                removeFromHand(M, id, hands, player_index, upper , '吃3')
                removeFromHand(M, id, hands, player_index, middle, '吃3')
                players_fulu[player_index].append((lower, middle, upper))
            elif offer_id == 2:
                removeFromHand(M, id, hands, player_index, upper , '吃2')
                removeFromHand(M, id, hands, player_index, lower , '吃2')
                players_fulu[player_index].append((middle, lower, upper))
            elif offer_id == 3:
                removeFromHand(M, id, hands, player_index, middle, '吃1')
                removeFromHand(M, id, hands, player_index, lower , '吃1')
                players_fulu[player_index].append((upper, lower, middle))
        elif action_type == 4:
            # 碰
            offer_id = get_offer(action_detail)
            tile = get_pack_tile(action_detail)
            hands_pair = []
            for t in hands[player_index][:]:
                if (t&0xFC) == (tile&0xFC):
                    removeFromHand(M, id, hands, player_index, t , '碰')
                    hands_pair.append(t)
            if offer_id == 1:
                players_fulu[player_index].append([last_discard]+hands_pair)
            elif offer_id == 2:
                players_fulu[player_index].append([hands_pair[0], last_discard, hands_pair[1]])
            elif offer_id == 3:
                players_fulu[player_index].append(hands_pair+[last_discard])
        elif action_type == 5:
            # 杠
            tile = get_pack_tile(action_detail)
            if is_add_kong(action_detail):
                # 加杠
                for fulu in players_fulu[player_index]:
                    if isinstance(fulu, list) and len(fulu)>0 and (fulu[0]&0xFC)==(tile&0xFC):
                        fulu.append(fulu[0])
            else:
                offer_id = get_offer(action_detail)
                if offer_id:
                    # 明杠
                    pass
                else:
                    # 暗杠
                    pass
                players_fulu[player_index].append(list(range(tile, tile+4)))
            for h in hands[player_index][:]:
                if h&0xFC == tile&0xFC:
                    removeFromHand(M, id, hands, player_index, h , '杠')
        elif action_type == 6:
            # 和牌
            draw = get_hi(action_detail)
            winner_index = player_index
        elif action_type == 7:
            # 补牌(如花牌)
            draw = get_lo(action_detail)
            hands[player_index].append(draw)
            last_draw = draw

    ''' one action:
    {
        'player_name': 'xxxx',
        'player_id': 111111,
        "action_type":"discard",
        "player":1,
        "paihe":{
            "0":[ 120, 124, 132, 86, 111, 110, 93, 112, 90 ],
        },
        "fulu":{
            "0":[ (83, 72, 79) ],
        },
        "is_hand_discard":{
            "0":[ true, true, true, true, true, true, true, false, false ],
        },
        "hands":[ 51, 2, 48, 27, 4, 92, 30, 21, 19, 89, 96, 56, 62, 59 ],
        "discard":59
    }
    '''

    # 输出胜利者出牌过程
    actions = (seq(actions)
        .map(lambda e: {
            'game_id': id,
            'player_name': M['p'][e['player']]['n'],
            'player_id': M['p'][e['player']]['i'],
            **e,
        })
        .to_list()
    )
    game_count = seq(M['p']).map(lambda e: (e['i'], 1)).to_dict()
    win_count = seq(M['p']).map(lambda e: (e['i'], 0)).to_dict()
    if winner_index >= 0:
        winner_id = M['p'][winner_index]['i']
        win_count[winner_id] = 1
    else:
        winner_id = -1
    return dict(
        winner_id=winner_id,
        game_count=game_count,
        win_count=win_count,
        actions=actions,
    )


def convertTileTypeId (a):
    dictConvertValue(a, 'paihe', getTileTypeId)
    dictConvertValue(a, 'fulu', getTileTypeId)
    dictConvertValue(a, 'hands', getTileTypeId)
    dictConvertValue(a, 'discard', getTileTypeId)
    return a

def convertTrainData(rank_info, games):
    """
    将原始动作数据(含paihe, fulu, hands, discard等)转换为可训练的输入、标签、权重。
    
    :param actions: list or iterable of dict，每个dict包含与训练相关的字段，例如：
        {
            'player_name': 'xxxx',
            'player_id': 111111,
            "action_type": "discard",
            "player": 1,
            "paihe": { "0": [120, 124, 132, ...], "1": [...], ... },
            "fulu": { "0": [(83, 72, 79)], ... },
            "is_hand_discard": { "0": [true, ...], ... },
            "hands": [51, 2, 48, ...],
            "discard": 59,
        }
    :return: list of tuples: [ (input_encoded, label_encoded, weight), ... ]
             或根据需求也可以拆分成 X, y, sample_weights
    """
    data_for_training = []
    for game in games:
        for action in game['actions']:
            # 1) 获取当前玩家的 rank_info
            #    注意：示例中 player_id 形如 111111，需要和 rank_info 中的 key(形如 '10014')对应。
            #    本示例假设action中包含rank_info，且player_id可以在rank_info字典里找到（真实情况需自己映射）。
            player_id_str = action['player_id']

            # 2) 计算权重：排名越靠前，权重越高(此处仅示例)
            #    下面的例子假定 rank=1 => weight=1.0, rank=2 => weight=0.75, rank=3 => weight=0.5, rank=4 => weight=0.25
            weight = len(rank_info) - rank_info[player_id_str]['win_rate']
            if rank_info[player_id_str]['win_count'] < 100:
                weight *= rank_info[player_id_str]['win_count'] // 10 * 0.1
            
            # 3) 编码输入：将paihe、fulu、hands进行整合
            paihe = action.get('paihe', {})
            fulu = action.get('fulu', {})
            hands = action.get('hands', [])
            
            # 3.1) 外部函数，将paihe, fulu, hands转成统一格式(二维/三维张量等)
            #      比如可以对每一张牌调用getTileTypeId(...)再汇总编码
            #      下面仅作示例，使用假函数 encodeTableState(...)
            # 外部函数，未来扩展
            input_encoded = encodeTableState(paihe, fulu, hands)
            
            # 4) 编码标签：discard那张牌
            discard_tile = action['discard']
            # 外部函数，未来扩展
            label_encoded = getTileTypeId(discard_tile)
            
            # 5) 将(input, label, weight)打包
            data_for_training.append((input_encoded, label_encoded, weight))
    
    return data_for_training
def encodeTableState(paihe, fulu, hands):
    """
    将当前牌局信息(paihe、fulu、hands等)编码为一个固定形状的张量(示例: (10, 10, 1)).
    
    :param paihe: dict, e.g. { "0":[tileId, ...], "1":[...], ... } ，各玩家弃牌
    :param fulu: dict, e.g. { "0":[(tileId1, tileId2, tileId3), ...], ... } ，副露(吃/碰/杠)
    :param hands: list, e.g. [tileId, tileId, ...] ，自己的手牌
    :return: np.ndarray, shape=(10, 10, 1) 的编码结果(示例)
    """
    # 这里只是一个示例，实际可根据麻将牌数量及CNN需求设计更合适的维度。
    # 例如 (5, 9, 4) 或 (34, 4) 或 (10,10) 等。
    # 这里做一个10x10的二维数组，再扩维至(10,10,1).
    
    state = np.zeros((10, 10), dtype=np.float32)
    
    # 1) 编码自己的手牌
    for tile_id in hands:
        suit, rank = getTileTypeId(tile_id)
        if suit < 10 and rank < 10:
            state[suit, rank] += 1.0  # 自己手牌可计为+1
    
    # 2) 编码所有玩家的弃牌(paihe)
    #    示例：把每张弃牌记为+2，用于和手牌做区分
    for player_idx, discard_list in paihe.items():
        for tile_id in discard_list:
            suit, rank = getTileTypeId(tile_id)
            if suit < 10 and rank < 10:
                state[suit, rank] += 2.0
    
    # 3) 编码副露(fulu)，示例记为+1.5
    #    每个元素可能是一个tuple，例如 (83, 72, 79) 代表吃/碰/杠的三张牌
    for player_idx, fulu_combos in fulu.items():
        for combo in fulu_combos:
            for tile_id in combo:
                suit, rank = getTileTypeId(tile_id)
                if suit < 10 and rank < 10:
                    state[suit, rank] += 1.5
    
    # 扩展到(10,10,1)
    return state.reshape((10, 10, 1))


def getData(begin=10001, end=114466):
    games = (seq(range(begin, end + 1))
        .map(lambda e: getOneGame(e))
        .filter(lambda e: e)
        .to_list()
    )
    rank_info = rankPlayers(games)
    actions = seq(games).flat_map(lambda e: e['actions'])
    def convertRelative (a):
        player = a['player']  # 当前玩家编号
        # 初始化目标 b
        b = {
            'paihe': {},
            'fulu': {},
            'hands': a['hands'],     # 直接复制
            'discard': a['discard'], # 直接复制
            'weight': len(games) - rank_info[a['player_id']]['rank'],
        }
        # 重排
        for i in range(4):
            # 新的 i => a['paihe'][(player + i) % 4]
            old_index = (player + i) % 4
            b['paihe'][i] = a['paihe'][old_index]
            b['fulu'][i] = a['fulu'][old_index]
        return b
    def getWeight (e):
        i = rank_info[e['player_id']]
        weight = len(rank_info) - i['win_rate']
        if i['win_count'] < 100:
            weight *= i['win_count'] // 10 * 0.1
        e['weight'] = weight
        return e
    l = (seq(actions)
        .map(convertTileTypeId)
        .map(getWeight)
        .map(convertRelative)
        .to_list()
    )
    return l

def myprint(e):
    print(e)
    return e

def rankPlayers(games):
    d = {}
    for k in 'win_count', 'game_count':
        d[k] = (seq(games)
            .flat_map(lambda e: (e[k]).items())
            .group_by(lambda e: e[0])
            .map(lambda e: (e[0], seq(e[1]).map(lambda f: f[1]).sum()))
            .to_dict()
        )
    rate = {}
    for id in d['game_count'].keys():
        rate[id] = d['win_count'][id] / d['game_count'][id]
    
    ret = (seq(rate.items())
        .sorted(lambda e: e[1], reverse=True)
        .enumerate(start=1)
        .map(lambda e: (e[1][0], dict(
            win_rate=e[1][1],
            rank=e[0],
            win_count=d['win_count'][e[1][0]],
            )))
        .to_dict()
    )
    #print(ret)
    return ret



import torch

if __name__ == '__main__':
    for e in getOneGame(10001):
        print(e)
        break
    data = getData(end=10010)
    print(data[0])
