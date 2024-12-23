from copy import deepcopy
from functional import seq
from tqdm import tqdm
from record import read_record

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
        return _getOneGame(id)
    except Exception as err:
        print('getOneGame error:', id, err)
        return None

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
    players = {0: [], 1: [], 2: [], 3: []}
    players_fulu = {0: [], 1: [], 2: [], 3:[]}
    paihe = {0:[], 1:[], 2:[], 3:[]}
    is_hand_discard_dict = {0:[], 1:[], 2:[], 3:[]}
    huapai = {0:0, 1:0, 2:0, 3:0}

    for _ in range(3):
        for player in players:
            players[player].extend(walls[:4])
            walls = walls[4:]
    for player in players:
        players[player].append(walls[0])
        walls = walls[1:]
    players[0].append(walls[0])
    walls = walls[1:]

    last_discard = None
    last_draw = None
    winner_index = -1

    #print('game id:', id)
    for action in M['a']:
        player = action['p']
        action_type = action['a']
        action_detail = action['d']
        #print('action:', action)

        if action_type == 0:
            # 无动作(过)
            continue
        elif action_type == 1:
            # 摸打
            draw = get_hi(action_detail)
            discard = get_lo(action_detail)
            players[player].append(draw)
            if discard in players[player]:
                players[player].remove(discard)
            huapai[player] += 1
            last_draw = None
        elif action_type == 2:
            # 弃牌
            discard = get_lo(action_detail)
            is_hand_discard = bool((action_detail>>8)&0xFF)
            is_hand_discard_dict[player].append(is_hand_discard)

            state_dict = {
                "action_type":"discard",
                "player": player,
                "paihe": deepcopy(paihe),
                "fulu": deepcopy(players_fulu),
                "is_hand_discard": deepcopy(is_hand_discard_dict),
                "hands":deepcopy(players[player]),
                "discard":discard,
            }
            actions.append(state_dict)

            paihe[player].append(discard)
            if discard in players[player]:
                players[player].remove(discard)
            last_discard = discard
        elif action_type == 3:
            # 吃
            offer_id = get_offer(action_detail)
            lower, middle, upper = get_offer_tile(action_detail)
            if offer_id == 1:
                players[player].remove(middle)
                players[player].remove(upper)
                players_fulu[player].append((lower, middle, upper))
            elif offer_id == 2:
                players[player].remove(lower)
                players[player].remove(upper)
                players_fulu[player].append((middle, lower, upper))
            elif offer_id == 3:
                #print('before remove:', players, 'going to remove:', lower, middle, upper)
                players[player].remove(lower)
                players[player].remove(middle)
                players_fulu[player].append((upper, lower, middle))
        elif action_type == 4:
            # 碰
            offer_id = get_offer(action_detail)
            tile = get_pack_tile(action_detail)
            count = 0
            hands_pair = []
            for t in players[player][:]:
                if (t&0xFC) == (tile&0xFC):
                    players[player].remove(t)
                    hands_pair.append(t)
                    count += 1
                    if count == 2:
                        break
            if offer_id == 1:
                players_fulu[player].append([last_discard]+hands_pair)
            elif offer_id == 2:
                players_fulu[player].append([hands_pair[0], last_discard, hands_pair[1]])
            elif offer_id == 3:
                players_fulu[player].append(hands_pair+[last_discard])
        elif action_type == 5:
            # 杠
            tile = get_pack_tile(action_detail)
            if is_add_kong(action_detail):
                # 加杠
                for fulu in players_fulu[player]:
                    if isinstance(fulu, list) and len(fulu)>0 and (fulu[0]&0xFC)==(tile&0xFC):
                        if last_draw in players[player]:
                            players[player].remove(last_draw)
                            fulu.append(last_draw)
                        break
            else:
                offer_id = get_offer(action_detail)
                if offer_id:
                    # 明杠
                    if last_draw in players[player]:
                        players[player].remove(last_draw)
                    players_fulu[player].append(tuple(range(tile, tile+4)))
                else:
                    # 暗杠
                    players_fulu[player].append(list(range(tile, tile+4)))
                count = 0
                for h in players[player][:]:
                    if h&0xFC == tile&0xFC:
                        players[player].remove(h)
                        count += 1
                        if count==3:
                            break
        elif action_type == 6:
            # 和牌
            draw = get_hi(action_detail)
            winner_index = player
        elif action_type == 7:
            # 补牌(如花牌)
            draw = get_lo(action_detail)
            players[player].append(draw)
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

def getAllGames(to=114466):
    l = (seq(tqdm(range(10001, to), desc="Processing Records"))
        .map(lambda e: getOneGame(e))
        .filter(lambda e: e)
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
        .map(lambda e: (str(e[1][0]), dict(win_rate=e[1][1], rank=e[0])))
        .to_dict()
    )
    print(ret)
    return ret

if __name__ == '__main__':
    for e in getOneGame(10001):
        print(e)
        break
    games = getAllGames(to=10010)
    print(games[0].keys())
    print(games[0]['actions'][-1])
    r = rankPlayers(games)
    print(r)