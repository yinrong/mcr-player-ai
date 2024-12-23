from record import read_record
from copy import deepcopy

TILES = ('1m','2m','3m','4m','5m','6m','7m','8m','9m','1s','2s',
         '3s','4s','5s','6s','7s','8s','9s','1p','2p','3p','4p',
         '5p','6p','7p','8p','9p','1z','2z','3z','4z','5z','6z',
         '7z','1f','2f')

def get_hi(p):
    return (p>>8)&0xFF

def get_lo(p):
    return p&0xFF

def get_tiles(tiles_id: int) -> str:
    return TILES[tiles_id//4]

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

def get_action(id)->list:
    r"""
    read and transfer the data from database into a long list, containg all the nessary information
    of the game.
    :params:id : the index of the database.
    :return: long list of every discard of the game.
    """
    ret = dict()
    action_list = []
    ret['action_list'] = action_list
    mahjong_record = read_record(id)
    if mahjong_record is None:
        return None
    walls = mahjong_record['script']['w']
    dice = mahjong_record['script']['d']
    actions = mahjong_record['script']['a']
    start_id = (36*((4-(dice[0]+dice[1]-1)%4)%4) + 2*(dice[0]+dice[1]+dice[2]+dice[3])) % 143
    # initial tiles 
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
    for action in actions:
        print('action:', action)
        player = action['p']
        action_type = action['a']
        action_detail = action['d']
        if action_type==0:
            continue
        elif action_type==1:
            draw = get_hi(action_detail)
            discard = get_lo(action_detail)
            players[player].append(draw)
            players[player].remove(discard)
            huapai[player]+=1
        elif action_type==2:
            discard = get_lo(action_detail)
            is_hand_discard = bool(get_hi(action_detail))
            is_hand_discard_dict[player].append(is_hand_discard)
            state_dict = {
                "action_type":"discard",
                "player": player,
                "paihe": deepcopy(paihe),
                "fulu": deepcopy(players_fulu),
                "is_hand_discard":is_hand_discard_dict,
                "hands":deepcopy(players[player]),
                "discard":discard,
            }
            action_list.append(state_dict)
            paihe[player].append(discard)
            players[player].remove(discard)
            last_discard = discard
        elif action_type==3:
            offer_id = get_offer(action_detail)
            lower, middle, upper = get_offer_tile(action_detail)
            if offer_id==1:
                players[player].remove(middle)
                players[player].remove(upper)
                players_fulu[player].append((lower, middle, upper))
            elif offer_id==2:
                players[player].remove(lower)
                players[player].remove(upper)
                players_fulu[player].append((middle, lower, upper))
            elif offer_id==3:
                players[player].remove(lower)
                players[player].remove(middle)
                players_fulu[player].append((upper, lower, middle))
        elif action_type==4:
            offer_id = get_offer(action_detail)
            lower = get_pack_tile(action_detail)
            count = 0
            hands_pair = []
            for tile in players[player]:
                if lower&0xFC==tile&0xFC:
                    hands_pair.append(tile)
                    players[player].remove(tile)
                    count += 1
                if count == 2:
                    break
            if offer_id == 1:
                players_fulu[player].append([last_discard]+hands_pair)
            elif offer_id == 2:
                players_fulu[player].append([hands_pair[0], last_discard, hands_pair[1]])
            elif offer_id == 3:
                players_fulu[player].append(hands_pair+[last_discard])
        elif action_type==5:
            tile = get_pack_tile(action_detail)
            if is_add_kong(action_detail):
                for fulu in players_fulu[player]:
                    if isinstance(fulu, list) and fulu[0]&0xFC==tile&0xFC:
                        fulu.append(last_draw)
                        players[player].remove(last_draw)
                        break
            else:
                offer_id = get_offer(action_detail)
                if offer_id:
                    players_fulu[player].append(tuple(range(tile, tile+4)))
                    players[player].remove(last_draw)
                else:
                    players_fulu[player].append(list(range(tile, tile+4)))
                count = 0
                for hand in players[player]:
                    if hand&0xFC == tile&0xFC:
                        players[player].remove(hand)
                        count += 1
                    if count==3:
                        break
        elif action_type==6:
            ret['win'] = player
            draw = get_hi(action_detail)
        elif action_type==7:
            draw = get_lo(action_detail)
            players[player].append(draw)
            last_draw = draw
    return ret

def formatted_print(action_list, hands_id=None):
    r"""
    Formatted print the discard of the hands
    """
    for action in action_list:
        player = action['player']
        fulu = action['fulu'][player]
        hands = action['hands']
        discard = action['discard']
        for fulu_i in fulu:
            print('[', end='')
            for i in fulu_i:
                print(get_tiles(i), end='')
            print(']', end=' ')
        hands.sort()
        for hand in hands:
            print(get_tiles(hand), end='')
        print('  ', f"discard: {get_tiles(discard)}")
from copy import deepcopy

def get_all_actions (game_id):
    """
    获取最终胜利的玩家，从游戏开始到胜利的完整步骤。
    每一步包括：当前副露、当前手牌、选择的弃牌。
    
    :param game_id: 数据库中的游戏ID
    :return: 胜利玩家的弃牌序列，包含每一步的状态（副露、手牌、弃牌）
    """
    steps = []
    mahjong_record = read_record(game_id)  # 读取游戏记录
    
    if mahjong_record is None:
        return None

    walls = mahjong_record['script']['w']
    dice = mahjong_record['script']['d']
    actions = mahjong_record['script']['a']

    # 初始化开始的摸牌索引
    start_id = (36 * ((4 - (dice[0] + dice[1] - 1) % 4) % 4) + 2 * (dice[0] + dice[1] + dice[2] + dice[3])) % 143
    walls = walls[start_id:] + walls[:start_id]

    # 玩家初始信息
    players = {i: [] for i in range(4)}  # 每个玩家的手牌
    players_fulu = {i: [] for i in range(4)}  # 每个玩家的副露（吃、碰、杠）
    paihe = {i: [] for i in range(4)}  # 玩家弃牌堆
    winner_id = None

    # 初始发牌
    for _ in range(3):
        for player in players:
            players[player].extend(walls[:4])
            walls = walls[4:]
    for player in players:
        players[player].append(walls.pop(0))
    players[0].append(walls.pop(0))  # 庄家多摸一张

    # 遍历所有动作
    for action in actions:
        player = action['p']  # 当前玩家
        action_type = action['a']  # 动作类型
        action_detail = action['d']  # 动作细节

        if action_type == 0:  # 无动作，跳过
            continue

        elif action_type == 1:  # 摸牌并打出
            draw = get_hi(action_detail)  # 摸到的牌
            discard = get_lo(action_detail)  # 打出的牌
            players[player].append(draw)  # 添加摸到的牌
            players[player].remove(discard)  # 从手牌中移除打出的牌

            # 记录当前状态
            step = {
                "player": player,
                "fulu": deepcopy(players_fulu[player]),
                "hands": deepcopy(players[player]),
                "discard": discard
            }
            if winner_id is None:  # 只记录胜利玩家的过程
                steps.append(step)
            paihe[player].append(discard)  # 弃牌加入牌河

        elif action_type == 2:  # 弃牌
            discard = get_lo(action_detail)
            players[player].remove(discard)

            # 记录当前状态
            step = {
                "player": player,
                "fulu": deepcopy(players_fulu[player]),
                "hands": deepcopy(players[player]),
                "discard": discard
            }
            if winner_id is None:  # 只记录胜利玩家的过程
                steps.append(step)
            paihe[player].append(discard)

        elif action_type == 3:  # 吃牌
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
                players[player].remove(lower)
                players[player].remove(middle)
                players_fulu[player].append((upper, lower, middle))

        elif action_type == 4:  # 碰牌
            tile = get_pack_tile(action_detail)
            count = 0
            hands_pair = []
            for t in players[player]:
                if tile & 0xFC == t & 0xFC:
                    hands_pair.append(t)
                    players[player].remove(t)
                    count += 1
                if count == 2:
                    break
            players_fulu[player].append(hands_pair + [tile])

        elif action_type == 5:  # 杠牌
            tile = get_pack_tile(action_detail)
            if is_add_kong(action_detail):  # 加杠
                for fulu in players_fulu[player]:
                    if isinstance(fulu, list) and fulu[0] & 0xFC == tile & 0xFC:
                        fulu.append(tile)
                        players[player].remove(tile)
                        break
            else:  # 明杠或暗杠
                players_fulu[player].append([tile] * 4)
                players[player] = [t for t in players[player] if t & 0xFC != tile & 0xFC]

        elif action_type == 6:  # 和牌 (胜利)
            winner_id = player
            break  # 记录胜利玩家，退出循环

    # 提取胜利玩家的序列
    return dict(
        winner_id=winner_id,
        steps=steps,
    )


def get_player_discard_sequence(game_id: int) -> dict:
    """
    获取胜利玩家的弃牌过程，包括当前副露、手牌和选择的弃牌。
    
    :param game_id: 游戏记录的唯一 ID
    :return: 一个字典，包含胜利玩家的动作序列。如果记录无效，返回 None。
    """
    # 初始化返回值
    ret = {'winner': None, 'sequence': []}

    # 读取游戏记录
    mahjong_record = read_record(game_id)  # 假定已实现的函数
    print(mahjong_record)
    if mahjong_record is None:
        return None

    # 提取脚本数据
    walls = mahjong_record['script']['w']
    dice = mahjong_record['script']['d']
    actions = mahjong_record['script']['a']

    # 根据骰子确定起始发牌位置
    start_id = (36 * ((4 - (dice[0] + dice[1] - 1) % 4) % 4) + 
                2 * (dice[0] + dice[1] + dice[2] + dice[3])) % 143
    walls = walls[start_id:] + walls[:start_id]

    # 初始化玩家数据
    players = {0: [], 1: [], 2: [], 3: []}  # 每个玩家的手牌
    players_fulu = {0: [], 1: [], 2: [], 3: []}  # 副露牌组
    paihe = {0: [], 1: [], 2: [], 3: []}  # 牌河（弃牌区）

    # 发牌
    for _ in range(3):  # 每人三轮抓4张
        for player in players:
            players[player].extend(walls[:4])
            walls = walls[4:]
    for player in players:  # 每人抓1张
        players[player].append(walls[0])
        walls = walls[1:]
    players[0].append(walls[0])  # 庄家多抓一张
    walls = walls[1:]

    # 游戏逻辑变量
    winner = None
    last_discard = None
    last_draw = None

    # 遍历动作序列
    for action in actions:
        player = action['p']
        action_type = action['a']
        action_detail = action['d']

        if action_type == 0:  # 无效动作
            continue

        elif action_type == 1:  # 花牌补牌（draw & discard）
            draw = get_hi(action_detail)
            discard = get_lo(action_detail)
            players[player].append(draw)
            players[player].remove(discard)

        elif action_type == 2:  # 弃牌
            discard = get_lo(action_detail)
            paihe[player].append(discard)
            players[player].remove(discard)

            # 记录当前状态
            ret['sequence'].append({
                "player": player,
                "hands": deepcopy(players[player]),
                "fulu": deepcopy(players_fulu[player]),
                "discard": discard,
            })
            last_discard = discard

        elif action_type == 3:  # 吃牌
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
                players[player].remove(lower)
                players[player].remove(middle)
                players_fulu[player].append((upper, lower, middle))

        elif action_type == 4:  # 碰牌
            offer_id = get_offer(action_detail)
            lower = get_pack_tile(action_detail)
            count = 0
            hands_pair = []
            for tile in players[player]:
                if lower & 0xFC == tile & 0xFC:
                    hands_pair.append(tile)
                    players[player].remove(tile)
                    count += 1
                if count == 2:
                    break
            if offer_id == 1:
                players_fulu[player].append([last_discard] + hands_pair)
            elif offer_id == 2:
                players_fulu[player].append([hands_pair[0], last_discard, hands_pair[1]])
            elif offer_id == 3:
                players_fulu[player].append(hands_pair + [last_discard])

        elif action_type == 5:  # 杠牌
            tile = get_pack_tile(action_detail)
            if is_add_kong(action_detail):  # 加杠
                for fulu in players_fulu[player]:
                    if isinstance(fulu, list) and fulu[0] & 0xFC == tile & 0xFC:
                        fulu.append(last_draw)
                        players[player].remove(last_draw)
                        break
            else:  # 明杠或暗杠
                offer_id = get_offer(action_detail)
                if offer_id:
                    players_fulu[player].append(tuple(range(tile, tile + 4)))
                    players[player].remove(last_draw)
                else:
                    players_fulu[player].append(list(range(tile, tile + 4)))
                count = 0
                for hand in players[player]:
                    if hand & 0xFC == tile & 0xFC:
                        players[player].remove(hand)
                        count += 1
                    if count == 3:
                        break

        elif action_type == 6:  # 胡牌
            winner = player
            ret['winner'] = winner
            draw = get_hi(action_detail)

        elif action_type == 7:  # 摸牌
            draw = get_lo(action_detail)
            players[player].append(draw)
            last_draw = draw

    return ret

def main ():
    ''' format v1:
    model input:

    n_fulu01
    type_fulu01
    n_fulu02
    type_fulu02
    ...
    ...
    n_fulu34
    type_fulu34


    n_card01
    n_card02
    ...
    n_card34

    model output:
    discard_card_id
    discard_score = player_score + game_score + hand_score
    '''

    action_list = get_action(10001)
    formatted_print(action_list=action_list['action_list'])

if __name__ == '__main__':
    main()
    l = get_player_discard_sequence(10001)
    print(l)