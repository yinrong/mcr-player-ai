from collections import defaultdict
import itertools
import re

import mj
from functional import seq
import mjutil

MAP = {}
_map_i = 0
def assignMap (N, letter_dict, has_digit):
    n=0
    global _map_i
    while n < N:
        MAP[_map_i] = f'{n+1 if has_digit else ""}{letter_dict[n % len(letter_dict)]}'
        n += 1
        _map_i += 1


assignMap(9, 'm', True)
assignMap(9, 's', True)
assignMap(9, 'p', True)
assignMap(7, 'ESWNCFP', False)

def completeLast (hand, n_lack, blacklist=[]):
    if n_lack > 2:
        return f'缺少{n_lack}张牌，不支持计算'
    all = []
    for all_i in itertools.product(range(34),repeat=n_lack):
        if len(set(all_i).intersection(blacklist)) > 0:
            continue
        hand_add = seq(all_i).map(lambda e: MAP[e]).make_string('')
        text, hu = calc(hand + hand_add, all_i[-1]+1)
        lines = text.split('\n')[1:3]
        if not hu:
            continue

        hand_add_full = seq(all_i).map(lambda e: MAP[e]).make_string('')
        all.append([
            int(lines[0].split('：')[1]), # score
            text,
            render(hand_add_full),
        ])
    ret = (seq(all)
            .sorted(lambda e: e[0], True)
            .map(lambda e: (e[1].split('\n'), e[2]))
            .map(lambda e: [e[0][0],] + [f'等待：{e[1]}',] + e[0][1:])
            .map(lambda e: '\n'.join(e))
            .make_string('\n'))
    return ret

render_base = {
    'm': 9*0,
    's': 9*1,
    'p': 9*2,
    'E': 9*3 + 0,
    'S': 9*3 + 1,
    'W': 9*3 + 2,
    'N': 9*3 + 3,
    'C': 9*3 + 4,
    'F': 9*3 + 5,
    'P': 9*3 + 6,
}
emoji = '🀇🀈🀉🀊🀋🀌🀍🀎🀏🀐🀑🀒🀓🀔🀕🀖🀗🀘🀙🀚🀛🀜🀝🀞🀟🀠🀡🀀🀁🀂🀃🀄🀅🀆'

def renderSimple (n, hand, discard):
    s = ''
    if discard:
        hand.remove(discard)
    for h in hand:
        s += emoji[h - 1]
    if discard:
        s += ' '
        s += emoji[discard - 1]
    print(f'{n}.', s)


def isdigit (c):
    return re.match(r'[0-9]', c)
def render (s):
    if type(s) == list:
        s = encodeForLib(s)
    out = ''
    i = 0
    while i < len(s):
        if s[i] == '|':
            break
        if s[i] in '[]':
            out += '.'
        elif s[i] == ',':
            while s[i] != ']':
                i += 1
            out += '  '
        elif not isdigit(s[i]):
            out += emoji[render_base[s[i]]]
        else:
            j = i + 1
            while j < len(s) and isdigit(s[j]):
                j += 1
            if not isdigit(s[j]):
                base = render_base[s[j]]
                out += emoji[base + int(s[i])-1]
                if j - i == 1:
                    i += 1
            else:
                return '格式化失败：' + s
        i += 1
    return out

encodeForLib_from = 'mspESWNCFP'
encodeForLib_to = '万条饼东南西北中发白'
encodeForLib_dict = {}
for i in range(len(encodeForLib_from)):
    encodeForLib_dict[encodeForLib_from[i]] = encodeForLib_to[i]
def encodeForLib (ids):
    ids.sort()
    ret = ''
    for i in range(len(ids)):
        str0 = MAP[ids[i]]
        if i + 1 < len(ids):
            str1 = MAP[ids[i+1]]
            if len(str0) > 1 and str0[-1] == str1[-1]:
                str0 = str0[:-1]
        ret += str0
    return ret


def calcIds (ids, fulu=''):
    print('in calcIds', ids)
    n = len(ids) + fulu.count(']') * 3
    hand_str = render(ids)
    text = ''
    hu = False
    while True:
        if n == 14:
            hand = encodeForLib(ids)
            ret, hu = calc(fulu + hand, ids[-1])
            if hu:
                text += ret
                break
            MAX_TRY = 1
            for remove_i_list in itertools.combinations(range(len(ids)), MAX_TRY):
                removed = [ids[i] for i in range(len(ids)) if i in remove_i_list]
                kept = [ids[i] for i in range(len(ids)) if i not in remove_i_list]
                new_hand = encodeForLib(kept)
                ret = completeLast(fulu + new_hand, MAX_TRY)
                if ret == '':
                    continue
                text += f'\n打出：{render(removed)}\n' + ret
                text += '\n----------\n'
            if len(text) > 0:
                break
        elif n < 14:
            hand = encodeForLib(ids)
            ret = completeLast(fulu + hand, 14-n)
            if ret != '':
                text += ret
                break
        break
    return dict(text=text, n=n, handlist=ids, hu=hu)
    

def test (s, f):
    if type(f) == list or type(f) == tuple:
        f = seq(f).map(lambda e: str(e)).make_string('\n')
    print('--------------------------')
    print(re.sub(r'^', s + ' | ', str(f), flags=re.MULTILINE))

def calc (a,b):
    try:
        ret = mj.calc(a, b + 1)
        lines = ret.split('\n')
        if len(lines) < 2:
            return ret, False
        hu = int(lines[1].split('：')[1]) == 1
        return '\n'.join(lines[0:1] + lines[2:]), hu
    except Exception as err:
        print('calc err:', err)
        return '', False

SuiteMaker_data = {}
for j in range(3):
    for i in range(9):
        SuiteMaker_data[j*9 + i] = {
            'suite': j,
            'num': i + 1,
        }
for j in range(3,4):
    for i in range(7):
        SuiteMaker_data[j*9 + i] = {
            'suite': j*9 + i,
            'num': i + 1,
        }
class SuiteMaker:
    def __init__ (self, ids):
        self._ids = ids
        self._a = seq(ids).map(lambda e: SuiteMaker_data[e]).to_list()
        self._delim = [0,]
    def get (self):
        self._succ_delim = []
        self._get(0)
        return self._succ_delim
    def _delim2Str (self):
        s = ''
        for i in range(len(self._delim) - 1):
            s += '['
            s += mjutil.encodeForLib(self._ids[self._delim[i] : self._delim[i+1]])
            s += ']'
        self._succ_delim.append(s)
    
    def _get (self, i):
        a = self._a
        if i == len(a):
            self._delim2Str()
            return True
        if i + 3 > len(a):
            return False
        if not (a[i]['suite'] == a[i+1]['suite'] == a[i+2]['suite']):
            return False
        if (i + 3 < len(a)
            and a[i  ]['num']
             == a[i+1]['num']
             == a[i+2]['num']
             == a[i+3]['num']
            and a[i+3]['suite'] == a[i]['suite']
        ):
            self._delim.append(i + 4)
            ret = self._get(i + 4)
            self._delim.pop()
            return ret
        if (a[i]['num'] == a[i+1]['num']   == a[i+2]['num']
         or a[i]['num'] == a[i+1]['num']-1 == a[i+2]['num']-2
        ):
            self._delim.append(i + 3)
            ret = self._get(i + 3)
            self._delim.pop()
            return ret
        return False

if __name__ == '__main__':
    #mj.enableDebug()
    test('1', calc("2334m29s6789pENN", 2))
    
    test('with quotes', calc("[456s,1][456s,1][456s,3]45s55m |EE0000|fah", 15))
    test('standard', calc("1m1m1m3m2m2m2m3m3m4m4m5m5m", 5))
    test('ids13', calcIds([0,0,0,1,1,1,2,2,2,5,6,7,13]))
    test('ids14', calcIds([0,0,0,1,1,1,2,2,2,5,6,7,13,13]))
    test('invalid', calcIds([0,9,0,1,1,1,2,2,2,5,6,7,13,13]))
    test('invalid', calcIds([0,9,0,1,1,1,7,13,13]))
    test('SuiteMaker', SuiteMaker([1,2,3,6,7,8]).get())
    test('with quotes', calc("[NNN][SSS][456s]123s4m", 4))
    test('completeLast-1', completeLast("1m1m1m3m2m2m2m3m3m4m4m6s6s", 1))
    test('completeLast-2', completeLast("1m1m1m3m2m2m2m3m3m4m4m6s", 2))
    test('completeLast-3', completeLast("1m1m1m3m2m2m2m3m3m4m4m", 3))
    test('completeLast-4', completeLast("1m1m1m3m2m2m2m3m3m4m", 4))
    fulu_raw = [32, 32, 32, 27, 27, 27, 3, 3, 3, 21, 22, 23]
    fulu = SuiteMaker(fulu_raw).get()[0]
    hand = [30]
    test('bug1', calcIds(hand, fulu))
    test('replace1', calcIds([0,0,0,1,1,1,2,2,2,5,6,7,13,17]))
