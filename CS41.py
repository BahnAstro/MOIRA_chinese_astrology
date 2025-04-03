import pandas as pd
import numpy as np
import ephem
from datetime import datetime
from bisect import bisect_left
from collections import defaultdict
from tqdm import tqdm
from scipy.optimize import minimize # type: ignore
import logging
import re
import sys
import random
import os  # 新增導入 os 模組

# 配置 logging，設置兩個處理器：一個寫入文件，一個輸出錯誤到控制臺
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 設置全局日誌級別

# 文件處理器，寫入所有日誌（包括 DEBUG）
file_handler = logging.FileHandler('astro_debug.log', mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 控制臺處理器，僅輸出 ERROR 和更高級別的日誌
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 初始化錯誤計數
錯誤計數 = {
    '缺失農曆月份': 0,
    '無效農曆月份': 0,
    '缺失元素關係': 0,
    '未知元素關係': 0
}

# 用於儲存無效的農曆數值
無效農曆值 = []

# 定義五行元素
五行元素 = ['water', 'metal', 'fire', 'wood', 'earth']

# 定義天體因子（初始值）
天體因子 = {
    'sun': 1,
    'moon': 1,
    'mercury': 1,
    'venus': 1,
    'mars': 1,
    'jupiter': 1.5,
    'saturn': 1.5
}

# 定義岐度乘數計算函數
def 計算岐度乘數(celestial, zodiac_degree, mansion_degree, central_degree=None):
    """
    計算天體在岐度時的乘數，範圍為1.1至2.0
    
    參數:
        celestial: 天體名稱
        zodiac_degree: 黃道度數
        mansion_degree: 宿度數
        central_degree: 宿中心度數（如有）
    
    返回:
        乘數值，1.1至2.0
    """
    # 基礎乘數
    base_multiplier = 1.0
    
    # 檢查黃道岐度
    if (0 <= zodiac_degree <= 1.5) or (28.5 <= zodiac_degree <= 30) or (358.5 <= zodiac_degree <= 360):
        # 距離邊界越近，乘數越高
        edge_distance = min(zodiac_degree, 1.5 - zodiac_degree if zodiac_degree <= 1.5 else 
                           30 - zodiac_degree if zodiac_degree <= 30 else 
                           360 - zodiac_degree)
        zodiac_multiplier = 1.1 + 0.4 * (1.5 - min(edge_distance, 1.5)) / 1.5
        base_multiplier = max(base_multiplier, zodiac_multiplier)
    
    # 檢查宿位岐度
    if central_degree is not None:
        distance = abs(mansion_degree - central_degree)
        if distance <= 1.5:
            # 距離中心越近，乘數越高
            mansion_multiplier = 1.1 + 0.9 * (1.5 - distance) / 1.5
            base_multiplier = max(base_multiplier, mansion_multiplier)
    
    # 特定天體的額外權重
    if celestial in ['jupiter', 'saturn']:
        base_multiplier *= 1.1
        
    # 確保乘數不超過2.0
    return min(base_multiplier, 2.0)

# 定義天體與元素的映射（小寫）
天體元素 = {
    'sun': 'sun_element',
    'moon': 'moon_element',
    'mercury': 'metal',
    'venus': 'metal',
    'mars': 'fire',
    'jupiter': 'wood',
    'saturn': 'earth',
    'lilith': 'water',
    'selena': 'metal',
    'moon_north_node': 'fire',
    'moon_south_node': 'earth',
    'asc': 'sun_element',  # 將 asc 映射為 sun_element
    'mc': 'sun_element',   # 將 mc 映射為 sun_element
    'part_of_fortune': 'sun_element',  # 將 part_of_fortune 映射為 sun_element
    'life_degree': 'sun_element',       # 將 life_degree 設置為 sun_element
    'year_degree': None                  # 動態映射，稍後處理
}

# 定義狀態基準分數，旺和相為正分，休、囚、死為負分
狀態基準分數 = {
    '旺': 5,
    '相': 4,
    '休': 1,
    '囚': -1,
    '死': -2
}

def 設定狀態基準分數(旺=5, 相=4, 休=1, 囚=-1, 死=-2):
    global 狀態基準分數
    狀態基準分數 = {
        '旺': 旺,
        '相': 相,
        '休': 休,
        '囚': 囚,
        '死': 死
    }

# 初始化分數字典（小寫鍵）
def 初始化分數():
    return {
        'sun_element': 0,
        'moon_element': 0,
        'water': 0,
        'metal': 0,
        'fire': 0,
        'wood': 0,
        'earth': 0,
        'year_degree_mansion_score': 0,
        'year_degree_zodiac_sign_score': 0,
        'life_degree': 0,
        'aspects_score': 0,
        'conjunction_score': 0,
        'mansion_score': 0,
        'zodiac_sign_score': 0,
        'yunu_score': 0
    }

# 定義 positions, elements_mapping, 和 signs mappings
positions = {
    (203.8374893, 214.4898543): '角宿_Horn',
    (214.4898543, 225.0215638): '亢宿_Neck',
    (225.0215638, 242.9360091): '氐宿_Root',
    (242.9360091, 249.7584245): '房宿_Room',
    (249.7584245, 256.1517382): '心宿_Heart',
    (256.1517382, 271.2575671): '尾宿_Tail',
    (271.2575671, 280.1774751): '箕宿_Winnowing_Basket',
    (280.1774751, 304.0435179): '斗宿_Dipper',
    (304.0435179, 311.7193257): '牛宿_Ox',
    (311.7193257, 323.3911983): '女宿_Girl',
    (323.3911983, 333.348599): '虛宿_Emptiness',
    (333.348599, 9.152166707): '危宿_Rooftop',
    (9.152166707, 22.37214699): '室宿_Encampment',
    (22.37214699, 33.96614257): '奎宿_Legs',
    (33.96614257, 46.93116249): '婁宿_Bond',
    (46.93116249, 59.40804317): '胃宿_Stomach',
    (59.40804317, 68.46117549): '昴宿_Hairy_Head',
    (68.46117549, 83.70296314): '畢宿_Net',
    (83.70296314, 84.67745824): '觜宿_Turtle_Beak',
    (84.67745824, 95.29802279): '參宿_Three_Stars',
    (95.29802279, 125.7245614): '井宿_Well',
    (125.7245614, 130.3004766): '鬼宿_Ghost',
    (130.3004766, 147.275341): '柳宿_Willow',
    (147.275341, 155.6873952): '星宿_Star',
    (155.6873952, 173.6855706): '張宿_Extended_Net',
    (173.6855706, 190.7217613): '翼宿_Wings',
    (190.7217613, 203.8374893): '軫宿_Chariot'
}

elements_mapping = {
    'wood': ['角宿_Horn', '斗宿_Dipper', '奎宿_Legs', '井宿_Well'],
    'metal': ['亢宿_Neck', '牛宿_Ox', '婁宿_Bond', '鬼宿_Ghost'],
    'earth': ['氐宿_Root', '女宿_Girl', '胃宿_Stomach', '柳宿_Willow'],
    'sun_element': ['房宿_Room', '虛宿_Emptiness', '昴宿_Hairy_Head', '星宿_Star'],
    'moon_element': ['心宿_Heart', '危宿_Rooftop', '畢宿_Net', '張宿_Extended_Net'],
    'fire': ['尾宿_Tail', '室宿_Encampment', '觜宿_Turtle_Beak', '翼宿_Wings'],
    'water': ['箕宿_Winnowing_Basket', '參宿_Three_Stars', '軫宿_Chariot']
}

signs = {
    (0, 30): 'Aries_fire',
    (30, 60): 'Taurus_metal',
    (60, 90): 'Gemini_water',
    (90, 120): 'Cancer_moon',
    (120, 150): 'Leo_sun',
    (150, 180): 'Virgo_water',
    (180, 210): 'Libra_metal',
    (210, 240): 'Scorpio_fire',
    (240, 270): 'Sagittarius_wood',
    (270, 300): 'Capricorn_earth',
    (300, 330): 'Aquarius_earth',
    (330, 360): 'Pisces_wood'
}

# 定義吉星和凶星
吉星 = set()
凶星 = set()

def 建立五行能量字典(five_energy_chart, 元素到天體):
    five_energy_dict = defaultdict(lambda: defaultdict(dict))
    celestial_columns = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']

    for idx, row in five_energy_chart.iterrows():
        try:
            lunar_month_num = int(row['農曆月份'])
        except (ValueError, TypeError):
            logger.warning(f"行 {idx}: 農曆月份 '{row['農曆月份']}' 無法轉換為整數。")
            錯誤計數['無效農曆月份'] += 1
            無效農曆值.append(row.get('農曆月份', '未知'))
            continue
        element = row['element'].strip().lower()
        if element not in 五行元素 and element not in 天體元素.values():
            logger.warning(f"行 {idx}: 元素 '{element}' 不在五行元素或天體元素列表中。")
            錯誤計數['未知元素關係'] += 1
            continue
        for celestial in celestial_columns:
            value = row.get(celestial, 0)
            try:
                value = int(value) if pd.notnull(value) else 0
            except ValueError:
                logger.warning(f"行 {idx}: 天體 {celestial} 的值 '{value}' 無法轉換為整數。")
                value = 0
            five_energy_dict[lunar_month_num][element][celestial] = value
            logger.debug(f"行 {idx}: 農曆{lunar_month_num}月，元素 {element} 天體 {celestial} 值 {value}")
    logger.info("五行能量字典建立完成。")
    return five_energy_dict

def 創建元素到天體映射(天體元素):
    元素到天體 = defaultdict(list)
    for celestial, element in 天體元素.items():
        if element:
            元素到天體[element].append(celestial)
    return 元素到天體

def is_qidu(zodiac_degree, mansion_degree, central_degrees):
    # 檢查轉換宮位（zodiac sign）區域：28.5至30度,0至1.5度，或358.5至360度
    if (0 <= zodiac_degree <= 1.5) or (28.5 <= zodiac_degree <= 30) or (358.5 <= zodiac_degree <= 360):
        return True
    
    # 檢查28宿星宿（mansion）中心度數附近的區域
    for central_degree in central_degrees:
        if (central_degree - 1.5) <= mansion_degree <= (central_degree + 1.5):
            return True
    
    return False

def 計算太陽分數(datetime_obj, 太陽因子):
    import random

    # 定義分數範圍，確保巳 > 午 > 未 > 辰 > 卯 > 寅 > 申 > 酉
    時辰分數範圍 = {
        '巳': (1, 6),   # 巳時分數範圍
        '午': (1, 6),   # 午時分數範圍
        '未': (1, 5),   # 未時分數範圍
        '辰': (1, 5),   # 辰時分數範圍
        '卯': (1, 5),   # 卯時分數範圍
        '寅': (1, 5),   # 寅時分數範圍
        '申': (1, 5),   # 申時分數範圍
        '酉': (1, 5)    # 酉時分數範圍
    }

    時辰隨機分數範圍 = {
        '戌': (0.8, 1.1),
        '亥': (0.8, 1.1),
        '子': (0.8, 1.1),
        '丑': (0.8, 1.1)
    }

    時辰列表 = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
    時辰開始時間 = [23, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    current_時辰 = '子'  # 默認為子時
    hour = datetime_obj.hour
    for i, start_hour in enumerate(時辰開始時間):
        end_hour = (start_hour + 2) % 24
        if start_hour < end_hour:
            if start_hour <= hour < end_hour:
                current_時辰 = 時辰列表[i]
                break
        else:
            if hour >= start_hour or hour < end_hour:
                current_時辰 = 時辰列表[i]
                break

    if current_時辰 in 時辰分數範圍:
        min_score, max_score = 時辰分數範圍[current_時辰]
        sun_score = random.uniform(min_score, max_score)
    elif current_時辰 in 時辰隨機分數範圍:
        min_score, max_score = 時辰隨機分數範圍[current_時辰]
        sun_score = random.uniform(min_score, max_score)
    else:
        sun_score = 0  # 默認分數

    logger.debug(f"當前時辰: {current_時辰}, 太陽分數: {sun_score}")
    return sun_score * 太陽因子

def 計算月亮分數(datetime_obj, 月亮因子):
    import random

    observer = ephem.Observer()
    observer.date = datetime_obj.strftime('%Y/%m/%d %H:%M:%S')  # ephem 需要字符串格式
    observer.lat = '0'
    observer.lon = '0'

    sun = ephem.Sun(observer)
    moon = ephem.Moon(observer)

    sun_angle = np.degrees(sun.ra) % 360
    moon_angle = np.degrees(moon.ra) % 360
    angle_diff = abs(sun_angle - moon_angle)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    月亮分數範圍 = [
        ((150, 180), (3, 5.5)),   # 滿月附近
        ((90, 150), (3, 4.5)),    # 盈凸月
        ((30, 90), (2, 3.5)),     # 上/下弦月
        ((0, 30), (2, 3.5)),      # 新月附近
        ((210, 270), (1, 3)),   # 殘月
        ((270, 330), (0.5, 2.5)),   # 下弦月
        ((330, 360), (0.5, 2.5))    # 朔月
    ]

    moon_score = 0  # 默認分數
    for angle_range, score_range in 月亮分數範圍:
        min_angle, max_angle = angle_range
        if min_angle <= angle_diff < max_angle:
            min_score, max_score = score_range
            moon_score = random.uniform(min_score, max_score)
            break

    logger.debug(f"日月角度差: {angle_diff}, 月亮分數: {moon_score}")
    return moon_score * 月亮因子
def 獲取關係符號(element1, element2, month_relations, 元素到天體):
    if not element1 or not isinstance(element1, str):
        錯誤計數['未知元素關係'] += 1
        logger.warning(f"未知的元素關係: element1 是 {element1}，對 {element2}")
        return 0

    element1 = element1.strip().lower()
    element2 = element2.strip().lower()

    celestial_list = 元素到天體.get(element2, [])
    if not celestial_list:
        錯誤計數['未知元素關係'] += 1
        logger.warning(f"未知的元素關係: {element1} 對 {element2}")
        return 0

    relation_sign = sum(month_relations.get(element1, {}).get(celestial, 0) for celestial in celestial_list)
    relation_sign = max(min(relation_sign, 1), -1)

    if relation_sign not in [1, -1, 0]:
        logger.warning(f"無效的關係符號 {relation_sign} for {element1} 對 {element2} ({celestial_list})")
        return 0
    return relation_sign

def 獲取節氣索引(datetime_obj):
    節氣日期 = [
        '01-05', '01-20', '02-04', '02-19', '03-06', '03-21',
        '04-05', '04-20', '05-06', '05-21', '06-06', '06-21',
        '07-07', '07-22', '08-07', '08-23', '09-08', '09-23',
        '10-08', '10-23', '11-07', '11-22', '12-07', '12-22'
    ]
    try:
        日期列表 = [datetime.strptime(f"{datetime_obj.year}-{d}", '%Y-%m-%d') for d in 節氣日期]
    except ValueError as e:
        logger.error(f"節氣日期格式錯誤: {e}")
        return -1
    日期列表.append(datetime.strptime(f"{datetime_obj.year + 1}-01-05", '%Y-%m-%d'))  # 下一年的小寒

    current_date = datetime_obj.replace(tzinfo=None)
    節氣索引 = bisect_left(日期列表, current_date) - 1
    if 節氣索引 == -1:
        節氣索引 = 23  # 當前日期在小寒之前，視為上一年的冬至
    return 節氣索引 % 24

def 獲取行星能量值(節氣索引):
    # 根據節氣索引從五行能量字典中獲取行星能量值
    # 此處留空，根據實際需求填寫
    return {}

def 檢查相位(row, celestial1, celestial2):
    aspect_col = f'birth_{celestial1}_vs_current_{celestial2}_aspect_count'
    count = row.get(aspect_col, 0)
    return count > 0

def 計算特殊組合分數(行星列表, lunar_month, row, five_energy_dict, element_state_map):
    特殊組合 = {
        1: [
            (['venus', 'mercury', 'mars'], 1),
            (['venus', 'mars', 'saturn'], 1),
            (['mars', 'venus', 'jupiter'], 1),
            (['saturn', 'mercury', 'mars'], 1),
            (['jupiter', 'mars', 'saturn'], 1),
            (['saturn', 'venus', 'jupiter'], 1),
            (['mars', 'venus', 'mercury'], -1)
        ],
        2: [
            (['mercury', 'jupiter', 'venus'], 1),
            (['jupiter', 'mars', 'saturn'], 1),
            (['saturn', 'jupiter', 'venus'], 1),
            (['venus', 'mercury', 'saturn'], 1),
            (['jupiter', 'saturn', 'venus', 'mercury'], -2)
        ],
        3: [
            (['venus', 'saturn', 'mercury'], 1),
            (['mars', 'mercury', 'saturn'], 1),
            (['saturn', 'mercury', 'jupiter'], -1),
            (['saturn', 'jupiter', 'venus'], 1),
            (['mars_water_zodiac_mansion'], 1),  # 特殊處理
            (['mars_mercury_conjunction_aspects'], -1)  # 特殊處理
        ],
        4: [
            (['venus', 'mars', 'saturn'], 1),
            (['mars', 'mercury', 'jupiter'], 1),
            (['saturn', 'venus', 'jupiter'], 1),
            (['saturn', 'venus', 'mercury'], -1)
        ],
        5: [
            (['venus', 'mars', 'saturn'], 1),
            (['saturn', 'jupiter', 'mars'], -1),
            (['saturn', 'venus', 'mercury'], 1),
            (['mars', 'mercury', 'saturn'], 1),
            (['mars', 'mercury', 'jupiter'], 1)
        ],
        6: [
            (['venus', 'mars', 'saturn'], 1),
            (['saturn', 'sun', 'mercury'], 1),
            (['saturn', 'mars', 'mercury'], 1)
        ],
        7: [
            (['venus', 'mars', 'saturn'], 1),
            (['jupiter', 'mercury', 'venus'], 1),
            (['saturn', 'mars', 'mercury'], 1)
        ],
        8: [
            (['sun', 'venus', 'mercury'], 1),
            (['moon', 'saturn'], 1),  # 特殊處理: When New/Waning Moon
            (['mercury', 'venus', 'mars'], 1),
            (['mercury', 'saturn', 'venus'], 1),
            (['mars', 'venus', 'jupiter'], 1),
            (['mars', 'saturn', 'jupiter'], 1)
        ],
        9: [
            (['mercury', 'venus', 'mars'], 1),
            (['mercury', 'saturn', 'venus'], 1),
            (['mars', 'venus', 'jupiter'], 1),
            (['mars', 'saturn', 'jupiter'], 1)
        ],
        10: [
            (['mercury', 'venus', 'saturn'], 1),
            (['mercury', 'jupiter', 'saturn'], 1),
            (['mercury', 'venus', 'mars'], 1),
            (['venus', 'saturn', 'mars'], 1),
            (['venus', 'mercury', 'mars'], 1),
            (['mars', 'mercury', 'jupiter'], 1),
            (['jupiter', 'mars', 'saturn'], 2),
            (['saturn', 'lilith', 'jupiter'], 1),
            (['saturn', 'lilith', 'jupiter', 'mars'], 1),
            (['saturn', 'jupiter', 'mars'], 1)
        ],
        11: [
            (['mercury', 'venus', 'saturn'], 1),
            (['mercury', 'jupiter', 'saturn'], 1),
            (['mercury', 'venus', 'mars'], 1),
            (['venus', 'saturn', 'mars'], 1),
            (['venus', 'mercury', 'mars'], 1),
            (['mars', 'mercury', 'jupiter'], 1),
            (['jupiter', 'mars', 'saturn'], 2),
            (['saturn', 'lilith', 'jupiter'], 1),
            (['saturn', 'lilith', 'jupiter', 'mars'], 1),
            (['saturn', 'jupiter', 'mars'], 1)
        ],
        12: [
            (['mercury', 'venus', 'saturn'], 1),
            (['mercury', 'jupiter', 'saturn'], 1),
            (['mercury', 'venus', 'mars'], 1),
            (['venus', 'saturn', 'mars'], 1),
            (['venus', 'mercury', 'mars'], 1),
            (['mars', 'mercury', 'jupiter'], 1),
            (['jupiter', 'mars', 'saturn'], 2),
            (['saturn', 'lilith', 'jupiter'], 1),
            (['saturn', 'lilith', 'jupiter', 'mars'], 1),
            (['saturn', 'jupiter', 'mars'], 1)
        ]
    }
    月份組合 = 特殊組合.get(lunar_month, [])
    total_score = 0

    for combo, base_score in 月份組合:
        if isinstance(combo, list):
            if all(planet in 行星列表 for planet in combo):
                elements = [天體元素.get(planet) for planet in combo]
                states = [狀態基準分數.get(element_state_map.get(elem, '休'), 1) for elem in elements]
                relation_score = sum(states)
                final_score = base_score * relation_score
                
                # 檢查組合中的行星是否有處於岐度的
                qidu_multiplier = 1.0
                for planet in combo:
                    try:
                        # 獲取天體的岐度乘數
                        zodiac_degree = float(row.get(f'{planet}_zodiac_degree', 0))
                        mansion_degree = float(row.get(f'{planet}_mansion_degree', 0))
                        mansion_position = row.get(f'{planet}_mansion_positions', '')
                        central_degree = None
                        if pd.notnull(mansion_position) and mansion_position in mansion_central_degrees:
                            central_degree = mansion_central_degrees[mansion_position]
                        
                        # 計算岐度乘數
                        planet_multiplier = 計算岐度乘數(planet, zodiac_degree, mansion_degree, central_degree)
                        qidu_multiplier = max(qidu_multiplier, planet_multiplier)
                    except Exception as e:
                        logger.warning(f"行 {row.name}: 計算 {planet} 岐度乘數時出錯: {e}")
                
                # 應用岐度乘數
                final_score *= qidu_multiplier
                if qidu_multiplier > 1.0:
                    logger.debug(f"行 {row.name}: 特殊組合 {combo} 中有岐度行星，乘數 {qidu_multiplier}")
                
                total_score += final_score
                logger.debug(f"行 {row.name}: 特殊組合 {combo} 對應分數 {final_score}")
        elif isinstance(combo, str):
            # 處理特殊情況
            if combo == 'mars_water_zodiac_mansion' and 'mars' in 行星列表:
                score = 1  # 根據具體需求調整
                
                # 檢查火星是否在岐度
                try:
                    zodiac_degree = float(row.get('mars_zodiac_degree', 0))
                    mansion_degree = float(row.get('mars_mansion_degree', 0))
                    mansion_position = row.get('mars_mansion_positions', '')
                    central_degree = None
                    if pd.notnull(mansion_position) and mansion_position in mansion_central_degrees:
                        central_degree = mansion_central_degrees[mansion_position]
                    
                    qidu_multiplier = 計算岐度乘數('mars', zodiac_degree, mansion_degree, central_degree)
                    score *= qidu_multiplier
                    if qidu_multiplier > 1.0:
                        logger.debug(f"行 {row.name}: 特殊組合 {combo} 中的火星在岐度，乘數 {qidu_multiplier}")
                except Exception as e:
                    logger.warning(f"行 {row.name}: 計算火星岐度乘數時出錯: {e}")
                
                total_score += score
                logger.debug(f"行 {row.name}: 特殊組合 {combo} 對應分數 {score}")
            elif combo == 'mars_mercury_conjunction_aspects' and 'mars' in 行星列表 and 'mercury' in 行星列表:
                score = -1  # 根據具體需求調整
                
                # 檢查火星和水星是否在岐度
                try:
                    # 火星岐度檢查
                    mars_zodiac_degree = float(row.get('mars_zodiac_degree', 0))
                    mars_mansion_degree = float(row.get('mars_mansion_degree', 0))
                    mars_mansion_position = row.get('mars_mansion_positions', '')
                    mars_central_degree = None
                    if pd.notnull(mars_mansion_position) and mars_mansion_position in mansion_central_degrees:
                        mars_central_degree = mansion_central_degrees[mars_mansion_position]
                    
                    mars_multiplier = 計算岐度乘數('mars', mars_zodiac_degree, mars_mansion_degree, mars_central_degree)
                    
                    # 水星岐度檢查
                    mercury_zodiac_degree = float(row.get('mercury_zodiac_degree', 0))
                    mercury_mansion_degree = float(row.get('mercury_mansion_degree', 0))
                    mercury_mansion_position = row.get('mercury_mansion_positions', '')
                    mercury_central_degree = None
                    if pd.notnull(mercury_mansion_position) and mercury_mansion_position in mansion_central_degrees:
                        mercury_central_degree = mansion_central_degrees[mercury_mansion_position]
                    
                    mercury_multiplier = 計算岐度乘數('mercury', mercury_zodiac_degree, mercury_mansion_degree, mercury_central_degree)
                    
                    # 取較大的乘數
                    qidu_multiplier = max(mars_multiplier, mercury_multiplier)
                    score *= qidu_multiplier
                    if qidu_multiplier > 1.0:
                        logger.debug(f"行 {row.name}: 特殊組合 {combo} 中有岐度行星，乘數 {qidu_multiplier}")
                except Exception as e:
                    logger.warning(f"行 {row.name}: 計算火星或水星岐度乘數時出錯: {e}")
                
                total_score += score
                logger.debug(f"行 {row.name}: 特殊組合 {combo} 對應分數 {score}")
    
    return total_score

def 計算_aspects_score(row, 額外分數, 月份關係, 元素到天體, 天體元素, 天體乘數, element_state_map):
    aspects_columns = [col for col in row.index if re.search(r'_aspect_count$', col, re.IGNORECASE)]
    for col in aspects_columns:
        count = row.get(col, 0)
        if pd.notnull(count) and count > 0:
            cleaned_col = re.sub(r'_aspect_count$', '', col, re.IGNORECASE)
            parts = re.split(r'_vs_', cleaned_col, re.IGNORECASE)
            if len(parts) == 2:
                planet1_name = re.sub(r'^birth_', '', parts[0], re.IGNORECASE).lower()
                planet2_name = re.sub(r'^current_', '', parts[1], re.IGNORECASE).lower()
            else:
                continue

            element1 = 天體元素.get(planet1_name, None)
            element2 = 天體元素.get(planet2_name, None)

            if element1 and element2:
                relation_sign = 獲取關係符號(element1, element2, 月份關係, 元素到天體)
                element2_state = element_state_map.get(element2.lower(), '休')
                element2_score = 狀態基準分數.get(element2_state, 1)
                factor = min(天體因子.get(planet1_name, 1), 3)
                score = relation_sign * element2_score * factor * count

                # 根據分數決定吉星或凶星
                if score > 0:
                    吉星.update([planet1_name, planet2_name])
                elif score < 0:
                    凶星.update([planet1_name, planet2_name])

                # 調整分數基於吉星和凶星的距離
                try:
                    elong1 = float(row.get(f'{planet1_name}_elong', 0)) % 360
                    elong2 = float(row.get(f'{planet2_name}_elong', 0)) % 360
                    distance = min(abs(elong1 - elong2), 360 - abs(elong1 - elong2))
                    increment = 1.1 + ((30 - (distance % 3)) // 3) * 0.05
                    increment = min(max(increment, 1.1), 1.5)
                    score *= increment
                except ValueError:
                    logger.warning(f"行 {row.name}: 計算距離時出錯。")

                # 檢查是否在岐度，使用岐度乘數
                try:
                    # 獲取planet1的岐度乘數
                    zodiac_degree1 = float(row.get(f'{planet1_name}_zodiac_degree', 0))
                    mansion_degree1 = float(row.get(f'{planet1_name}_mansion_degree', 0))
                    mansion_position1 = row.get(f'{planet1_name}_mansion_positions', '')
                    central_degree1 = None
                    if pd.notnull(mansion_position1) and mansion_position1 in mansion_central_degrees:
                        central_degree1 = mansion_central_degrees[mansion_position1]
                    
                    # 獲取planet2的岐度乘數
                    zodiac_degree2 = float(row.get(f'{planet2_name}_zodiac_degree', 0))
                    mansion_degree2 = float(row.get(f'{planet2_name}_mansion_degree', 0))
                    mansion_position2 = row.get(f'{planet2_name}_mansion_positions', '')
                    central_degree2 = None
                    if pd.notnull(mansion_position2) and mansion_position2 in mansion_central_degrees:
                        central_degree2 = mansion_central_degrees[mansion_position2]
                    
                    # 計算岐度乘數
                    multiplier1 = 計算岐度乘數(planet1_name, zodiac_degree1, mansion_degree1, central_degree1)
                    multiplier2 = 計算岐度乘數(planet2_name, zodiac_degree2, mansion_degree2, central_degree2)
                    
                    # 取較大的乘數
                    qidu_multiplier = max(multiplier1, multiplier2)
                    score *= qidu_multiplier
                    
                    if qidu_multiplier > 1.0:
                        logger.debug(f"行 {row.name}: {planet1_name} 和 {planet2_name} 在岐度，乘數 {qidu_multiplier}")
                except Exception as e:
                    logger.warning(f"行 {row.name}: 計算岐度乘數時出錯: {e}")

                額外分數['aspects_score'] += score

def 計算_conjunction_score(row, 額外分數, 月份關係, 元素到天體, 天體元素, 天體乘數, element_state_map):
    conjunction_columns = [col for col in row.index if re.search(r'_conjunction_count$', col, re.IGNORECASE)]
    for col in conjunction_columns:
        count = row.get(col, 0)
        if pd.notnull(count) and count > 0:
            cleaned_col = re.sub(r'_conjunction_count$', '', col, re.IGNORECASE)
            parts = re.split(r'_vs_', cleaned_col, re.IGNORECASE)
            if len(parts) == 2:
                planet1_name = re.sub(r'^birth_', '', parts[0], re.IGNORECASE).lower()
                planet2_name = re.sub(r'^current_', '', parts[1], re.IGNORECASE).lower()
            else:
                continue

            element1 = 天體元素.get(planet1_name, None)
            element2 = 天體元素.get(planet2_name, None)

            if element1 and element2:
                relation_sign = 獲取關係符號(element1, element2, 月份關係, 元素到天體)
                element2_state = element_state_map.get(element2.lower(), '休')
                element2_score = 狀態基準分數.get(element2_state, 1)
                factor = min(天體因子.get(planet1_name, 1), 3)
                score = relation_sign * element2_score * factor * count * 2  # 合相分數加倍

                # 根據分數決定吉星或凶星
                if score > 0:
                    吉星.update([planet1_name, planet2_name])
                elif score < 0:
                    凶星.update([planet1_name, planet2_name])

                # 調整分數基於吉星和凶星的距離
                try:
                    elong1 = float(row.get(f'{planet1_name}_elong', 0)) % 360
                    elong2 = float(row.get(f'{planet2_name}_elong', 0)) % 360
                    distance = min(abs(elong1 - elong2), 360 - abs(elong1 - elong2))
                    increment = 1.1 + ((30 - (distance % 3)) // 3) * 0.05
                    increment = min(max(increment, 1), 1.5)
                    score *= increment
                except ValueError:
                    logger.warning(f"行 {row.name}: 計算距離時出錯。")

                # 檢查是否在岐度，使用岐度乘數
                try:
                    # 獲取planet1的岐度乘數
                    zodiac_degree1 = float(row.get(f'{planet1_name}_zodiac_degree', 0))
                    mansion_degree1 = float(row.get(f'{planet1_name}_mansion_degree', 0))
                    mansion_position1 = row.get(f'{planet1_name}_mansion_positions', '')
                    central_degree1 = None
                    if pd.notnull(mansion_position1) and mansion_position1 in mansion_central_degrees:
                        central_degree1 = mansion_central_degrees[mansion_position1]
                    
                    # 獲取planet2的岐度乘數
                    zodiac_degree2 = float(row.get(f'{planet2_name}_zodiac_degree', 0))
                    mansion_degree2 = float(row.get(f'{planet2_name}_mansion_degree', 0))
                    mansion_position2 = row.get(f'{planet2_name}_mansion_positions', '')
                    central_degree2 = None
                    if pd.notnull(mansion_position2) and mansion_position2 in mansion_central_degrees:
                        central_degree2 = mansion_central_degrees[mansion_position2]
                    
                    # 計算岐度乘數
                    multiplier1 = 計算岐度乘數(planet1_name, zodiac_degree1, mansion_degree1, central_degree1)
                    multiplier2 = 計算岐度乘數(planet2_name, zodiac_degree2, mansion_degree2, central_degree2)
                    
                    # 取較大的乘數
                    qidu_multiplier = max(multiplier1, multiplier2)
                    score *= qidu_multiplier
                    
                    if qidu_multiplier > 1.0:
                        logger.debug(f"行 {row.name}: {planet1_name} 和 {planet2_name} 在岐度，乘數 {qidu_multiplier}")
                except Exception as e:
                    logger.warning(f"行 {row.name}: 計算岐度乘數時出錯: {e}")

                額外分數['conjunction_score'] += score

def 計算_yunu_score(row, 分數, element_state_map, 天體因子):
    for element in 五行元素:
        element_state = element_state_map.get(element, '休')
        element_score = 狀態基準分數.get(element_state, 1)

        guard_count = row.get(f'{element}_yunu_guard_count', 0)
        if pd.notnull(guard_count) and guard_count > 0 and element_state == '旺':
            yunu_score = guard_count * (element_score / 2)
            分數['yunu_score'] += yunu_score

        offend_count = row.get(f'{element}_yunu_offend_count', 0)
        if pd.notnull(offend_count) and offend_count > 0 and element_state != '旺':
            yunu_score = offend_count * (element_score / 2) * -1
            分數['yunu_score'] += yunu_score

def 計算分數(row, five_energy_dict, 天體元素, 元素到天體, 天體乘數, 太陽因子, 月亮因子):
    分數 = 初始化分數()
    額外分數 = {
        'aspects_score': 0,
        'conjunction_score': 0,
        'mansion_score': 0,
        'zodiac_sign_score': 0,
        'yunu_score': 0
    }

    element_state_map = {}

    try:
        lunar_month = int(row['農曆月份'])
        for element, celestial_relations in five_energy_dict.get(lunar_month, {}).items():
            for celestial, relation_value in celestial_relations.items():
                state = '旺' if relation_value == 1 else '死' if relation_value == -1 else '休'
                element_key = 天體元素.get(celestial, '').lower()
                if element_key:
                    element_state_map[element_key] = state
    except (ValueError, TypeError):
        錯誤計數['缺失農曆月份'] += 1
        無效農曆值.append(row.get('農曆月份', '未知'))
        return 分數, 額外分數, 吉星, 凶星, element_state_map, 0

    月份資料 = five_energy_dict.get(lunar_month, {})
    if not 月份資料:
        錯誤計數['缺失元素關係'] += 1
        return 分數, 額外分數, 吉星, 凶星, element_state_map, 0

    for element in 五行元素:
        if element in 月份資料:
            for celestial, relation_value in 月份資料[element].items():
                分數[element] += relation_value

    # 收集處於岐度的天體
    qidu_celestials = set()
    mansion_central_degrees = {
        '角宿_Horn': 203.8374893,
        '亢宿_Neck': 214.4898543,
        '氐宿_Root': 225.0215638,
        '房宿_Room': 242.9360091,
        '心宿_Heart': 249.7584245,
        '尾宿_Tail': 256.1517382,
        '箕宿_Winnowing_Basket': 271.2575671,
        '斗宿_Dipper': 280.1774751,
        '牛宿_Ox': 304.0435179,
        '女宿_Girl': 311.7193257,
        '虛宿_Emptiness': 323.3911983,
        '危宿_Rooftop': 333.348599,
        '室宿_Encampment': 9.152166707,
        '奎宿_Legs': 22.37214699,
        '婁宿_Bond': 33.96614257,
        '胃宿_Stomach': 46.93116249,
        '昴宿_Hairy_Head': 59.40804317,
        '畢宿_Net': 68.46117549,
        '觜宿_Turtle_Beak': 83.70296314,
        '參宿_Three_Stars': 84.67745824,
        '井宿_Well': 95.29802279,
        '鬼宿_Ghost': 125.7245614,
        '柳宿_Willow': 130.3004766,
        '星宿_Star': 147.275341,
        '張宿_Extended_Net': 155.6873952,
        '翼宿_Wings': 173.6855706,
        '軫宿_Chariot': 190.7217613
    }
    celestial_bodies = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn',
                        'lilith', 'selena', 'moon_north_node', 'moon_south_node',
                        'asc', 'mc', 'part_of_fortune', 'life_degree', 'year_degree']

    for celestial in celestial_bodies:
        zodiac_degree_col = f'{celestial}_zodiac_degree'
        mansion_degree_col = f'{celestial}_mansion_degree'
        mansion_position_col = f'{celestial}_mansion_positions'

        zodiac_degree = row.get(zodiac_degree_col, None)
        mansion_degree = row.get(mansion_degree_col, None)
        mansion_position = row.get(mansion_position_col, None)

        if pd.notnull(zodiac_degree) and pd.notnull(mansion_degree) and pd.notnull(mansion_position):
            try:
                zodiac_degree = float(zodiac_degree)
                mansion_degree = float(mansion_degree)
                central_degree = mansion_central_degrees.get(mansion_position, None)
                if central_degree and is_qidu(zodiac_degree, mansion_degree, [central_degree]):
                    qidu_celestials.add(celestial)
            except ValueError:
                錯誤計數['未知元素關係'] += 1

    # 計算太陽和月亮的分數
    if 'datetime(utc)' in row and pd.notnull(row['datetime(utc)']):
        try:
            datetime_obj = pd.to_datetime(row['datetime(utc)'], utc=True, errors='coerce')
            if pd.isnull(datetime_obj):
                raise ValueError("datetime(utc) 無法解析。")
            datetime_obj = datetime_obj.tz_convert(None)  # 轉換為 naive datetime
            sun_score = 計算太陽分數(datetime_obj, 太陽因子)
            moon_score = 計算月亮分數(datetime_obj, 月亮因子)
            分數['sun_element'] += sun_score
            分數['moon_element'] += moon_score

            節氣索引 = 獲取節氣索引(datetime_obj)
            if 節氣索引 != -1:
                行星能量值 = 獲取行星能量值(節氣索引)
                for planet, energy in 行星能量值.items():
                    if 天體元素.get(planet):
                        分數[天體元素[planet]] += energy
            else:
                錯誤計數['未知元素關係'] += 1
        except Exception as e:
            logger.error(f"行 {row.name}: 解析 DateTime(utc) 時出錯: {row['datetime(utc)']} - {e}")
            錯誤計數['缺失元素關係'] += 1
    else:
        錯誤計數['缺失元素關係'] += 1
        無效農曆值.append(row.get('農曆月份', '未知'))

    # 計算 aspects_score 和 conjunction_score
    月份關係 = five_energy_dict.get(lunar_month, {})

    計算_aspects_score(row, 額外分數, 月份關係, 元素到天體, 天體元素, 天體乘數, element_state_map)
    計算_conjunction_score(row, 額外分數, 月份關係, 元素到天體, 天體元素, 天體乘數, element_state_map)

    # 計算 yunu_score
    計算_yunu_score(row, 分數, element_state_map, 天體因子)

    # 計算特殊組合分數
    try:
        lunar_month = int(row['農曆月份'])
    except (ValueError, TypeError):
        lunar_month = None
    if lunar_month and lunar_month in five_energy_dict:
        行星列表 = set(天體元素.keys()).intersection(
            celestial for element_relations in five_energy_dict.get(lunar_month, {}).values() for celestial in element_relations.keys()
        )
        特殊分數 = 計算特殊組合分數(行星列表, lunar_month, row, five_energy_dict, element_state_map)
        額外分數['aspects_score'] += 特殊分數
    else:
        logger.warning(f"行 {row.name}: 農曆月份 {lunar_month} 不存在於五行能量字典中。")

    # 計算 year_degree_mansion_score
    year_degree_wuxing = row.get('year_degree_wuxing', None)
    year_degree_mansion_element = row.get('year_degree_mansion_element', None)

    if pd.notnull(year_degree_wuxing) and pd.notnull(year_degree_mansion_element):
        year_degree_wuxing = year_degree_wuxing.strip().lower()
        year_degree_mansion_element = year_degree_mansion_element.strip().lower()

        # 定義 year_degree_wuxing 到天體的映射
        year_degree_wuxing_map = {
            'water': 'mercury',
            'metal': 'venus',
            'fire': 'mars',
            'wood': 'jupiter',
            'earth': 'saturn'
        }
        mapped_celestial = year_degree_wuxing_map.get(year_degree_wuxing, None)

        if mapped_celestial:
            # 獲取農曆月份的五行能量關係
            relation_value = five_energy_dict.get(lunar_month, {}).get(year_degree_mansion_element, {}).get(mapped_celestial, 0)

            # 考慮元素的狀態
            state = element_state_map.get(year_degree_mansion_element, '休')
            state_score = 狀態基準分數.get(state, 1)

            # 計算 year_degree_mansion_score
            year_degree_mansion_score = relation_value * state_score

            分數['year_degree_mansion_score'] += year_degree_mansion_score
            logger.debug(f"行 {row.name}: year_degree_wuxing={year_degree_wuxing}, mapped_celestial={mapped_celestial}, year_degree_mansion_element={year_degree_mansion_element}, relation_value={relation_value}, state_score={state_score}, year_degree_mansion_score={year_degree_mansion_score}")
        else:
            logger.warning(f"行 {row.name}: 無法映射 year_degree_wuxing='{year_degree_wuxing}' 到天體。")
            錯誤計數['未知元素關係'] += 1
    else:
        logger.warning(f"行 {row.name}: year_degree_wuxing 或 year_degree_mansion_element 缺失。")
        錯誤計數['缺失元素關係'] += 1

    # 計算 year_degree_zodiac_sign_element 分數
    year_degree_zodiac_sign_element = row.get('year_degree_zodiac_sign_element', None)

    if pd.notnull(year_degree_zodiac_sign_element):
        year_degree_zodiac_sign_element = year_degree_zodiac_sign_element.strip().lower()

        # 定義 year_degree_zodiac_sign_element 到天體的映射
        year_degree_zodiac_sign_map = {
            'water': 'mercury',
            'metal': 'venus',
            'fire': 'mars',
            'wood': 'jupiter',
            'earth': 'saturn'
        }
        mapped_zodiac_celestial = year_degree_zodiac_sign_map.get(year_degree_zodiac_sign_element, None)

        if mapped_zodiac_celestial:
            # 獲取農曆月份的五行能量關係
            relation_value = five_energy_dict.get(lunar_month, {}).get(year_degree_zodiac_sign_element, {}).get(mapped_zodiac_celestial, 0)

            # 考慮元素的狀態
            state = element_state_map.get(year_degree_zodiac_sign_element, '休')
            state_score = 狀態基準分數.get(state, 1)

            # 計算 year_degree_zodiac_sign_score
            year_degree_zodiac_sign_score = relation_value * state_score

            分數['year_degree_zodiac_sign_score'] += year_degree_zodiac_sign_score
            logger.debug(f"行 {row.name}: year_degree_zodiac_sign_element={year_degree_zodiac_sign_element}, mapped_zodiac_celestial={mapped_zodiac_celestial}, relation_value={relation_value}, state_score={state_score}, year_degree_zodiac_sign_score={year_degree_zodiac_sign_score}")
        else:
            logger.warning(f"行 {row.name}: 無法映射 year_degree_zodiac_sign_element='{year_degree_zodiac_sign_element}' 到天體。")
            錯誤計數['未知元素關係'] += 1
    else:
        logger.warning(f"行 {row.name}: year_degree_zodiac_sign_element 缺失。")
        錯誤計數['缺失元素關係'] += 1

    # 計算 total_score
    total_score = sum(分數.values()) + sum(額外分數.values())

    return 分數, 額外分數, 吉星, 凶星, element_state_map, total_score

def 處理元素欄位(row, 分數, 額外分數, element_state_map, 月份關係, 元素到天體, 天體因子, 天體乘數):
    for suffix, score_key in [('_zodiac_sign_element', 'zodiac_sign_score'), ('_mansion_element', 'mansion_score')]:
        elements = [col for col in row.index if col.endswith(suffix)]
        for col in elements:
            element = row[col]
            if pd.notnull(element):
                element = element.lower()
                celestial_match = re.match(r'^(.*?)' + re.escape(suffix) + r'$', col, re.IGNORECASE)
                if celestial_match:
                    celestial = celestial_match.group(1).lower()
                else:
                    celestial = 'unknown'

                if element in 五行元素:
                    element_to_use = element
                elif element in [v.lower() for v in 天體元素.values() if v is not None]:
                    element_to_use = element
                else:
                    element_to_use = 'unknown'

                if element_to_use != 'unknown':
                    element1 = 天體元素.get(celestial, None)
                    if element1 is None or element1 == 'unknown':
                        element1_state = element_state_map.get(element_to_use, '休')
                    else:
                        element1_state = element_state_map.get(element1.lower(), '休')

                    relation_sign = 獲取關係符號(element1, element_to_use, 月份關係, 元素到天體)
                    element2_state = element_state_map.get(element_to_use, '休')
                    element2_score = 狀態基準分數.get(element2_state, 3)
                    factor = 天體因子.get(celestial, 1)
                    score = relation_sign * element2_score * factor

                    if score > 0:
                        吉星.add(celestial)
                    elif score < 0:
                        凶星.add(celestial)

                    try:
                        elong1 = float(row.get(f'{celestial}_elong', 0)) % 360
                        elong2 = float(row.get(f'{celestial}_elong', 0)) % 360
                        distance = min(abs(elong1 - elong2), 360 - abs(elong1 - elong2))
                        increment = 1.1 + ((30 - (distance % 3)) // 3) * 0.05
                        increment = min(max(increment, 1.1), 1.5)
                        score *= increment
                    except ValueError:
                        logger.warning(f"行 {row.name}: 計算距離時出錯。")

                    # 檢查是否在岐度，使用岐度乘數
                    try:
                        # 獲取天體的岐度乘數
                        zodiac_degree = float(row.get(f'{celestial}_zodiac_degree', 0))
                        mansion_degree = float(row.get(f'{celestial}_mansion_degree', 0))
                        mansion_position = row.get(f'{celestial}_mansion_positions', '')
                        central_degree = None
                        if pd.notnull(mansion_position) and mansion_position in mansion_central_degrees:
                            central_degree = mansion_central_degrees[mansion_position]
                        
                        # 計算岐度乘數
                        qidu_multiplier = 計算岐度乘數(celestial, zodiac_degree, mansion_degree, central_degree)
                        score *= qidu_multiplier
                        
                        if qidu_multiplier > 1.0:
                            logger.debug(f"行 {row.name}: {celestial} 在岐度，乘數 {qidu_multiplier}")
                    except Exception as e:
                        logger.warning(f"行 {row.name}: 計算岐度乘數時出錯: {e}")

                    speed_type = row.get(f'{celestial}_speed_type', 'N/A')
                    if speed_type == 'fast':
                        if celestial in 凶星:
                            multiplier = 1.1
                            score = -score * multiplier
                    elif speed_type == 'invisible':
                        if celestial in 吉星:
                            score *= 0.5

                    if element_to_use in 分數:
                        分數[element_to_use] += score
                    else:
                        logger.warning(f"行 {row.name}: 未知的元素 '{element_to_use}'，無法加分。")

                    if suffix == '_mansion_element':
                        額外分數['mansion_score'] += score
                    elif suffix == '_zodiac_sign_element':
                        額外分數['zodiac_sign_score'] += score


def 處理特殊點(row, 分數, 額外分數, five_energy_dict, 天體元素, 元素到天體, element_state_map, 天體乘數, 月份關係):
    for point in ['asc', 'mc', 'part_of_fortune', 'life_degree', 'year_degree']:
        mansion_col = f'{point}_mansion_element'
        zodiac_col = f'{point}_zodiac_sign_element'
        mansion_element = row.get(mansion_col, None)
        zodiac_element = row.get(zodiac_col, None)
        element = None
        celestial = point.lower()

        if pd.notnull(mansion_element):
            element = mansion_element.lower()
            suffix = '_mansion_element'
        elif pd.notnull(zodiac_element):
            element = zodiac_element.lower()
            suffix = '_zodiac_sign_element'

        if element:
            if element in 五行元素:
                element_to_use = element
            elif element in [v.lower() for v in 天體元素.values() if v is not None]:
                element_to_use = element
            else:
                element_to_use = 'unknown'

            if element_to_use != 'unknown':
                element1 = 天體元素.get(celestial, None)
                if element1 is None or element1 == 'unknown':
                    element1_state = element_state_map.get(element_to_use, '休')
                else:
                    element1_state = element_state_map.get(element1.lower(), '休')

                relation_sign = 獲取關係符號(element1, element_to_use, 月份關係, 元素到天體)
                element2_state = element_state_map.get(element_to_use, '休')
                element2_score = 狀態基準分數.get(element2_state, 1)
                factor = 天體因子.get(celestial, 1)
                score = relation_sign * element2_score * factor

                if score > 0:
                    吉星.add(celestial)
                elif score < 0:
                    凶星.add(celestial)

                try:
                    elong1 = float(row.get(f'{celestial}_elong', 0)) % 360
                    elong2 = float(row.get(f'{celestial}_elong', 0)) % 360
                    distance = min(abs(elong1 - elong2), 360 - abs(elong1 - elong2))
                    increment = 1.1 + ((30 - (distance % 3)) // 3) * 0.05
                    increment = min(max(increment, 1.1), 1.5)
                    score *= increment
                except ValueError:
                    logger.warning(f"行 {row.name}: 計算距離時出錯。")

                # 檢查是否在岐度，使用岐度乘數
                try:
                    # 獲取天體的岐度乘數
                    zodiac_degree = float(row.get(f'{celestial}_zodiac_degree', 0))
                    mansion_degree = float(row.get(f'{celestial}_mansion_degree', 0))
                    mansion_position = row.get(f'{celestial}_mansion_positions', '')
                    central_degree = None
                    if pd.notnull(mansion_position) and mansion_position in mansion_central_degrees:
                        central_degree = mansion_central_degrees[mansion_position]
                    
                    # 計算岐度乘數
                    qidu_multiplier = 計算岐度乘數(celestial, zodiac_degree, mansion_degree, central_degree)
                    score *= qidu_multiplier
                    
                    if qidu_multiplier > 1.0:
                        logger.debug(f"行 {row.name}: {celestial} 在岐度，乘數 {qidu_multiplier}")
                except Exception as e:
                    logger.warning(f"行 {row.name}: 計算岐度乘數時出錯: {e}")

                speed_type = row.get(f'{celestial}_speed_type', 'N/A')
                if speed_type == 'fast':
                    if celestial in 凶星:
                        multiplier = 1.1
                        score = -score * multiplier
                elif speed_type == 'invisible':
                    if celestial in 吉星:
                        score *= 0.5

                if element_to_use in 分數:
                    分數[element_to_use] += score
                else:
                    logger.warning(f"行 {row.name}: 未知的元素 '{element_to_use}'，無法加分。")

                if suffix == '_mansion_element':
                    額外分數['mansion_score'] += score
                elif suffix == '_zodiac_sign_element':
                    額外分數['zodiac_sign_score'] += score

def optimize_state_scores(alldata, five_energy_dict, 天體元素, 元素到天體, 天體因子, 天體乘數, num_runs=1000, run_interval=2):
    initial_guess = [5, 4, 1, -1, -2, 1, 1]
    bounds = [
        (3, 6),    # 旺
        (2, 3),    # 相
        (-1, 1),   # 休
        (-2, -1),  # 囚
        (-4, -2),  # 死
        (0.5, 2.5),  # 太陽因子，每0.1測試
        (0.5, 2.5)   # 月亮因子，每0.1測試
    ]

    # 定義約束條件：旺 > 相 和 囚 > 死
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # 旺 - 相 >= 0
        {'type': 'ineq', 'fun': lambda x: x[3] - x[4]},  # 囚 - 死 >= 0
    ]

    best_result = None
    best_error = float('inf')
    output_counter = 1  # 用於生成順序文件名

    last_valid_result = None  # 新增：保存最後一個有效的優化結果

    def objective(params):
        state_scores = params[:5]
        sun_factor = params[5]
        moon_factor = params[6]

        try:
            # 更新狀態基準分數和天體因子
            設定狀態基準分數(
                旺=state_scores[0],
                相=state_scores[1],
                休=state_scores[2],
                囚=state_scores[3],
                死=state_scores[4]
            )
            天體因子['sun'] = sun_factor
            天體因子['moon'] = moon_factor

            total_error = 0
            # 遍歷數據並計算總分誤差
            for index, row in alldata.iterrows():
                分數, 額外分數, _, _, element_state_map, total_score = 計算分數(
                    row,
                    five_energy_dict,
                    天體元素,
                    元素到天體,
                    天體乘數,
                    sun_factor,
                    moon_factor
                )
                # 假設目標是某種理想分數，這裡需要根據具體需求設置
                ideal_score = 10  # 示例理想分數，根據實際需求調整
                current_score = sum(分數.values()) + sum(額外分數.values())
                error = (current_score - ideal_score) ** 2
                total_error += error

            logger.debug(f"Total error in objective function: {total_error}")
            return total_error
        except Exception as e:
            logger.error(f"目標函數執行時出錯: {e}", exc_info=True)
            return np.inf  # 返回一個大的錯誤值

    # 初始化一個列表以跟蹤最佳結果
    best_results_list = []

    with tqdm(total=num_runs, desc="優化分數", unit='run') as pbar:
        for run in range(1, num_runs + 1):
            # 隨機生成一個初始猜測，避免所有運行從同一點開始
            random_initial_guess = [
                round(np.random.uniform(bounds[0][0], bounds[0][1]), 1),
                round(np.random.uniform(bounds[1][0], bounds[1][1]), 1),
                round(np.random.uniform(bounds[2][0], bounds[2][1]), 1),
                round(np.random.uniform(bounds[3][0], bounds[3][1]), 1),
                round(np.random.uniform(bounds[4][0], bounds[4][1]), 1),
                round(np.random.uniform(bounds[5][0], bounds[5][1]), 1),
                round(np.random.uniform(bounds[6][0], bounds[6][1]), 1)
            ]

            try:
                result = minimize(
                    objective,
                    random_initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-6,
                        'disp': False
                    }
                )
                logger.debug(f"Run {run}: Optimization success: {result.success}, Function value: {result.fun}")
                if result.success:
                    last_valid_result = result  # 保存最後一個成功的優化結果
            except Exception as e:
                logger.error(f"第 {run} 次優化過程中出錯: {e}", exc_info=True)
                pbar.update(1)
                continue

            if result.success and result.fun < best_error:
                best_error = result.fun
                best_result = result.x
                logger.info(f"找到新的最佳結果: {best_result}, 誤差: {best_error}")

            # 每 run_interval 次運行保存一次結果
            if run % run_interval == 0 or run == num_runs:
                if last_valid_result is not None:
                    # 準備狀態基準分數和天體因子
                    state_scores = {
                        '旺': last_valid_result.x[0],
                        '相': last_valid_result.x[1],
                        '休': last_valid_result.x[2],
                        '囚': last_valid_result.x[3],
                        '死': last_valid_result.x[4]
                    }
                    celestial_factors = {
                        'sun': last_valid_result.x[5],
                        'moon': last_valid_result.x[6]
                    }

                    # 使用最佳參數計算每行的分數
                    設定狀態基準分數(**state_scores)
                    天體因子.update(celestial_factors)

                    # 計算 results
                    results = []
                    types = alldata['type'].tolist() if 'type' in alldata.columns else []
                    prices = alldata['price'].tolist() if 'price' in alldata.columns else []  # 確保 'price' 字段存在

                    for index, row in alldata.iterrows():
                        分數, 額外分數, 吉星, 凶星, element_state_map, total_score = 計算分數(
                            row,
                            five_energy_dict,
                            天體元素,
                            元素到天體,
                            天體乘數,
                            celestial_factors['sun'],
                            celestial_factors['moon']
                        )

                        # 修正行星列表的計算
                        try:
                            lunar_month = int(row['農曆月份'])
                        except (ValueError, TypeError):
                            lunar_month = None
                        if lunar_month and lunar_month in five_energy_dict:
                            行星列表 = set(天體元素.keys()).intersection(
                                celestial for element_relations in five_energy_dict.get(lunar_month, {}).values() for celestial in element_relations.keys()
                            )
                            特殊分數 = 計算特殊組合分數(行星列表, lunar_month, row, five_energy_dict, element_state_map)
                            額外分數['aspects_score'] = 額外分數.get('aspects_score', 0) + 特殊分數
                        else:
                            logger.warning(f"行 {row.name}: 農曆月份 {lunar_month} 不存在於五行能量字典中。")

                        # 計算 total_score
                        total_score = sum(分數.values()) + sum(額外分數.values())

                        # 根據 total_score 和 price type 進行匹配檢查
                        match_result = 'unmatch'
                        if index > 0 and index < len(prices):
                            previous_score = results[index - 1]['total_score'] if index - 1 < len(results) else 0
                            previous_price = prices[index - 1] if index - 1 < len(prices) else 0
                            current_price = prices[index]
                            current_type = types[index] if index < len(types) else 'N/A'

                            if pd.notnull(current_price) and pd.notnull(previous_price):
                                if current_price > previous_price and total_score > previous_score:
                                    match_result = 'match'
                                elif current_price < previous_price and total_score < previous_score:
                                    match_result = 'match'

                        results.append({
                            'datetime(utc)': row['datetime(utc)'] if 'datetime(utc)' in row and pd.notnull(row['datetime(utc)']) else 'N/A',
                            'sun_element_score': 分數.get('sun_element', 0),
                            'moon_element_score': 分數.get('moon_element', 0),
                            'water_score': 分數.get('water', 0),
                            'metal_score': 分數.get('metal', 0),
                            'fire_score': 分數.get('fire', 0),
                            'wood_score': 分數.get('wood', 0),
                            'earth_score': 分數.get('earth', 0),
                            'aspects_score': 額外分數.get('aspects_score', 0),
                            'conjunction_score': 額外分數.get('conjunction_score', 0),
                            'mansion_score': 額外分數.get('mansion_score', 0),
                            'zodiac_sign_score': 額外分數.get('zodiac_sign_score', 0),
                            'yunu_score': 分數.get('yunu_score', 0),
                            'year_degree_mansion_score': 分數.get('year_degree_mansion_score', 0),
                            'year_degree_zodiac_sign_score': 分數.get('year_degree_zodiac_sign_score', 0),
                            'life_degree_score': 分數.get('life_degree', 0),
                            'total_score': total_score,
                            'price': row['price'] if 'price' in row and pd.notnull(row['price']) else 'N/A',
                            'type': row['type'] if 'type' in row and pd.notnull(row['type']) else 'N/A',
                            'match': match_result
                        })

                    # 計算準確率
                    total_matches = sum(1 for result in results if result['match'] == 'match')
                    accuracy = total_matches / len(results) * 100 if len(results) > 0 else 0

                    # 將結果轉換為 DataFrame
                    results_df = pd.DataFrame(results)
                    state_scores_df = pd.DataFrame([state_scores])
                    celestial_factors_df = pd.DataFrame([celestial_factors])
                    accuracy_df = pd.DataFrame([{'accuracy (%)': accuracy}])

                    # 確保輸出目錄存在
                    output_dir = '/Users/jacky/Desktop/testfinaldata/'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # 保存結果到 Excel 文件
                    output_filename = os.path.join(output_dir, f'UltimateComprehensiveScore_37_{output_counter}.xlsx')
                    output_counter += 1

                    try:
                        with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
                            results_df.to_excel(writer, sheet_name='Scores', index=False)
                            state_scores_df.to_excel(writer, sheet_name='State_Scores', index=False)
                            celestial_factors_df.to_excel(writer, sheet_name='Celestial_Factors', index=False)
                            accuracy_df.to_excel(writer, sheet_name='Accuracy', index=False)
                        logger.info(f"已保存最佳結果至 {output_filename}")
                        best_results_list.append({'filename': output_filename})
                    except Exception as e:
                        logger.error(f"保存最佳結果至 {output_filename} 時出錯: {e}", exc_info=True)
                else:
                    logger.warning("未找到有效的優化結果，無法保存文件。")

            pbar.update(1)

    if best_result is not None:
        logger.info(f"優化完成。最佳狀態基準分數: 旺={best_result[0]:.2f}, 相={best_result[1]:.2f}, 休={best_result[2]:.2f}, 囚={best_result[3]:.2f}, 死={best_result[4]:.2f}, 太陽因子={best_result[5]:.2f}, 月亮因子={best_result[6]:.2f}，總誤差={best_error:.4f}")
        return {
            'state_scores': {
                '旺': best_result[0],
                '相': best_result[1],
                '休': best_result[2],
                '囚': best_result[3],
                '死': best_result[4]
            },
            'celestial_factors': {
                'sun': best_result[5],
                'moon': best_result[6]
            }
        }
    else:
        logger.warning("所有優化運行均未成功。使用預設分數。")
        return {
            'state_scores': {
                '旺': 5,
                '相': 4,
                '休': 1,
                '囚': -1,
                '死': -2
            },
            'celestial_factors': {
                'sun': 1,
                'moon': 1
            }
        }

def main():
    celestial_bodies = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn',
                        'lilith', 'selena', 'moon_north_node', 'moon_south_node',
                        'asc', 'mc', 'part_of_fortune', 'life_degree', 'year_degree']

    # 更新文件路徑
    alldata_o1_path = '/Users/jacky/Desktop/alldata_29.csv'
    five_energy_path = '/Users/jacky/Desktop/fiveenergycalculation_small3.csv'
    invalid_output_path = '/Users/jacky/Desktop/newdata/Invalid_Lunar_Dates.csv'

    # 讀取 alldata_29.csv
    try:
        alldata = pd.read_csv(alldata_o1_path, low_memory=False)
        logger.info(f"成功讀取 '{alldata_o1_path}'。")
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        logger.error(f"CSV 文件為空: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"讀取 CSV 文件時出錯: {e}")
        sys.exit(1)

    # 檢查農曆月份字段
    lunar_month_col = '農曆月份'
    if lunar_month_col not in alldata.columns:
        logger.error(f"錯誤: 農曆月份的字段名稱 '{lunar_month_col}' 不存在。")
        sys.exit(1)
    logger.info(f"識別到的農曆月份字段名稱: '{lunar_month_col}'")

    # 轉換 'datetime(utc)' 字段
    if 'datetime(utc)' in alldata.columns:
        alldata['datetime(utc)'] = pd.to_datetime(alldata['datetime(utc)'], errors='coerce', utc=True).dt.tz_convert(None)
    else:
        logger.error("錯誤: 'datetime(utc)' 字段不存在於數據中。")
        sys.exit(1)

    # 讀取五行能量計算表
    try:
        five_energy_chart = pd.read_csv(five_energy_path, encoding='utf-8-sig')
        logger.info("成功讀取五行能量計算表。")
    except Exception as e:
        logger.error(f"讀取五行能量計算表時出錯: {e}")
        sys.exit(1)

    # 創建元素到天體的反向映射
    元素到天體 = 創建元素到天體映射(天體元素)

    logger.info("\n元素到天體的映射:")
    for element, celestial_list in 元素到天體.items():
        logger.info(f"{element} -> {celestial_list}")

    # 建立五行能量字典
    five_energy_dict = 建立五行能量字典(five_energy_chart, 元素到天體)

    # 設定初始狀態基準分數和天體因子
    設定狀態基準分數(旺=5, 相=4, 休=1, 囚=-1, 死=-2)
    天體因子 = {
        'sun': 1,
        'moon': 1,
        'mercury': 1,
        'venus': 1,
        'mars': 1,
        'jupiter': 1.5,
        'saturn': 1.5
    }

    # 設定天體乘數
    天體乘數 = {
        'saturn': 1.5,
        'jupiter': 2.0
    }

    # 進行狀態基準分數和天體因子的優化
    optimized_values = optimize_state_scores(
        alldata,  # 使用完整數據進行優化
        five_energy_dict,
        天體元素,
        元素到天體,
        天體因子,
        天體乘數,
        num_runs=1000,
        run_interval=2
    )
    logger.info(f"最佳結果: {optimized_values}")

if __name__ == "__main__":
    main()