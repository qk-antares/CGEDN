import os
import json

def calculate_average_mn(directory_paths):
    total_ratio = 0
    total_files = 0

    # 遍历所有目录
    for directory_path in directory_paths:
        # 遍历指定目录下的所有文件
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                filepath = os.path.join(directory_path, filename)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    m = data['m']
                    n = data['n']
                    ratio = m / n
                    total_ratio += ratio
                    total_files += 1

    # 计算平均值
    if total_files > 0:
        average_ratio = total_ratio / total_files
        return average_ratio
    else:
        return None
    
def calculate_average_m(directory_paths):
    total_files = 0
    total_m = 0

    # 遍历所有目录
    for directory_path in directory_paths:
        # 遍历指定目录下的所有文件
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                filepath = os.path.join(directory_path, filename)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    m = data['n']
                    total_m += m
                    total_files += 1

    # 计算平均值
    if total_files > 0:
        average_m = total_m / total_files
        return average_m
    else:
        return None

def main():
    # directory1 = './AIDS_700/json/train'
    # directory2 = './AIDS_700/json/test/query'

    # directory1 = './Linux/json/train'
    # directory2 = './Linux/json/test/query'


    # directory_paths = [directory3, directory4]

    directory1 = './IMDB_small/json/train'
    directory2 = './IMDB_small/json/test/query'
    directory3 = './IMDB_large/json/train'
    directory4 = './IMDB_large/json/test/query'
    
    # directory1 = './AIDS_small/json/train'
    # directory2 = './AIDS_small/json/test/query'
    # directory3 = './AIDS_large/json/train'
    # directory4 = './AIDS_large/json/test/query'

    directory_paths = [directory1, directory2, directory3, directory4]

    # 计算合并后的 m/n 均值
    average_ratio = calculate_average_mn(directory_paths)

    # 打印结果
    if average_ratio is not None:
        print(f"Average m/n ratio for all JSON files: {average_ratio}")
    else:
        print("No JSON files found in the directories.")

if __name__ == "__main__":
    main()
