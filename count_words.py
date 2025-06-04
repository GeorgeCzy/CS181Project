# 统计文档中出现的特定单词数量

def count_words(file_path):
    words_to_count = ['Red', 'draw', 'Blue']
    counts = {word: 0 for word in words_to_count}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            for word in words_to_count:
                counts[word] += line.count(word)

    return counts

# 使用示例
file_path = 'c:\\ChengZY\\college5\\CS181\\CS181Project\\random_minimax_log.txt'
word_counts = count_words(file_path)
print(word_counts)