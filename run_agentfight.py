import subprocess

# 打开一个文件用于记录输出
with open('random_greedy_log.txt', 'w') as log_file:
    for i in range(50):
        # 执行命令并捕获输出
        print(i)
        process = subprocess.Popen(['python', 'AgentFight.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # 获取输出的最后一行
        last_line = stdout.decode('utf-8').strip().split('\n')[-1]

        # 写入最后一行到日志文件
        log_file.write(f'Run {i+1} Last Line:\n')
        log_file.write(last_line + '\n')
        log_file.write(stderr.decode('utf-8'))
        log_file.write('\n' + '='*40 + '\n')

print('All commands executed and logged.')