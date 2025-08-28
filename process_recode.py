
def find_best(file, get_num = False):

    with open(file, 'r') as f:
        a = f.readlines()

    b = a[0] + a[4] + '     ' + a[5]
    val_res = []
    val_epoch = []
    test_epoch = []
    test_res = []
    test_res_list = []

    n = 1 if 'checkpoints_reg' in file else 4
    for index, i in enumerate(a):
        if 'val' in i:
            val_epoch.append(i.split('_')[0])
            val_res.append(float(a[index + 2].split('  ')[n]))

        if 'test' in i:
            b += i.split('_')[0] + a[index + 2]
            test_res_list.append(a[index + 2])
            test_res.append(float(a[index + 2].split('  ')[n]))
            test_epoch.append(i.split('_')[0])

    val_best = val_epoch[val_res.index(max(val_res))]
    test_best = test_epoch[test_res.index(max(test_res))]
    test_best_acc = test_res_list[test_res.index(max(test_res))]
    print(b)
    print('val best:', val_best)
    print('test best:', test_best + test_best_acc)

    if get_num:
        return b + '\n' + 'val best:' + val_best + '\n' + 'test best:' + test_best + test_best_acc + '\n', int(val_best), int(test_best)
    else:
        return b + '\n' + 'val best:' + val_best + '\n' + 'test best:' + test_best + test_best_acc + '\n'


if __name__ == '__main__':
    # file = r'checkpoints_cls\2024-03-02_12-28\val_and_test_acc.txt'
    # file = r'F:\pycharm_project\PRMD-UNet-ablation-fusion-v3.0\checkpoints_classify\2024-03-23_12-11\val_and_test_acc.txt'
    # file = r'F:\pycharm_project\PRMD-UNet-ablation-fusion-v3.0\checkpoints_reg\2024-03-23_13-16\val_and_test_acc.txt'
    file = r'F:\pycharm_project\PRMD-train-rain-geo-v4.0\val_and_test_acc+(1).txt'
    find_best(file)

