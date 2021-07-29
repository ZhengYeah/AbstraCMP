def verified_under_delta(filename, delta):
    """

    Args:
        filename: (str) robust radius results file
        delta: (float) perturbation size

    Returns: (int) number of inputs that verified robust under delta

    """
    robust_cnt = 0
    with open(filename, 'r') as f:
        for i in range(100):
            line_data = f.readline().replace('\n', '').split(' ')
            radius = float(line_data[0])
            if radius >= delta:
                robust_cnt += 1
    print('Robust number:', robust_cnt)
    pass


if __name__ == '__main__':
    delta = 0.006
    while delta <= 0.04 + 0.00001:
        print('Now delta = %.3f' % delta)
        verified_under_delta('mnist_20x50_deeppoly_results.txt', delta)
        delta += 0.001
