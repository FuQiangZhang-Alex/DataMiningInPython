

__all__ = ['file2list']


def file2list(file, headers=False, separator=','):
    data = []
    with open(file=file, mode='r') as data_file:
        if headers:
            print(headers)
        else:
            for line in data_file:
                data.append(list(line.rstrip('\n').split(separator)))
    return data
