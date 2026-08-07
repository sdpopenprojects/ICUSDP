import os
import pandas as pd


if __name__ == '__main__':

    # save_path = '../promise/'
    #
    # project_names = sorted(os.listdir('../promise66/'))
    # path = os.path.abspath('../promise66/')
    # pro_num = len(project_names)
    # c = 0
    # for i in range(pro_num):
    #     project_name = project_names[i]
    #     file = os.path.join(path, project_name)
    #     data = pd.read_csv(file)
    #     project_name = project_name[:-4]
    #
    #     # combine CK, NET, and PROC metrics
    #     # X = [data.iloc[:, 0:7], data.iloc[:, 7:31], data.iloc[:, 31:42]]
    #     if c == 0:
    #         df = data.iloc[:, :-2]
    #         c = c+1
    #     else:
    #         if c == 1:
    #             df = pd.concat([df, data.iloc[:, :-2]], axis=1)
    #         else:
    #             df = pd.concat([df, data], axis=1)
    #
    #         c = c+1
    #         if c == 3:
    #             c = 0
    #             df.to_csv(save_path + project_name[:-5] + '.csv', index=None)

    save_path = '../ck/'

    # file = os.path.abspath('../ck/MJ12A.txt')
    c = 0
    data = []
    n_modules = []
    projects = []
    with open("../ck/MJ12.txt", "r") as f:
        # data = f.read()
        metrics = f.readline()
        metrics = metrics.strip('\n')
        metrics = metrics.split(';')
        metrics = metrics[1:]

        for line in f.readlines():
            line = line.strip('\n')
            if line:
                line = line.split(';')
                project_name = '-'.join([line[0], line[1]])
                temp_data = line[3:]
                temp_data = [s.replace('-', '0') for s in temp_data]

                if c == 0:
                    temp_name = project_name
                    data.append(temp_data) # first
                    c = c+1
                else:
                    if temp_name == project_name:
                        data.append(temp_data)
                    else:
                        c = 0
                        n_modules.append(len(data))
                        projects.append(temp_name)

                        data = pd.DataFrame(data)
                        data.columns = metrics

                        id = ['nr', 'ndc', 'nml', 'ndpv', 'bugs']
                        temp = data[id]
                        df = data.drop(id, axis=1)
                        data = pd.concat([df, temp], axis=1)

                        data.to_csv(save_path + temp_name + '.csv', index=None)

                        data = []
                        data.append(temp_data)
            else: # the last project
                n_modules.append(len(data))
                projects.append(temp_name)

                data = pd.DataFrame(data)
                data.columns = metrics

                id = ['nr', 'ndc', 'nml', 'ndpv', 'bugs']
                temp = data[id]
                df = data.drop(id, axis=1)
                data = pd.concat([df, temp], axis=1)

                data.to_csv(save_path + temp_name + '.csv', index=None)
                print('the last !')

    n_modules = pd.DataFrame(n_modules)
    n_modules.index = projects
    n_modules.to_csv(save_path+'module_size.csv', header=None)
    print('done!')



