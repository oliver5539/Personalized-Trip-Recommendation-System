import pandas as pd
import numpy as np
import random
import datetime
import folium as fo
import math
import plotly.graph_objs as go
import time

startTi = time.time()
loc_final = '/Users/oliver/PycharmProjects/Genetic/GAsDataFinal/'

df_bicycling_od = pd.read_csv(loc_final + 'attraction_od_bicycling_v2.csv', header=0, index_col=0, encoding="utf-8")
df_driving_od = pd.read_csv(loc_final + 'attraction_od_driving.csv', header=0, index_col=0, encoding="utf-8")
df_walking_od = pd.read_csv(loc_final + 'attraction_od_walking.csv', header=0, index_col=0, encoding="utf-8")
df_att = pd.read_csv(loc_final + 'all_attraction_v2.csv', encoding="utf-8")
df_time = pd.read_csv(loc_final + 'attraction_time_v2.csv', encoding="utf-8")
df_time_in = pd.read_csv(loc_final + 'attraction_time_index_v2.csv', header=0, index_col=0, encoding="utf-8")
df_hot_time = pd.read_csv(loc_final + 'attraction_hot_time.csv', encoding="utf-8")
df_u_a = pd.read_csv(loc_final + 'ubike_attraction.csv', encoding="utf-8")
df_a_u = pd.read_csv(loc_final + 'attraction_ubike_v2.csv', encoding="utf-8")
df_u = pd.read_csv(loc_final + 'Hot_sta_v2.csv', header=0, index_col=0, encoding="utf-8")
print('df_bicycling_od', '\n', df_bicycling_od)
print('df_driving_od', '\n', df_driving_od)
print('df_walking_od', '\n', df_walking_od)
print('df_att', '\n', df_att)
print('df_time', '\n', df_time)
print('df_time_in', '\n', df_time_in)
print('df_hot_time', '\n', df_hot_time)
print('df_u_a', '\n', df_u_a)
print('df_a_u', '\n', df_a_u)

print(df_u_a.columns)


# 格式化輸出，objectname為輸出物件名稱，object為輸出物件，zz為是否顯示(0=F，1=T)
def print_(objectname, object):
    if (zz):
        print(objectname, "\n", object, "\n", "-" * 80)


zz = 0


# 畫圖
def draw(df, name):
    fig = go.Figure()
    xx = df.index
    yy = df['fit']
    fig.add_trace(go.Scatter(x=xx, y=yy))
    fig.update_layout(xaxis=dict(tickmode='linear',
                                 tick0=0,
                                 dtick=1))
    fig.update_layout(title_text=name)
    fig.show()


# 決定景點資料
def inputdata(df, dfa, dfu, slist):
    # dfa = dfa.drop([3, 7, 28, 36, 68, 73, 74, 75, 76])
    filter_data_e = pd.DataFrame()
    for i in slist:
        bool = dfa['地址'].str.contains(i)
        filter_data = dfa[bool]
        filter_data_e = filter_data_e.append(filter_data).reset_index(drop=True)
    l = filter_data_e['名稱'].tolist()
    temp = df[df['attraction'].isin(l)].reset_index(drop=True)
    temp.set_index('attraction', inplace=True)
    # print(temp)
    temp = temp.T.reset_index(drop=True)
    # print(temp.columns)
    ls = []
    for j in (l):
        # print(temp[j])
        ll = temp[j].tolist()
        nll = [x for x in ll if np.isnan(x) == False]
        ls.extend(nll)
    ls = set(ls)
    att = []
    for k in (ls):
        k = int(k)
        sk = str(k)
        a = dfu[sk].tolist()
        a = [str(x) for x in a]
        na = [y for y in a if y != 'nan']
        att.extend(na)
    att = set(att)
    ind = dfa[dfa['名稱'].isin(att)].reset_index(drop=True)
    return ind


'''第一階段'''


# 定義（種族大小、基因長度）、創立初始基因
def geneEncoding(pop_size, df, count):
    # 建立空的種族陣列
    new_population = np.empty((pop_size[0], pop_size[1]))
    l = len(df)
    # 插入隨機產生的基因序列
    for i in range(pop_size[0]):
        a = random.randint(1, count)
        # 產生隨機變數的list
        ran_list = []
        for x in range(a):
            ran_list.append(1)
        for y in range(l - a):
            ran_list.append(0)
        list = random.sample(ran_list, len(ran_list))
        # 放進去種族陣列
        new_population[i] = list
    return new_population


# 解碼
def decodechrom(pop, pop_size, df):
    list = []
    for i in range(pop_size[0]):
        df['gene'] = pop[i]
        rule = df['gene'] == 1
        temp = df[rule]
        l = temp['code'].tolist()
        list.append(l)
    return list


# （旅遊天數,初始陣列），計算每條基因的適應值
# (初始陣列,種族基因大小,inputData,交通od,景點逗留時間,旅遊時數(日),旅遊天數,寬放)
def get_fitness(pop, pop_size, df, df_od, dft, H, D, P):
    # 建立空的適應值list
    fitness = []
    decode = decodechrom(pop, pop_size, df)
    print_('decode', decode)
    for i in range(len(decode)):
        temp2 = df[df['code'].isin(decode[i])].reset_index(drop=True)
        total_distance = 0
        for j in range(len(temp2) - 1):
            td = 0
            a1 = temp2.at[j, '名稱']
            for k in range(len(temp2)):
                if k != j:
                    a2 = temp2.at[k, '名稱']
                    td = td + df_od.at[a1, a2]
                else:
                    td = td + 0
            tda = td / (len(temp2) - 1)
            total_distance += tda
        print_('total_distance', total_distance)
        tt = 0
        for k in range(len(temp2)):
            t = dft.at[k, 'Time']
            tt += t
        print_('tt', tt)
        fit = 1 - ((tt * P + (total_distance / 60)) / (D * H * 60))
        fit = abs(fit)
        fitt = math.exp(fit)
        fit_d = 1 / fitt
        fitness.append(fit_d)
    return fitness


# 選擇，用輪盤法
def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))

    # 輪盤法
    fitness = np.array(fitness)
    # 機率抽樣
    roulette = np.random.choice(np.arange(len(pop)), size=num_parents, replace=True, p=fitness / fitness.sum())
    parents[:] = pop[roulette]
    return parents


# 交配
def crossover(parents, offspring_size, cp):
    # 建立後代陣列
    offspring = np.empty(offspring_size)
    # 決定交配點，這邊採用單點交配
    crossover_point = np.uint8(offspring_size[1] / 2)
    for k in range(offspring_size[0]):
        # 要配對的第一個父母的索引
        parent1_idx = k % parents.shape[0]
        print_('parent1_idx', parent1_idx)
        # 要配對的第二個父母的索引
        parent2_idx = (k + 1) % parents.shape[0]
        print_('parent2_idx', parent2_idx)
        c = random.random()
        if c < cp:
        # 新的後代將有其前一半的基因來自第一個父母
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # 新的後代將有其後一半的基因來自第二個父母
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        else:
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent1_idx, crossover_point:]
    print_('offspring', offspring)
    return offspring


# 突變，突變會隨機改變每個後代中的單個基因
def mutation(offspring_crossover, size, z):
    for idx in range(offspring_crossover.shape[0]):
        a = random.random()
        if (a < z):
            idy = np.random.randint(0, size, 1)
            offspring_crossover[idx, idy] = np.random.randint(0, 1)
    return offspring_crossover


'''第二階段'''


def inputdata2(dfa, d):
    df = dfa
    cd = d - 1
    for i in range(cd):
        l = ['換天', 'c', 'c', 'c', 'c', '0']
        temp = pd.DataFrame([l])
        temp.columns = ['名稱', '地址', 'lng', 'lat', 'tag', 'code']
        df = df.append(temp).reset_index(drop=True)
    print_('dfa2', df)
    return df


# 定義（種族大小、基因長度）、創立初始基因
def geneEncoding2(pop_size):
    # 建立空的種族陣列
    new_population = np.empty((pop_size[0], pop_size[1]))
    ran_list = []
    for x in range(pop_size[1]):
        ran_list.append(x + 1)
    # 插入隨機產生的基因序列
    for i in range(pop_size[0]):
        # 產生隨機變數的list
        list = random.sample(ran_list, len(ran_list))
        # 放進去種族陣列
        new_population[i] = list
    # print_('new_population', new_population)
    return new_population


# 解碼
def decodechrom2(pop, pop_size, df):
    list = []
    for i in range(pop_size[0]):
        df['gene'] = pop[i]
        temp = df.sort_values(by=['gene']).reset_index(drop=True)
        l = temp['code'].tolist()
        list.append(l)
    return list


# （旅遊天數,初始陣列），計算每條基因的適應值
# (初始陣列,種族基因大小,inputData,交通od,景點逗留時間,景點熱點,開始時間)
def get_fitness2(pop, pop_size, df, df_od, dft, start_o, df_u, onedayhour):
    loc_Hot = '/Users/oliver/PycharmProjects/Genetic/GAsDataFinal/HotTime/'
    # 建立空的適應值list

    fitness = []
    decode = decodechrom2(pop, pop_size, df)
    print_('decode', decode)
    temp2 = df

    # 景點間的交通評分
    att_list = df['名稱'].tolist()
    # att_list.remove('換天')
    # print(df_od)
    df_od_f = df_od.loc[att_list]
    df_od_f = df_od_f[att_list]
    df_od_f_d = 1 / df_od_f
    df_od_f_d = df_od_f_d.replace([np.inf], 0)
    ml = df_od_f_d.max()
    ml = ml.tolist()
    m = max(ml)
    df_od_f_d = df_od_f_d / m
    # =======

    for i in range(pop_size[0]):
        start = start_o
        print_('i', i)
        temp2['gene'] = df['gene'] = pop[i]
        temp3 = df.sort_values(by=['gene']).reset_index(drop=True)
        temp3['code'] = temp3['code'].astype('int64')
        transportation_grade = 0
        trg = 0
        distance = 0
        grage = 0
        cgrade = 0
        for k in range(len(temp3) - 1):
            aa = temp3.at[k, 'code']
            bb = temp3.at[k + 1, 'code']
            if (aa != 0 and bb != 0):
                # 計算交通評分
                a1 = temp3.at[k, '名稱']
                a2 = temp3.at[k + 1, '名稱']
                tg = df_od_f_d.at[a1, a2]
                trg += tg
                # ===
                h = start.hour
                d = start.weekday() + 1
                d = str(d)
                dis = df_od.at[a1, a2]
                dis = dis / 60
                distance = distance + dis
                sta = df_u.at[a1, 'sta']
                # print(sta)
                hotData = pd.read_csv(loc_Hot + str(sta) + '_HotTime.csv', encoding="utf-8")
                g = hotData.at[h, d]
                grage += g
                dis = dis.astype(np.float64)
                att = dft.at[a1, 'Time']
                att = att.astype(np.float64)
                # 時間累計
                start = start + datetime.timedelta(minutes=dis) + datetime.timedelta(minutes=att)
            elif (aa != 0 and bb == 0):
                transportation_grade += trg
                trg = 0
                # ===
                h = start.hour
                d = start.weekday() + 1
                d = str(d)
                last = temp3.at[k, '名稱']
                sta = df_u.at[last, 'sta']
                hotData = pd.read_csv(loc_Hot + str(sta) + '_HotTime.csv', encoding="utf-8")
                att = dft.at[last, 'Time']
                att = att.astype(np.float64)
                start = start + datetime.timedelta(minutes=att)
                # eh = start.hour
                # sh = start_o.hour
                p = start - start_o
                ph = p.seconds / 3600
                m = abs(onedayhour - ph)
                mm = 1 / math.exp(m)
                cgrade += mm
                g = hotData.at[h, d]
                grage += g
                # print(start)
                start = start_o
                # print(start)
                start = start + datetime.timedelta(days=1)
                # print(start)
            else:
                distance = 0
            # ==========
            if (k == (len(temp3) - 2)):
                h = start.hour
                d = start.weekday() + 1
                d = str(d)
                last = temp3.at[k + 1, '名稱']
                first = temp3.at[0, '名稱']
                pos = first.rfind('夜市')
                if (last == '換天'):
                    transportation_grade = 0
                    grage = 0
                elif (pos > 0):
                    transportation_grade = 0
                    grage = 0
                else:
                    sta = df_u.at[last, 'sta']
                    hotData = pd.read_csv(loc_Hot + str(sta) + '_HotTime.csv', encoding="utf-8")
                    g = hotData.at[h, d]
                    grage += g
        # print('='*20)
        ff = transportation_grade + grage + cgrade
        fitness.append(ff)
    return fitness


# 選擇，用輪盤法
def select_mating_pool2(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    # 輪盤法
    fitness = np.array(fitness)
    # 機率抽樣
    roulette = np.random.choice(np.arange(len(pop)), size=num_parents, replace=True, p=fitness / fitness.sum())
    # print('-'*80,'\n','roulette','\n',roulette,'\n','-'*80,'\n')
    parents[:] = pop[roulette]
    return parents


# 交配
def crossover2(parents, offspring_size, cp):
    # 建立後代陣列
    offspring = np.empty(offspring_size)
    # 決定交配點，這邊採用單點交配
    crossover_point = np.uint8(offspring_size[1] / 2)
    for k in range(offspring_size[0]):
        # 要配對的第一個父母的索引
        parent1_idx = k % parents.shape[0]
        print_('parent1_idx', parent1_idx)
        # 要配對的第二個父母的索引
        parent2_idx = (k + 1) % parents.shape[0]
        print_('parent2_idx', parent2_idx)
        c = random.random()
        if c < cp:
        # 新的後代將有其前一半的基因來自第一個父母
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # 新的後代將有其後一半的基因來自第二個父母
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        else:
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent1_idx, crossover_point:]
    print_('offspring', offspring)
    return offspring


# 突變，突變會隨機改變每個後代中的單個基因
def mutation2(offspring_crossover, size, z):
    for idx in range(offspring_crossover.shape[0]):
        a = random.random()
        if (a < z):
            idy = np.random.randint(0, size, 1)
            offspring_crossover[idx, idy] = np.random.randint(1, size)
    return offspring_crossover


# 淘汰
def delete2(offspring_crossover, size):
    count = 0
    de = []
    for idx in range(offspring_crossover.shape[0]):
        dou = []
        for z in range(size):
            # print(z+1)
            li = offspring_crossover[idx].tolist()
            li = list(map(int, li))
            cc = li.count(z + 1)
            if cc > 1:
                dou.append(z + 1)
        if len(dou) <= 1:
            count += 1
        if len(dou) > 1:
            de.append(idx)
    offspring_crossover = np.delete(offspring_crossover, de, axis=0)
    # print(offspring_crossover)
    return offspring_crossover


# 調整
def adjustment2(offspring_crossover, size):
    for idx in range(offspring_crossover.shape[0]):
        dou = []
        ze = []
        for z in range(size):
            # print(z+1)
            li = offspring_crossover[idx].tolist()
            li = list(map(int, li))
            cc = li.count(z + 1)
            if cc > 1:
                dou.append((z + 1))
            if cc == 0:
                ze.append((z + 1))
        cc = []
        for x in (dou):
            s = np.where(offspring_crossover[idx] == x)
            ss = s[0].tolist()
            cc.extend(ss)
        cc.sort()
        dou.extend(ze)
        ranl = random.sample(dou, len(dou))
        count = 0
        for idy in (cc):
            offspring_crossover[idx, idy] = ranl[count]
            count += 1
    return offspring_crossover


s = 10
bestData_end_1 = pd.DataFrame()
bestData_end_2 = pd.DataFrame()
for ss in range(s):
    print('====='*30, ss)
    '''第一階段'''
    d = 1
    p = 1
    h = 8
    z1 = 0.1
    c1 = 0.5
    slist = ['中山區', '士林區', '大同區']
    # dfs = pd.read_csv(loc_final + "篩選後.csv", encoding="utf-8")
    inputData = inputdata(df_a_u,df_att,df_u_a,slist)
    # inputData = dfs
    print('inputData', '\n', inputData)
    pop_size = (100, len(inputData))
    new_population = geneEncoding(pop_size, inputData, 4)
    # print(new_population[0])
    num_generations = 200
    num_parents_mating = 50

    max_fitvalue = 0.
    max_population = []
    best_fit = []

    for generation in range(num_generations):
        print_("Generation : ", generation)
        fitness = get_fitness(new_population, pop_size, inputData, df_bicycling_od, df_time, h, d, p)
        max_fitness = np.max(fitness)
        print_("Best result : ", max_fitness)
        if max_fitness > max_fitvalue:
            max_fitvalue = max_fitness
            best_match_idx = np.where(fitness == max_fitness)
            max_population = new_population[best_match_idx, :]
        best_fit.append(max_fitvalue)
        # 選擇父母
        parents = select_mating_pool(new_population, fitness, num_parents_mating)
        print_('parent', parents)
        # 交配
        offspring_crossover = crossover(parents, pop_size, c1)
        print_('offspring_crossover', offspring_crossover)
        # 突變
        offspring_mutation = mutation(offspring_crossover, pop_size[1], z1)
        # 更新種族
        new_population[:, :] = offspring_mutation

    print("Best solution : ", max_population)
    print("Best solution fitness : ", max_fitvalue)
    best = max_population[0][0]
    inputData['gene'] = max_population[0][0]
    r = inputData['gene'] == 1
    best_att = inputData[r].reset_index(drop=True)
    best_att = best_att.drop(['gene'], axis=1)
    print(best_att)
    bestData = pd.DataFrame(best_fit)
    # bestData['count'] = bestData.index+1
    bestData.columns = [ss]
    bestData_end_1 = pd.concat([bestData_end_1,bestData], axis=1)

    # bestData.to_csv(loc_final + 'bestData1.csv', encoding='utf-8', index=False)

    '''第二階段'''
    input_data = inputdata2(best_att, d)
    pop_size = (100, len(input_data))
    new_population = geneEncoding2(pop_size)
    num_generations = 200
    num_parents_mating = 50
    startime = '2019-11-20 10:00:00'
    startime = datetime.datetime.strptime(startime, '%Y-%m-%d %H:%M:%S')
    z2 = 0.1
    c2 = 0.5

    max_fitvalue = 0.
    max_population = []
    best_fit = []

    for generation in range(num_generations):
        print_("Generation : ", generation)
        fitness = get_fitness2(new_population, pop_size, input_data, df_bicycling_od, df_time_in, startime, df_u, h)
        max_fitness = np.max(fitness)
        print_("Best result : ", max_fitness)
        if max_fitness > max_fitvalue:
            max_fitvalue = max_fitness
            best_match_idx = np.where(fitness == max_fitness)
            max_population = new_population[best_match_idx, :]
        best_fit.append(max_fitvalue)
        # 選擇父母
        parents = select_mating_pool2(new_population, fitness, num_parents_mating)
        print_('parent', parents)
        # 交配
        offspring_crossover = crossover2(parents, pop_size, c2)
        print_('offspring_crossover', offspring_crossover)
        # 突變
        offspring_mutation = mutation2(offspring_crossover, pop_size[1], z2)
        # 調整
        offspring_adjustment = adjustment2(offspring_mutation, pop_size[1])
        # 更新種族
        new_population[:, :] = offspring_mutation

    print("Best solution : ", max_population)
    print("Best solution fitness : ", max_fitvalue)
    best = max_population[0][0]
    input_data['gene'] = max_population[0][0]
    input_data = input_data.sort_values(by=['gene']).reset_index(drop=True)
    print(input_data)
    besta = input_data['名稱'].tolist()
    print(besta)
    bestData2 = pd.DataFrame(best_fit)
    # # bestData2['count'] = bestData2.index+1
    bestData2.columns = [ss]
    bestData_end_2 = pd.concat([bestData_end_2, bestData2],axis=1)
    # bestData2.to_csv(loc_final + 'bestData22.csv', encoding='utf-8', index=False)

bestData_end_1['mean'] = bestData_end_1.mean(axis=1)
bestData_end_2['mean'] = bestData_end_2.mean(axis=1)
print(bestData_end_1)
print(bestData_end_2)
loc_se = '/Users/oliver/PycharmProjects/Genetic/GAsDataFinal/se/'
bestData_end_1.to_csv(loc_se + 'bestData_end_1_200.csv', encoding='utf-8', index=False)
# bestData_end_2.to_csv(loc_se + 'bestData_end_2.csv', encoding='utf-8', index=False)


# map
def map2(df, name):
    fmap = fo.Map(location=[25.090293, 121.50189],
                  zoom_start=11.3)
    for i in range(0, len(df)):
        if df.at[i, '名稱'] != '換天':
            print_('景點：', i)
            b = [df.at[i, '名稱']]
            # print(b)
            m1 = fo.Marker(location=[df.at[i, 'lat'], df.at[i, 'lng']],
                           color='red',
                           radius=50,
                           popup=b,
                           fill=True,
                           fill_opacity=1)
            fmap.add_child(child=m1)
        fmap.save(loc_final + name + '.html')


# map2(input_data, 'GAs_ft')

# draw(bestData, 'first')
# draw(bestData2, 'second')

endTi = time.time()
print('RunTime:', endTi - startTi)

# df_se = []