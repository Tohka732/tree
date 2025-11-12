from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib

def cal_shannon_ent(dataset):
    """
    计算熵
    """
    num_entries = len(dataset)
    labels_counts = {}
    
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in labels_counts.keys():
            labels_counts[current_label] = 0
        labels_counts[current_label] += 1

    shannon_ent = 0.0
    for key in labels_counts:
        prob = float(labels_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def create_dataSet():
    """
    熵接近 1，说明"yes"和"no"两个类别的比例比较接近，数据集的不确定性较高。
    熵接近 0，类别越集中，数据集越"纯"或"确定性越强"
    """
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels

def split_dataset(dataset, axis, value):
    """
    按照指定特征划分数据集
    """
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset

def choose_best_feature_split(dataset):
    """
    选择信息增益最大的特征
    """
    num_features = len(dataset[0]) - 1
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)
        new_entropy = 0.0
        
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * cal_shannon_ent(sub_dataset)
            
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    
    return best_feature if best_feature != -1 else 0

def majority_cnt(class_list):
    """
    返回出现次数最多的类别
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), 
                               key=operator.itemgetter(1), 
                               reverse=True)
    return sorted_class_count[0][0]

def creat_tree(dataset, labels):
    """
    创建决策树
    """
    class_list = [example[-1] for example in dataset]
    
    # 所有样本同类
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    
    # 没有可用特征
    if len(dataset[0]) == 1 or not labels:
        return majority_cnt(class_list)
    
    best_feat = choose_best_feature_split(dataset)
    best_feat = min(best_feat, len(labels) - 1)
    best_feat = max(best_feat, 0)
    
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    
    # 保存当前特征标签后删除
    del(labels[best_feat])
    
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    
    for value in unique_vals:
        sub_labels = labels[:]  # 拷贝标签列表
        sub_dataset = split_dataset(dataset, best_feat, value)
        my_tree[best_feat_label][value] = creat_tree(sub_dataset, sub_labels)
    
    return my_tree

# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 节点样式定义
decision_node = dict(boxstyle="sawtooth", fc='0.8')
leaf_node = dict(boxstyle="round4", fc='0.8')
arrow_args = dict(arrowstyle="<-")

def plot_node(ax, node_txt, center_pt, parent_pt, node_type):
    ax.annotate(node_txt,
                xy=parent_pt, xycoords='axes fraction',
                xytext=center_pt, textcoords='axes fraction',
                va="center", ha="center",
                bbox=node_type, arrowprops=arrow_args,
                fontsize=11, color='black')

def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def plot_mid_text(ax, center_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] + center_pt[0]) / 2.0
    y_mid = (parent_pt[1] + center_pt[1]) / 2.0
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", fontsize=10)

def plot_tree(ax, my_tree, parent_pt, node_txt, total_w, total_d, x_off_y):
    first_str = next(iter(my_tree))
    child_dict = my_tree[first_str]

    num_leafs = get_num_leafs(my_tree)
    center_pt = (x_off_y['x_off'] + (1.0 + num_leafs) / (2.0 * total_w), 
                 x_off_y['y_off'])

    if node_txt:
        plot_mid_text(ax, center_pt, parent_pt, node_txt)

    plot_node(ax, first_str, center_pt, parent_pt, decision_node)

    x_off_y['y_off'] -= 1.0 / total_d
    for key, child in child_dict.items():
        if isinstance(child, dict):
            plot_tree(ax, child, center_pt, str(key), total_w, total_d, x_off_y)
        else:
            x_off_y['x_off'] += 1.0 / total_w
            leaf_pt = (x_off_y['x_off'], x_off_y['y_off'])
            plot_node(ax, str(child), leaf_pt, center_pt, leaf_node)
            plot_mid_text(ax, leaf_pt, center_pt, str(key))
    x_off_y['y_off'] += 1.0 / total_d

def create_plot(my_tree):
    """
    绘制决策树
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_axis_off()

    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off_y = {'x_off': -0.5 / total_w, 'y_off': 1.0}

    plot_tree(ax, my_tree, (0.5, 1.0), '', total_w, total_d, x_off_y)
    plt.tight_layout()
    plt.show()

def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树进行分类预测
    """
    first_str = next(iter(input_tree))
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if isinstance(second_dict[key], dict):
                return classify(second_dict[key], feat_labels, test_vec)
            else:
                return second_dict[key]
    return None

def load_lenses(path):
    """
    加载隐形眼镜数据集
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                data.append(parts)
    return data

# 主程序
if __name__ == "__main__":
    # 测试数据集
    dataset, labels = create_dataSet()
    print("测试数据集熵:", cal_shannon_ent(dataset))
    
    # 加载隐形眼镜数据
    lenses_data = load_lenses('lenses.txt')
    lenses_labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
    
    # 构建决策树
    lenses_tree = creat_tree(lenses_data, lenses_labels[:])
    print("决策树构建完成")
    
    # 绘制决策树
    create_plot(lenses_tree)
    
    # 计算准确率
    correct = 0
    feature_labels = ['age', 'prescription', 'astigmatic', 'tear_rate']
    
    for row in lenses_data:
        x, y = row[:-1], row[-1]
        yhat = classify(lenses_tree, feature_labels, x)
        if yhat == y:
            correct += 1
    
    accuracy = correct / len(lenses_data)
    print(f"训练集准确率: {correct}/{len(lenses_data)} = {accuracy:.2%}")
