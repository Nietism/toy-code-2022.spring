import json
INIT = 'Ġ'
def read_fun(data_path):
    with open(data_path,encoding='utf-8') as f:
        data_lines=json.load(f)
    f.close()
    return data_lines
def count_pre_e_fun(p_seq):
    Event_types = {"Movement:Transport": 1, "Personnel:Elect": 1, "Personnel:Start-Position": 1,
                   "Personnel:Nominate": 1, "Conflict:Attack": 1, "Personnel:End-Position": 1, "Life:Die": 1,
                   "Contact:Meet": 1, "Life:Marry": 1, "Contact:Phone-Write": 1, "Transaction:Transfer-Money": 1,
                   "Justice:Sue": 1, "Conflict:Demonstrate": 1, "Justice:Fine": 1, "Life:Injure": 1,
                   "Business:End-Org": 1, "Justice:Trial-Hearing": 1, "Business:Start-Org": 1, "Justice:Arrest-Jail": 1,
                   "Transaction:Transfer-Ownership": 1, "Justice:Execute": 1, "Justice:Sentence": 1, "Life:Be-Born": 1,
                   "Justice:Charge-Indict": 1, "Business:Declare-Bankruptcy": 1, "Justice:Convict": 1,
                   "Justice:Release-Parole": 1, "Justice:Pardon": 1, "Justice:Appeal": 1, "Business:Merge-Org": 1,
                   "Justice:Extradite": 1, "Life:Divorce": 1, "Justice:Acquit": 1}
    p_e_count=0
    p_e_list=[]
    for node in p_seq:
        node=node.replace(INIT,'')
        if node in Event_types:
            p_e_count+=1
            p_e_list.append(node)
    return p_e_list,p_e_count
def count_arg_fun(p_line):
    p_arg_count=len(p_line)
    p_arg_list=[]
    p_triple_list=[]
    for i,triple_item in enumerate(p_line):
        triple=triple_item[str(i)]
        args=[triple[0],triple[-1]]
        p_arg_list.append(args)
        p_triple_list.append(triple)
    return p_arg_list,p_triple_list,p_arg_count
def compute_F1(p,g):
    e_count=0
    e_list=[]
    arg_count=0
    pre_e_count=0
    pre_e_list=[]
    pre_arg_count=0
    correct_e_count=0
    correct_arg_count=0
    correct_triple_count=0
    for i,line in enumerate(g):
        e_list = []
        arg_list=[]
        triple_list=[]
        p_line=p[i]
        e_count+=len(line['golden_event_mentions'])
        for e in line['golden_event_mentions']:
            e_list.append(e['event_type'])
            arg_count+=len(e['arguments'])
            for a in e['arguments']:
                head=a['head']
                text=a['text']
                if head in text:
                    arg=head[-1]
                else:
                    arg=text[-1]
                role=a['role']
                arg_list.append([e['event_type'],arg])
                triple_list.append([e['event_type'],role,arg])
        pre_e_list,single_pre_e_count=count_pre_e_fun(p_line[1])
        pre_arg_list,pre_triple_list,single_pre_arg_count=count_arg_fun(p_line[0])
        pre_e_count+=single_pre_e_count
        pre_arg_count+=single_pre_arg_count
        for p_e in pre_e_list:
            if p_e in e_list:
                correct_e_count+=1
                e_list.remove(p_e)
        for p_arg in pre_arg_list:
            if p_arg in arg_list:
                correct_arg_count+=1
                arg_list.remove(p_arg)
        for p_triple in pre_triple_list:
            if p_triple in triple_list:
                correct_triple_count+=1
                triple_list.remove(p_triple)
    #事件检测
    e_detect_p=(correct_e_count/e_count)
    if pre_e_count==0:
        e_detect_r=0.0
    else:
        e_detect_r=(correct_e_count/pre_e_count)
    if int(e_detect_p)==0 or int(e_detect_r)==0:
        e_detect_f=0
    else:
        e_detect_f=((e_detect_p*e_detect_r)*2)/(e_detect_p+e_detect_r)
    #论元检测
    if arg_count==0:
        arg_detect_p=0
    else:
        arg_detect_p = (correct_arg_count / arg_count)
    if pre_arg_count==0:
        arg_detect_r=0
    else:
        arg_detect_r = (correct_arg_count / pre_arg_count)
    if int(arg_detect_p)==0 or int(arg_detect_r)==0:
        arg_detect_f=0
    else:
        arg_detect_f = ((arg_detect_p * arg_detect_r) * 2) / (arg_detect_p + arg_detect_r)
    #论元分类
    if arg_count==0:
        triple_detect_p=0
    else:
        triple_detect_p = (correct_triple_count / arg_count)
    if pre_arg_count==0:
        triple_detect_r=0
    else:
        triple_detect_r = (correct_triple_count / pre_arg_count)
    if int(triple_detect_p)==0 or int(triple_detect_r)==0:
        triple_detect_f=0
    else:
        triple_detect_f = ((triple_detect_p * triple_detect_r) * 2) / (triple_detect_p + triple_detect_r)
    a=0


if __name__=="__main__":
    file_path='D:\\bart_ee\ee_data\sample.json'
    data_lines=read_fun(file_path)
    p_path='D:\\bart_ee\output\pred2.json'
    p_data_lines = read_fun(p_path)
    compute_F1(p_data_lines,data_lines)
    a=0