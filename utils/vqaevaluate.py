import numpy as np
import json
import os.path as osp

class VQAEval:
    def __init__(self, ans_path, n = 8):
        self.n = n
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                            "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                            "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                            "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                            "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                            "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                            "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                            "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                            "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                            "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                            "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                            "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                            "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                            "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                            "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                            "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                            "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                            "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                            "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                            "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.digits = {
            'zero' : 0,
            'one' : 1,
            'two' : 2,
            'three' : 3,
            'four' : 4,
            'five' : 5,
            'six' : 6,
            'seven' : 7,
            'eight' : 8,
            'nine' : 9
        }
        self.articles = {'a', 'an', 'the'}
        self.answer = {'train' : json.load(open(osp.join(ans_path, "train.json"), 'r', encoding='utf-8')), 
                       'test' : json.load(open(osp.join(ans_path, "test.json"), 'r', encoding='utf-8')), 
                       'val' : json.load(open(osp.join(ans_path, "val.json"), 'r', encoding='utf-8'))}
        self.freq = {}
        self.acc = {'train' : {}, 'test' : {}, 'val' : {}}


    def get_freq(self):
        """
        对于freq小于n的answer，采取删除的措施
        此函数运行结束之后self.freq中保存了满足条件的answer，key为answer，value为answer的index
        对于其他不在字典中的回答，对应的ans_idx都为0，用self.freq.get(word, 0)即可
        """
        for answers in self.answer['train']['annotations']:
            stemmed_word = self.contractions.get(answers['multiple_choice_answer'], answers['multiple_choice_answer'])
            self.freq[stemmed_word] = self.freq.get(stemmed_word, 0) + 1
        tmp = []
        for key, val in self.freq.items():
            if val < self.n:
                tmp.append(key)
        for i in tmp:
            self.freq.pop(i)
        
        for idx, key in enumerate(list(self.freq.keys())):
            self.freq[key] = idx + 1
        self.freq['<unk>'] = 0

    def get_vec(self, split = 'train'):
        """
        根据公式计算出最终的Acc值
        """
        for annotation in self.answer[split]['annotations']:
            self.acc[split][annotation['question_id']] = np.zeros( len(list(self.freq.keys())) ).astype(np.float32)
            for ans in annotation['answers']:
                stemmed_word = self.contractions.get(ans["answer"], ans["answer"])
                idx = self.freq.get(stemmed_word, 0)
                self.acc[split][annotation['question_id']][idx] += 1
            self.acc[split][annotation['question_id']] = np.where(self.acc[split][annotation['question_id']] >= 3, 1, self.acc[split][annotation['question_id']] / 3)


    def process_digit(self, split = 'train'):
        """
        对数据进行预处理，包括规范化，统一大小写，去掉和更改部分单词等
        """
        for idx, answers in enumerate(self.answer[split]['annotations']):
            self.answer[split]['annotations'][idx]['multiple_choice_answer'] = str(self.answer[split]['annotations'][idx]['multiple_choice_answer']).lower()
            self.answer[split]['annotations'][idx]['multiple_choice_answer'] = self.process_single_sentence(self.answer[split]['annotations'][idx]['multiple_choice_answer'])
            for idx1, ans in enumerate(answers['answers']):
                self.answer[split]['annotations'][idx]['answers'][idx1]['answer'] = str(self.answer[split]['annotations'][idx]['answers'][idx1]['answer']).lower()
                self.answer[split]['annotations'][idx]['answers'][idx1]['answer'] = self.process_single_sentence(self.answer[split]['annotations'][idx]['answers'][idx1]['answer'])
    

    def process_single_sentence(self, sentence):
        s = ' '
        tmp = sentence.split(",")
        arr = []
        for i in tmp:
            for j in i.split():
                if j in self.articles:
                    continue
                j = self.digits.get(j, j)
                j = self.contractions.get(j, j)
                arr.append(str(j))
        return s.join(arr)


    def run(self):
        split = {'train', 'test', 'val'}
        for i in split:
            self.process_digit(split = i)
        self.get_freq()
        for i in split:
            self.get_vec(split = i)
    
    def get_acc(self, split = 'train'):
        return self.acc[split]