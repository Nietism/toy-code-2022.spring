# # -*- coding: utf-8 -*-
# """
# @Time   :   2021-11-25 21:08:29
# @File   :   macbert_corrector.py
# @Author :   wangzw
# @Email  :   wangzw@njnu.edu.cn
# """
# # import ...
# import operator
# import sys
# import time
# from pycorrector.utils.text_utils import convert_to_unicode
# from pycorrector.utils.logger import logger
# from pycorrector.macbert.correction_pipeline import CorrectionPipeline
# from pycorrector import config
# from pycorrector.transformers import BertTokenizer, BertForMaskedLM
# from pycorrector.utils.tokenizer import split_text_by_maxlen
#
# sys.path.append('../..')
#
#
# class MacBertCorrector(object):
#     def __init__(self, macbert_model_dir=config.macbert_model_dir):
#         super(MacBertCorrector, self).__init__()
#         self.name = 'macbert_corrector'
#         t1 = time.time()
#         macbert_model = BertForMaskedLM.from_pretrained(macbert_model_dir)
#         tokenizer = BertTokenizer.from_pretrained(macbert_model_dir)
#         self.model = CorrectionPipeline(
#             task='correction',
#             model=macbert_model,
#             tokenizer=tokenizer,
#             device=0,  # gpu device id
#         )
#         if self.model:
#             logger.debug('Loaded macbert model: %s, spend: %.3f s.' % (macbert_model_dir, time.time() - t1))
#
#     def macbert_correct(self, text):
#         """
#         句子纠错
#         :param text: 句子文本
#         :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
#         """
#         details = []
#         # 编码统一，utf-8 to unicode
#         text = convert_to_unicode(text)
#         # 长句切分为短句
#         blocks = split_text_by_maxlen(text, maxlen=128)
#         blocks = [block[0] for block in blocks]
#         results = self.model(blocks)
#         text_new = ''.join([rst['corrected_text'] for rst in results])
#
#         text_count = 0
#         text_new_count = 0
#         len_text = len(text)
#         len_text_new = len(text_new)
#         while text_count < len_text and text_new_count < len_text_new:
#             # pipeline 处理后的 text_new 不含空格，在此处补充空格。
#             if text[text_count] == ' ':
#                 text_new = text_new[:text_new_count] + ' ' + text_new[text_new_count:]
#                 continue
#             if text[text_count] != text_new[text_new_count]:
#                 # 多字处理
#                 if text_count + 1 < len_text and text[text_count + 1] == text_new[text_new_count]:
#                     details.append((text[text_count], '_', text_count, text_count + 1))
#                     text_count += 1
#                 # 少字处理
#                 elif text_new_count + 1 < len_text_new and text_new[text_new_count + 1] == text[text_count]:
#                     details.append(('_', text_new[text_new_count], text_count, text_count + 1))
#                 # 不改变字数的错误处理
#                 else:
#                     details.append((text[text_count], text_new[text_new_count], text_count, text_count + 1))
#             text_count += 1
#             text_new_count += 1
#
#         details = sorted(details, key=operator.itemgetter(2))
#         return text_new, details
#
#
# def move_quotationmark(str_sentence):
#     flag = 0
#     quo_list = list()
#     list_sentence = list(str_sentence)
#     while flag < len(list_sentence):
#         if list_sentence[flag] == '“' or list_sentence[flag] == '”' or list_sentence[flag] == '‘' or list_sentence[flag] == '’':
#             quo_list.append([flag, list_sentence[flag]])
#             list_sentence.pop(flag)
#         else:
#             flag += 1
#     if len(list_sentence) == len(str_sentence):
#         quo_list.append([0, ''])
#     for i in range(len(quo_list)):
#         quo_list[i][0] += i
#     sentence_movequo = "".join(list_sentence)
#     return sentence_movequo, quo_list
#
# if __name__ == "__main__":
#     # 载入input.txt，创建output.txt
#     f_input = open("input.txt", 'r', encoding='utf-8')
#     input_lines = f_input.readlines()
#     error_sentences = list()
#     f_output = open("output.txt", 'w+')
#     num_list = list()
#     quotation_mark_lists = list()
#     # 载入模型
#     m = MacBertCorrector('user_data/datasets/macbert_models/chinese_finetuned_correction')
#     for input_line in input_lines:
#         # 将带标号的句子拆分成标号和句子
#         # error_sentence为去除标号的句子，num_list为句子标号
#         error_sentence = input_line[input_line.index(')') + 1:-1] + '。'
#         num_list.append(input_line[0: input_line.index(')') + 1])
#         # 将模型不能处理的引号去除，并保存
#         # error_sentence_moveQuo为去除标号和引号后的句子，quotation_mark_list为[引号的位置, 引号的类型]
#         error_sentence_moveQuo, quotation_mark_list = move_quotationmark(error_sentence)
#         error_sentences.append(error_sentence_moveQuo)
#         quotation_mark_lists.append(quotation_mark_list)
#
#     for i in range(len(error_sentences)):
#         # 将数据代入模型
#         corrected_sentence, error_informations = m.macbert_correct(error_sentences[i])
#
#         error_sentence_list = list(error_sentences[i])
#         error_sentence_list.pop()
#         error_sentence_list.append('|')
#
#         error_count = 0
#         quotation_count = 0
#         left_bracket_list = []
#         # 错误标注
#         for error_information in error_informations:
#             if error_information[0] != '_':
#                 error_sentence_list.insert(error_information[2] + error_count * 2, '【')
#                 error_sentence_list.insert(error_information[2] + error_count * 2 + 2, '】')
#             else:
#                 error_sentence_list.insert(error_information[2] + error_count * 2, '【】')
#             left_bracket_list.append(error_information[2] + error_count * 2)
#             error_count += 1
#             error_sentence_list.append(error_information[1])
#
#         for j in range(len(quotation_mark_lists[i])):
#             left_bracket_flag = 0
#             len_left_bracket_list = len(left_bracket_list)
#             while left_bracket_flag < len_left_bracket_list and (quotation_mark_lists[i][j][0] + left_bracket_flag * 2 > left_bracket_list[left_bracket_flag]):
#                 left_bracket_flag += 1
#             error_sentence_list.insert(quotation_mark_lists[i][j][0] + left_bracket_flag * 2, quotation_mark_lists[i][j][1])
#
#         output_sentence = ''.join(error_sentence_list)
#         # 添加标号
#         output_line = num_list.pop(0) + output_sentence + '\n'
#         f_output.writelines(output_line)
#     # 完成
#     f_input.close()
#     f_output.close()
#     print("\nHey!The output.txt has been written!")

# -*- coding: utf-8 -*-
"""
@Time   :   2021-11-25 21:08:29
@File   :   macbert_corrector.py
@Author :   wangzw
@Email  :   wangzw@njnu.edu.cn
"""
import operator
import sys
import time
import os
from transformers import BertTokenizer, BertForMaskedLM
import torch
from pycorrector.utils.text_utils import convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector import config
from pycorrector.utils.tokenizer import split_text_by_maxlen

sys.path.append('../..')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MacBertCorrector(object):
    def __init__(self, macbert_model_dir=config.macbert_model_dir):
        super(MacBertCorrector, self).__init__()
        self.name = 'macbert_corrector'
        t1 = time.time()
        if not os.path.exists(os.path.join(macbert_model_dir, 'vocab.txt')):
            macbert_model_dir = "shibing624/macbert4csc-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(macbert_model_dir)
        self.model = BertForMaskedLM.from_pretrained(macbert_model_dir)
        self.model.to(device)
        self.unk_tokens = [' ', '“', '”', '‘', '’', '…', '—']
        logger.debug("device: {}".format(device))
        logger.debug('Loaded macbert model: %s, spend: %.3f s.' % (macbert_model_dir, time.time() - t1))

    def macbert_correct(self, text):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=128)
        blocks = [block[0] for block in blocks]
        inputs = self.tokenizer(blocks, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        def get_errors(text_new, text):
            details = list()
            text_count = 0
            text_new_count = 0
            len_text = len(text)
            len_text_new = len(text_new)
            while text_count < len_text and text_new_count < len_text_new:
                # pipeline 处理后的 text_new 不含空格，在此处补充空格。
                if text[text_count] == ' ':
                    text_new = text_new[:text_new_count] + ' ' + text_new[text_new_count:]
                    continue
                if text[text_count] != text_new[text_new_count]:
                    # 多字处理
                    if text_count + 1 < len_text and text[text_count + 1] == text_new[text_new_count]:
                        details.append((text[text_count], '_', text_count, text_count + 1))
                        text_count += 1
                    # 少字处理
                    elif text_new_count + 1 < len_text_new and text_new[text_new_count + 1] == text[text_count]:
                        details.append(('_', text_new[text_new_count], text_count, text_count + 1))
                    # 不改变字数的错误处理
                    else:
                        details.append((text[text_count], text_new[text_new_count], text_count, text_count + 1))
                text_count += 1
                text_new_count += 1
            details = sorted(details, key=operator.itemgetter(2))
            return corrected_text, details

        for ids, text in zip(outputs.logits, blocks):
            decode_tokens = self.tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
            corrected_text = decode_tokens[:len(text)]
            corrected_text, sub_details = get_errors(corrected_text, text)
            text_new += corrected_text
            details.extend(sub_details)
        return text_new, details

    def move_unk_tokens(self, str_sentence):
        flag = 0
        quo_list = list()
        list_sentence = list(str_sentence)
        while flag < len(list_sentence):
            if list_sentence[flag] in self.unk_tokens:
                quo_list.append([flag, list_sentence[flag]])
                list_sentence.pop(flag)
            else:
                flag += 1
        if len(list_sentence) == len(str_sentence):
            quo_list.append([0, ''])
        for i in range(len(quo_list)):
            quo_list[i][0] += i
        sentence_movequo = "".join(list_sentence)
        return sentence_movequo, quo_list

if __name__ == "__main__":
    # 载入input.txt，创建output.txt
    f_input = open("input.txt", 'r', encoding='utf-8')
    input_lines = f_input.readlines()
    error_sentences = list()
    f_output = open("output.txt", 'w+')
    num_list = list()
    quotation_mark_lists = list()
    # 载入模型
    m = MacBertCorrector('user_data/datasets/macbert_models/chinese_finetuned_correction')
    for input_line in input_lines:
        # 将带标号的句子拆分成标号和句子
        # error_sentence为去除标号的句子，num_list为句子标号
        error_sentence = input_line[input_line.index(')') + 1:-1] + '。'
        num_list.append(input_line[0: input_line.index(')') + 1])
        # 将模型不能处理的引号去除，并保存
        # error_sentence_moveQuo为去除标号和引号后的句子，quotation_mark_list为[引号的位置, 引号的类型]
        error_sentence_moveQuo, quotation_mark_list = m.move_unk_tokens(error_sentence)
        error_sentences.append(error_sentence_moveQuo)
        quotation_mark_lists.append(quotation_mark_list)

    for i in range(len(error_sentences)):
        # 将数据代入模型
        corrected_sentence, error_informations = m.macbert_correct(error_sentences[i])

        error_sentence_list = list(error_sentences[i])
        error_sentence_list.pop()
        error_sentence_list.append('|')

        error_count = 0
        quotation_count = 0
        left_bracket_list = []
        # 错误标注
        for error_information in error_informations:
            if error_information[0] != '_':
                error_sentence_list.insert(error_information[2] + error_count * 2, '【')
                error_sentence_list.insert(error_information[2] + error_count * 2 + 2, '】')
            else:
                error_sentence_list.insert(error_information[2] + error_count * 2, '【】')
            left_bracket_list.append(error_information[2] + error_count * 2)
            error_count += 1
            error_sentence_list.append(error_information[1])

        for j in range(len(quotation_mark_lists[i])):
            left_bracket_flag = 0
            len_left_bracket_list = len(left_bracket_list)
            while left_bracket_flag < len_left_bracket_list and (quotation_mark_lists[i][j][0] + left_bracket_flag * 2 > left_bracket_list[left_bracket_flag]):
                left_bracket_flag += 1
            error_sentence_list.insert(quotation_mark_lists[i][j][0] + left_bracket_flag * 2, quotation_mark_lists[i][j][1])

        output_sentence = ''.join(error_sentence_list)
        # 添加标号
        output_line = num_list.pop(0) + output_sentence + '\n'
        f_output.writelines(output_line)
    # 完成
    f_input.close()
    f_output.close()
    print("\nHey!The output.txt has been written!")