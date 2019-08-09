# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：daguanjinsai -> pre_process
@IDE    ：PyCharm
@Author ：Zhang zhe
@Date   ：2019/7/3 15:51
=================================================='''
import collections

def longest():
    longest=0
    with open("./datagrand/train.txt") as f:
        for line in f.readlines():
            line=line.strip().split("  ")
            tem_long=len(line)
            if tem_long>longest:
                longest=tem_long

    with open("./datagrand/test.txt") as f:
        for line in f.readlines():
            line = line.strip().split("_")
            tem_long=len(line)
            if tem_long>longest:
                longest=tem_long

    return longest



if __name__=="__main__":
    m_length=longest()
    print(m_length)
