#coding=utf-8
import socket
import time
import json
import numpy as np
HOST='localhost'
PORT=50007
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)      #定义socket类型，网络通信，TCP
s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
s.connect((HOST,PORT))       #要连接的IP与端口
message_to_send= {'task': 'replay',
                       'intention': 'inform',
                       'slot': {}}
message_to_send1= {'task': 'replay',
                       'intention': 'inform',
                       'slot': {'time':'8点30'}}
message_to_send2= {'task': 'replay',
                       'intention': 'inform',
                       'slot': {'channelname':'cctv1'}}
message_to_send3= {'task': 'replay',
                       'intention': 'inform',
                       'slot': {'data':'2018-06-16'}}
message_to_send4= {'task': 'replay',
                       'intention': 'inform',
                       'slot': {'program':'新闻联播'}}
replaytask=[message_to_send,message_to_send1,message_to_send2,message_to_send3,message_to_send4]
task5={'task':'order','intention':'inform','slot':{'videoName':'xxxx','modifier':'xxxxx'}}
task4={'task':'cmd','intention':'cmd','slot':{'cmdValue':'NEXTPAGE','cmdParam':'1s'}}
task6={'task':'live','intention':'inform','slot':{'channelname':'xxxx'}}
task3={'task':'cmd','intention':'cmd','slot':{'cmdValue':'xx','cmdParam':'1s'}}
total=[message_to_send,message_to_send1,message_to_send2,task3,task4,task5,task6]

i=0
flag=1
sever=1
while flag:      #与人交互，输入命令
       messageto_sends=replaytask[np.random.randint(0, len(replaytask))]
       s.send(json.dumps(messageto_sends).encode())      #把命令发送给对端
       data=s.recv(1024)     #把接收的数据定义为变量
       if data.decode() =='done':
           print(data.decode())
           #flag=0
       else:
           print(data.decode())
