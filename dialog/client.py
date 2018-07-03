#coding=utf-8
import socket
import time
import json
HOST='localhost'
PORT=50007
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)      #定义socket类型，网络通信，TCP
s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
s.connect((HOST,PORT))       #要连接的IP与端口
message_to_send= {'task': 'replay',
                       'intention': 'inform',
                       'slot': {'data':'xxxx'}}
message_to_send1= {'task': 'replay',
                       'intention': 'inform',
                       'slot': {'time':'xxxx'}}
message_to_send2= {'task': 'replay',
                       'intention': 'inform',
                       'slot': {'channelname':'xxxx'}}
task3={'task':'cmd','intention':'cmd','slot':{'cmdValue':'xx','cmdParam':'1s'}}
total=[message_to_send,message_to_send1,message_to_send2,task3]

i=0
flag=1
sever=1
while flag:      #与人交互，输入命令
       messageto_sends=total[i%4]
       i+=1
       s.send(json.dumps(messageto_sends).encode())      #把命令发送给对端
       data=s.recv(1024)     #把接收的数据定义为变量
       if data.decode() =='done':
           print(data.decode())
           #flag=0
       else:
           print(data.decode())
