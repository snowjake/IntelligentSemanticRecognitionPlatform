import re
# p = r'CCTV[-]?[1]?[-]?{HD}*{高清}*[综合 综合频道]*'
# p = r'CCTV[-]?1[-]?[HD ]*[高清 ]*[综合 综合频道]*'  # yes
# p1 = r'CCTV[-]?[1]?[-]?[HD ]*[高清 ]*[综合 综合频道]+'  # yes
# """
# CCTV-1 CCTV-1HD CCTV1 CCTV1HD	CCTV1-HD,CCTV1高清,CCTV-1高清,
# CCTV1-高清,CCTV-1高清,CCTV-1综合,CCTV-1综合频道,CCTV综合,CCTV综合频道
# """
# all = ['CCTV-1', 'CCTV-1HD', '啊啊CCTV1', 'aaCCTV1HD', 'CCTV1-HD', 'CCTV1高清', '看CCTV-1高清吧', 'CCTV1-高清',
#        'CCTV-1综合', '我想看CCTV-1综合频道', 'CCTV1-综合', 'CCTV1-综合频道', 'CCTV综合', 'CCTV综合频道', 'CCTV1综合',
#        'CCTV1综合频道', 'CCTV']
# for sss in all:
#     print(re.findall(p, sss))
# print('$ ' * 100)
# for sss in all:
#     print(re.findall(p1, sss))

p = r'CCTV[-]?2[-]?[HD ]*[高清 ]*[财经 财经频道]*'
p1 = r'CCTV[-]?[2]?[-]?[HD ]*[高清 ]*[财经 财经频道]+'
all1 = ['CCTV-2', 'CCTV-2HD', 'CCTV2', 'CCTV2HD', 'CCTV2-HD', 'CCTV2高清', 'CCTV-2高清', 'CCTV2-高清', 'CCTV-2财经',
        'CCTV2-财经', '看看你CCTV-2财经频道', '我要看CCTV2-财经频道', 'CCTV财经', 'CCTV财经频道', '呵呵CCTV2财经', 'CCTV2财经频道']
for sss in all1:
    print(re.findall(p, sss))
print('$ ' * 100)
for sss in all1:
    print(re.findall(p1, sss))
# p = r'CCTV[5]?[+]?[-]?[HD ]*[体育赛事 体育赛事频道]+'
# p1 = r'CCTV5\+[-]?[HD ]*[体育赛事 体育赛事频道]*'
# all2 = ['CCTV5+HD', 'CCTV5+', 'CCTV体育赛事', 'CCTV体育赛事频道', 'CCTV']
# for sss in all2:
#     print(re.findall(p, sss))
# print('&' * 100)
# for sss in all2:
#     print(re.findall(p1, sss))


sssss = [
    '芒果台---湖南卫视',
    '番茄台---东方卫视',
    '大蒜台---江苏卫视',
    '辣椒台---江西卫视',
    '金鱼台---新疆卫视',
    '气球台---青海卫视',
    '星星台---辽宁卫视',
    '闪电台---陕西卫视',
    '旋风台---四川卫视',
    '波浪台---内蒙卫视',
    '地球台---重庆卫视',
    '飞碟台---西藏卫视',
    '海豚台---安徽卫视',
    '大象台---河南卫视',
    '孔雀台---云南卫视',
    '龙台---黑龙江卫视',
    '鱼钩台---天津卫视',
    '长城台---河北卫视',
    '2台---浙江卫视',
    '股市台---宁夏卫视',
    '柠檬台---湖北卫视',
    '手链台---旅游卫视',
    '围巾台---山东卫视',
    '天鹅台---吉林卫视',
    'B T台---北京卫视',
    '玫瑰台---山西卫视',
    'S台---甘肃卫视',
    'F台---东南卫视',
    '红领巾台-广东卫视',
    '飘带台---广西卫视'
]
for ggg in sssss:
    s_ss = ggg.strip().split('---')
    print('\t'.join([s_ss[1], s_ss[0]]))