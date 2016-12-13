import sys
# line = sys.stdin.readline()
N = 2  # 学生数目
M = 2   # 办公室数目
K = 100  # 办公室之间走动的时间
# print(line)

global current_time

class office(object):
    def __init__(self,id,free_student,finish_student,free_time_list,K):
        self.id = id
        self.freetime = None
        self.is_dealing = None
        self.queue = []
        self.free_student = free_student
        self.finish_student = finish_student
        self.free_time_list = free_time_list
        self.K = K

    def put(self,stu_info):
        # stu_id = stu_info['id']
        # stu_time = stu_info['reach_time']
        # stu_o = stu_info['o']
        # stu_w = stu_info['w']
        self.queue.append(stu_info)
        if self.is_dealing == None:
            self.get_one()

    def get_one(self):
        assert self.is_dealing != None
        if self.queue.__len__()==0:
            return
        self.queue = sorted(self.queue, key=lambda x:x['id'])
        stu_info = self.queue.pop(0)
        self.is_dealing = stu_info
        o = stu_info['o']
        w = stu_info['w']
        assert o[0] == self.id
        global current_time
        self.freetime = current_time + w[0]
        self.free_time_list.append([self.id,self.freetime])

    def free(self):
        if self.is_dealing == None :
            pass
        else:
            stu_info = self.is_dealing
            stu_info['o'].pop(0)
            stu_info['w'].pop(0)
            global  current_time
            if stu_info['o'].__len__()==0:
                stu_info.pop('next_office')
                stu_info.pop('next_time')
                stu_info['finish_time'] = current_time
                stu_info['total_time'] = current_time - stu_info['reach_time']
                self.finish_student.append(stu_info)
            stu_info['next_office'] = stu_info['o'][0]
            stu_info['next_time'] = current_time + self.K
            self.free_student.append(stu_info)
            self.is_dealing = None

    def action(self):
        self.free()
        self.get_one()

class manager():
    def __init__(self,N,M,K):
        self.N = N
        self.M = M
        self.K = K
        self.office_list = []
        self.free_time_list = []
        self.free_student = []
        self.finish_student = []
        for i in range(M):
            o = office(i+1,self.free_student,self.finish_student,self.free_time_list,self.K)
            self.office_list.append(o)

    def add_line(self,line):
        l = line.split(',')
        id = str(l[0])
        reach_time = int(l[1])
        p = int(l[2])
        o = []
        w = []
        for i in range(p):
            o.append(int(l[3+i*2]))
            w.append(int(l[4+i*2]))
        info = dict(
            id = id,
            reach_time = reach_time,
            o = o,
            w = w,
            next_office = o[0],
            next_time = reach_time,
        )
        self.free_student.append(info)

    def run(self):
        pass




