import re
import torch

class readData:
    def __init__(self, inAdd, train2id, headRelation2Tail, tailRelation2Head, entity2id, id2entity, relation2id, id2relation, nums):
        self.inAdd = inAdd
        self.train2id = train2id
        self.headRelation2Tail = headRelation2Tail
        self.tailRelation2Head = tailRelation2Head
        self.nums = nums
        self.entity2id = entity2id
        self.id2entity = id2entity
        self.relation2id = relation2id
        self.id2relation = id2relation
        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0

        self.trainTriple = None

        self.readTrain2id()
        print "number of triples: " + str(self.numOfTriple)

        self.readEntity2id()
        print "number of entities: " + str(self.numOfEntity)

        self.readRelation2id()
        print "number of relations: " + str(self.numOfRelation)

        self.nums[0] = self.numOfTriple
        self.nums[1] = self.numOfEntity
        self.nums[2] = self.numOfRelation

        # print self.numOfTriple
        # print self.train2id
        # print self.numOfEntity
        # print self.entity2id
        # print self.id2entity
        # print self.numOfRelation
        # print self.relation2id
        # print self.id2relation
        # print self.headRelation2Tail
        # print self.tailRelation2Head

    def out(self):
        return self.trainTriple

    def readTrain2id(self):
        print "-----Reading train2id.txt from " + self.inAdd + "/-----"
        count = 0
        inputData = open(self.inAdd + "/train2id.txt")
        line = inputData.readline()
        self.numOfTriple = int(re.findall(r"\d+", line)[0])
        self.train2id["h"] = []
        self.train2id["t"] = []
        self.train2id["r"] = []
        self.trainTriple = torch.ones(self.numOfTriple, 3)
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(re.findall(r"\d+", line)[0])
                tmpTail = int(re.findall(r"\d+", line)[1])
                tmpRelation = int(re.findall(r"\d+", line)[2])
                self.train2id["h"].append(tmpHead)
                self.train2id["t"].append(tmpTail)
                self.train2id["r"].append(tmpRelation)
                self.trainTriple[count, 0] = tmpHead
                self.trainTriple[count, 1] = tmpRelation
                self.trainTriple[count, 2] = tmpTail
                if tmpHead not in self.headRelation2Tail.keys():
                    self.headRelation2Tail[tmpHead] = {}
                    self.headRelation2Tail[tmpHead][tmpRelation] = []
                    self.headRelation2Tail[tmpHead][tmpRelation].append(tmpTail)
                else:
                    if tmpRelation not in self.headRelation2Tail[tmpHead].keys():
                        self.headRelation2Tail[tmpHead][tmpRelation] = []
                        self.headRelation2Tail[tmpHead][tmpRelation].append(tmpTail)
                    else:
                        if tmpTail not in self.headRelation2Tail[tmpHead][tmpRelation]:
                            self.headRelation2Tail[tmpHead][tmpRelation].append(tmpTail)
                if tmpTail not in self.tailRelation2Head.keys():
                    self.tailRelation2Head[tmpTail] = {}
                    self.tailRelation2Head[tmpTail][tmpRelation] = []
                    self.tailRelation2Head[tmpTail][tmpRelation].append(tmpHead)
                else:
                    if tmpRelation not in self.tailRelation2Head[tmpTail].keys():
                        self.tailRelation2Head[tmpTail][tmpRelation] = []
                        self.tailRelation2Head[tmpTail][tmpRelation].append(tmpHead)
                    else:
                        if tmpHead not in self.tailRelation2Head[tmpTail][tmpRelation]:
                            self.tailRelation2Head[tmpTail][tmpRelation].append(tmpHead)
                count += 1
                line = inputData.readline()
            else:
                print "error in train2id.txt at Line " + str(count + 2)
                line = inputData.readline()
        inputData.close()
        if count == self.numOfTriple:
            self.trainTriple.long()
            return
        else:
            print "error in train2id.txt"
            return

    def readEntity2id(self):
        print "-----Reading entity2id.txt from " + self.inAdd + "/-----"
        count = 0
        inputData = open(self.inAdd + "/entity2id.txt")
        line = inputData.readline()
        self.numOfEntity = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.search(r"(.+)\t(\d+)", line)
            if reR:
                entity = reR.group(1)
                Eid = reR.group(2)
                self.entity2id[entity] = int(Eid)
                self.id2entity[int(Eid)] = entity
                count += 1
                line = inputData.readline()
            else:
                print "error in entity2id.txt at line " + str(count + 2)
                line = inputData.readline()
        inputData.close()
        if count == self.numOfEntity:
            return
        else:
            print "error in entity2id.txt"
            return

    def readRelation2id(self):
        print "-----Reading relation2id.txt from " + self.inAdd + "/-----"
        count = 0
        inputData = open(self.inAdd + "/relation2id.txt")
        line = inputData.readline()
        self.numOfRelation = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.search(r"(.+)\t(\d+)", line)
            if reR:
                relation = reR.group(1)
                Rid = int(reR.group(2))
                self.relation2id[relation] = Rid
                self.id2relation[Rid] = relation
                line = inputData.readline()
                count += 1
            else:
                print "error in relation2id.txt at line " + str(count + 2)
                line = inputData.readline()
        inputData.close()
        if count == self.numOfRelation:
            return
        else:
            print "error in relation2id.txt"
            return
