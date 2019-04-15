
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):

    def __init__(self, numOfEntity, numOfRelation, entityDimension, relationDimension, margin, norm):
        super(TransE, self).__init__()

        self.numOfEntity = numOfEntity
        self.numOfRelation = numOfRelation
        self.entityDimension = entityDimension
        self.relationDimension = relationDimension
        self.margin = margin
        self.norm = norm

        sqrtR = relationDimension**0.5
        sqrtE = entityDimension**0.5

        self.relation_embeddings = nn.Embedding(self.numOfRelation, self.relationDimension)
        self.relation_embeddings.weight.data = torch.FloatTensor(self.numOfRelation, self.relationDimension).uniform_(-6/sqrtR, 6/sqrtR)
        self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, 2, 1)

        self.entity_embeddings = nn.Embedding(self.numOfEntity, self.entityDimension)
        self.entity_embeddings.weight.data = torch.FloatTensor(self.numOfEntity, self.entityDimension).uniform_(-6/sqrtE, 6/sqrtE)
        self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, 2, 1)

    def forward(self, positiveBatchHead, positiveBatchRelation, positiveBatchTail, corruptedBatchHead, corruptedBatchRelation, corruptedBatchTail):
        # print positiveBatches
        pH_embeddings = self.entity_embeddings(positiveBatchHead)
        pR_embeddings = self.relation_embeddings(positiveBatchRelation)
        pT_embeddings = self.entity_embeddings(positiveBatchTail)

        nH_embeddings = self.entity_embeddings(corruptedBatchHead)
        nR_embeddings = self.relation_embeddings(corruptedBatchRelation)
        nT_embeddings = self.entity_embeddings(corruptedBatchTail)

        pH_embeddings = F.normalize(pH_embeddings, 2, 1)
        # pR_embeddings = F.normalize(pR_embeddings, 2, 1)
        pT_embeddings = F.normalize(pT_embeddings, 2, 1)
        nH_embeddings = F.normalize(nH_embeddings, 2, 1)
        # nR_embeddings = F.normalize(nR_embeddings, 2, 1)
        nT_embeddings = F.normalize(nT_embeddings, 2, 1)

        positiveLoss = torch.norm(pH_embeddings + pR_embeddings - pT_embeddings, self.norm, 1)
        # set parameter "1": calculate the "self.norm"-norm of each row
        negativeLoss = torch.norm(nH_embeddings + nR_embeddings - nT_embeddings, self.norm, 1)
        # the size of negativeLoss: negativeTriples["h"].size()

        return torch.cat((positiveLoss, negativeLoss))
        # the size of returned tensor: positiveLoss.size()

    def fastValidate(self, validateHead, validateRelation, validateTail):

        validateHeadEmbedding = self.entity_embeddings(validateHead)
        validateRelationEmbedding = self.relation_embeddings(validateRelation)
        validateTailEmbedding = self.entity_embeddings(validateTail)

        targetLoss = torch.norm(validateHeadEmbedding + validateRelationEmbedding - validateTailEmbedding, self.norm).repeat(self.numOfEntity, 1)
        tmpHeadEmbedding = validateHeadEmbedding.repeat(self.numOfEntity, 1)
        tmpRelationEmbedding = validateRelationEmbedding.repeat(self.numOfEntity, 1)
        tmpTailEmbedding = validateTailEmbedding.repeat(self.numOfEntity, 1)

        tmpHeadLoss = torch.norm(self.entity_embeddings.weight.data + tmpRelationEmbedding - tmpTailEmbedding,
                                 self.norm, 1).view(-1, 1)
        tmpTailLoss = torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - self.entity_embeddings.weight.data,
                                 self.norm, 1).view(-1, 1)

        rankH = torch.nonzero(nn.functional.relu(targetLoss - tmpHeadLoss)).size()[0]
        rankT = torch.nonzero(nn.functional.relu(targetLoss - tmpTailLoss)).size()[0]

        return (rankH + rankT + 2)/2

    def fastTest(self, meanRank, testHead, testRelation, testTail, trainTriple, numOfTestTriple):
        testHeadEmbedding = self.entity_embeddings(testHead)
        testRelationEmbedding = self.relation_embeddings(testRelation)
        testTailEmbedding = self.entity_embeddings(testTail)

        targetLoss = torch.norm(testHeadEmbedding + testRelationEmbedding - testTailEmbedding, self.norm, 1).view(-1, 1).repeat(
            1, self.numOfEntity)

        tmpTmpEntityEmbedding = torch.unsqueeze(self.entity_embeddings.weight.data, 0)
        tmpEntityEmbedding = tmpTmpEntityEmbedding
        for i in torch.arange(0, numOfTestTriple-1):
            tmpEntityEmbedding = torch.cat((tmpEntityEmbedding, tmpTmpEntityEmbedding), 0)

        tmpTmpHeadEmbedding = torch.unsqueeze(testHeadEmbedding, 1)
        tmpHeadEmbedding = tmpTmpHeadEmbedding
        tmpTmpRelationEmbedding = torch.unsqueeze(testRelationEmbedding, 1)
        tmpRelationEmbedding = tmpTmpRelationEmbedding
        tmpTmpTailEmbedding = torch.unsqueeze(testTailEmbedding, 1)
        tmpTailEmbedding = tmpTmpTailEmbedding
        for i in torch.arange(0, self.numOfEntity-1):
            tmpHeadEmbedding = torch.cat((tmpHeadEmbedding, tmpTmpHeadEmbedding), 1)
            tmpRelationEmbedding = torch.cat((tmpRelationEmbedding, tmpTmpRelationEmbedding), 1)
            tmpTailEmbedding = torch.cat((tmpTailEmbedding, tmpTmpTailEmbedding), 1)

        headLoss = targetLoss - torch.norm(tmpEntityEmbedding + tmpRelationEmbedding - tmpTailEmbedding, self.norm, 2)
        tailLoss = targetLoss - torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - tmpEntityEmbedding, self.norm, 2)

        wrongHead = torch.nonzero(nn.functional.relu(headLoss))
        wrongTail = torch.nonzero(nn.functional.relu(tailLoss))

        numOfWrongHead = wrongHead.size()[0]
        numOfWrongTail = wrongTail.size()[0]

        numOfFilterHead = 0
        numOfFilterTail = 0

        for tmpWrongHead in wrongHead:
            numOfFilterHead += trainTriple[(trainTriple[:,0]==tmpWrongHead[1].float())&(trainTriple[:,1]==testRelation[tmpWrongHead[0]].float())&(trainTriple[:,2]==testTail[tmpWrongHead[0]].float())].size()[0]
        for tmpWrongTail in wrongTail:
            numOfFilterTail += trainTriple[(trainTriple[:,0]==testHead[tmpWrongTail[0]].float())&(trainTriple[:,1]==testRelation[tmpWrongTail[0]].float())&(trainTriple[:,2]==tmpWrongTail[1].float())].size()[0]

        meanRank[0] = ((numOfWrongHead + numOfWrongTail + 2)/2)/numOfTestTriple
        meanRank[1] = ((numOfWrongHead + numOfWrongTail + 2 - numOfFilterHead - numOfFilterTail)/2)/numOfTestTriple

    def test(self, meanRank, testHead, testRelation, testTail, trainTriple):
        testHeadEmbedding = self.entity_embeddings(testHead)
        testRelationEmbedding = self.relation_embeddings(testRelation)
        testTailEmbedding = self.entity_embeddings(testTail)

        targetLoss = torch.norm(testHeadEmbedding + testRelationEmbedding - testTailEmbedding, self.norm).repeat(self.numOfEntity, 1)
        tmpHeadEmbedding = testHeadEmbedding.repeat(self.numOfEntity, 1)
        tmpRelationEmbedding = testRelationEmbedding.repeat(self.numOfEntity, 1)
        tmpTailEmbedding = testTailEmbedding.repeat(self.numOfEntity, 1)

        tmpHeadLoss = torch.norm(self.entity_embeddings.weight.data + tmpRelationEmbedding - tmpTailEmbedding,
                                 self.norm, 1).view(-1, 1)
        tmpTailLoss = torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - self.entity_embeddings.weight.data,
                                 self.norm, 1).view(-1, 1)

        unCorrH = torch.nonzero(nn.functional.relu(targetLoss - tmpHeadLoss))[:, 0]
        unCorrT = torch.nonzero(nn.functional.relu(targetLoss - tmpTailLoss))[:, 0]

        numOfWrongHead = unCorrH.size()[0]
        numOfWrongTail = unCorrT.size()[0]

        numOfFilterHead = 0
        numOfFilterTail = 0

        for wrongHead in unCorrH:
            if trainTriple[(trainTriple[:,0]==wrongHead.float())&(trainTriple[:,1]==testRelation.float())&(trainTriple[:,2]==testTail.float())].size()[0]:
                numOfFilterHead += 1
        for wrongTail in unCorrT:
            if trainTriple[(trainTriple[:,0]==testHead.float())&(trainTriple[:,1]==testRelation.float())&(trainTriple[:,2]==wrongTail.float())].size()[0]:
                numOfFilterTail += 1

        meanRank[0] = (numOfWrongHead + numOfWrongTail + 2)/2
        meanRank[1] = (numOfWrongHead + numOfWrongTail + 2 - numOfFilterHead - numOfFilterTail)/2






    '''
    def validate(self, numOfValidateTriple, validateHead, validateRelation, validateTail):
        validateHeadEmbedding = self.entity_embeddings(validateHead)
        validateRelationEmbedding = self.relation_embeddings(validateRelation)
        validateTailEmbedding = self.entity_embeddings(validateTail)

        targetLoss = torch.norm(validateHeadEmbedding + validateRelationEmbedding - validateTailEmbedding, self.norm, 1)

        fMeanRankH = 0
        fMeanRankT = 0

        for tmpTriple in range(numOfValidateTriple):
            # print "processing " + str(tmpTriple)
            tmpHeadEmbedding = validateHeadEmbedding[tmpTriple].repeat(self.numOfEntity, 1)
            tmpRelationEmbedding = validateRelationEmbedding[tmpTriple].repeat(self.numOfEntity, 1)
            tmpTailEmbedding = validateTailEmbedding[tmpTriple].repeat(self.numOfEntity, 1)

            tmpTargetLoss = targetLoss[tmpTriple]

            tmpHeadLoss = torch.norm(self.entity_embeddings.weight.data + tmpRelationEmbedding - tmpTailEmbedding, self.norm, 1)
            tmpTailLoss = torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - self.entity_embeddings.weight.data, self.norm, 1)

            tmpFMeanRankH = 1
            tmpFMeanRankT = 1

            for i in range(self.numOfEntity):
                if tmpTargetLoss > tmpHeadLoss[i]:
                    tmpFMeanRankH += 1
                    if i in self.headRelation2Tail.keys():
                        if validateRelation[tmpTriple].item() in self.headRelation2Tail[i].keys():
                            if validateTail[tmpTriple].item() in self.headRelation2Tail[i][validateRelation[tmpTriple].item()]:
                                tmpFMeanRankH -= 1
                if tmpTargetLoss > tmpTailLoss[i]:
                    tmpFMeanRankT += 1
                    if validateHead[tmpTriple].item() in self.headRelation2Tail.keys():
                        if validateRelation[tmpTriple].item() in self.headRelation2Tail[validateHead[tmpTriple].item()].keys():
                            if i in self.headRelation2Tail[validateHead[tmpTriple].item()][validateRelation[tmpTriple].item()]:
                                tmpFMeanRankT -= 1

            fMeanRankH += tmpFMeanRankH
            fMeanRankT += tmpFMeanRankT

        return (fMeanRankH + fMeanRankT)/(2 * numOfValidateTriple)
        '''


