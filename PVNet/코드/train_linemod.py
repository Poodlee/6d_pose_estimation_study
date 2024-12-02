import sys

from skimage.io import imsave


sys.path.append('.')
sys.path.append('..')
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3, \
    estimate_voting_distribution_with_mean, ransac_voting_layer_v5, ransac_motion_voting
from lib.networks.model_repository import *
from lib.datasets.linemod_dataset import LineModDatasetRealAug, ImageSizeBatchSampler, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB, TruncatedLineModImageDB
from lib.utils.arg_utils import args
from lib.utils.draw_utils import visualize_bounding_box, imagenet_to_uint8, visualize_mask, visualize_points, img_pts_to_pts_img
from lib.utils.base_utils import save_pickle
import json

from lib.utils.evaluation_utils import Evaluator
from lib.utils.net_utils import AverageMeter, Recorder, smooth_l1_loss, \
    load_model, save_model, adjust_learning_rate, compute_precision_recall, set_learning_rate
from lib.utils.config import cfg

from torch.nn import DataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn, optim
import torch
import torch.nn.functional as F
import os
import time
from collections import OrderedDict
import random
import numpy as np

# --cfg_file configs/linemod_train.json 이런 식으로 받은 아규먼트 받아서 읽어서 train_cfg 생성 
with open(args.cfg_file,'r') as f:
    train_cfg=json.load(f)
train_cfg['model_name']='{}_{}'.format(args.linemod_cls,train_cfg['model_name']) # --linemod_cls cat 이런 식으로 받은 아규먼트~ 해서 고양이클래스하시겠대... 모델 이름 설정 

# vote_type 설정  : 투표할 때 키포인트를 어떻게 정의하고 예측핮지 결정. 각 타입은 키포인트으ㅢ 구조를 다르게 정의한대... 
if train_cfg['vote_type']=='BB8C': # BB8C : 바운딩박스 코너점 + 중심점 아 그래서 9개 ㅇㅋㅇㅋ
    vote_type=VotingType.BB8C
    vote_num=9
elif train_cfg['vote_type']=='BB8S': # BB8S : 더 작은 바운딩박스코너점 + 중심점 => 9개 맞음
    vote_type=VotingType.BB8S
    vote_num=9
elif train_cfg['vote_type']=='Farthest': # 물체 표면에서 가장 먼 8개의 점. 에... 그때도 말했지만 가장 먼이라는게 뭐... 어디서 멀단건지? 중심점? 아 9개 중심점에서 먼 거 맞나봄 아 아니래 서로 최대한 떨어진 점이래 아 자연스럽게 뭐 돌출된부분이나 코너 선택되겠구나 ㅇㅇ
    vote_type=VotingType.Farthest
    vote_num=9
elif train_cfg['vote_type']=='Farthest4': # 물체 표면에서 가장 먼 4개의 점 +중심점 중심점은 물체의 전반적인 위치를 나타내는데 중요하대 필수정보래~ 
    vote_type=VotingType.Farthest4
    vote_num=5
elif train_cfg['vote_type']=='Farthest12': # 물체 표면에서 가장 먼 12개의 점 ㅇㅇ
    vote_type=VotingType.Farthest12
    vote_num=13
elif train_cfg['vote_type']=='Farthest16': # 물체 표면에서 가장 먼 16개의 점ㅇㅇ
    vote_type=VotingType.Farthest16
    vote_num=17
else:
    assert(train_cfg['vote_type']=='BB8') # 기본값... 8개의 바운딩박스코너점 
    vote_type=VotingType.BB8
    vote_num=8
# 나중에 넷웤에서 키포인트 예측할 때 vote_num 개수만큼 점들을 예측해야하는것임.
# 이 점들을 사용해서 란삭같은알고리즘으로 6DoF를 추정한대 (물체의 회전/위치)

# 평균과 현재 값을 계산하고 저장하는 객체들 생성 ~ 학습 기록용 객체들
seg_loss_rec = AverageMeter() # segmentation loss 기록용 객체 segmentation은 객체있는 픽셀 찾아내는 애. 물체가있는위치=마스크 출력함.(b배치사이즈c클래스개수hw)
ver_loss_rec = AverageMeter() # vertex loss 벌텍스:아 논문에서 본 바로 그 키포인트를 가리키는 방향벡터. 각 픽셀이 키포인트로 향하는 벡터를 예측하잖아? 그걸로 키포인트 위치 추정하고... 아 즉. 키포인트를 가리키는방향을 계산함. 그래서 가림잘림키포인트도 추정 가능하고... 나중에 란삭투표로 최종 키 포인트 결정하고~ : 각키포인트에대해 xy벡터포함 출력. 에 맞나 아 모르겠 
precision_rec = AverageMeter() # Precision 기록 : 예측객체중 실제객체픽셀비율
recall_rec = AverageMeter() # recall 기록  : 실제객체중 예측ㅁ객체픽셀비율
recs=[seg_loss_rec,ver_loss_rec,precision_rec,recall_rec]
recs_names=['scalar/seg','scalar/ver','scalar/precision','scalar/recall']

# 얘넨 걍. 학습/평가 시간 얼마나 걸리는지 관리할라고... 리얼 time을 말하는 것 맞음 
data_time = AverageMeter() # 데이터로드하는데걸린시간기록용객체
batch_time = AverageMeter() # 한 배치 처리하는데걸린시간
recorder = Recorder(True,os.path.join(cfg.REC_DIR,train_cfg['model_name']),
                    os.path.join(cfg.REC_DIR,train_cfg['model_name']+'.log')) # 학습과정과결과를기록하는객체...레코더 

# network_time,voting_time,load_time=[],[],[]

# poses_pr=[]
# poses_gt=[]

class NetWrapper(nn.Module): # 넷웤 래퍼 : 주어진 넷웤에 손실함수 추가한 객체 
    def __init__(self,net):
        super(NetWrapper,self).__init__()
        self.net=net # 주어진 넷웤 
        self.criterion=nn.CrossEntropyLoss(reduce=False) # loss function 정의 

    def forward(self, image, mask, vertex, vertex_weights): 
        # Segmentation, Vertex loss 계산
        seg_pred, vertex_pred = self.net(image) # 넷웤output : segmentation, vertex 예측 
        loss_seg = self.criterion(seg_pred, mask) # 세그멘테이션로스 : 객체있다예측마스크와 ㄹㅇ찐마스크 간의 loss
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0],-1),1) # 배치 평균 계산 
        loss_vertex = smooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False) # l1 loss로 vertex손실도 계산
        precision, recall = compute_precision_recall(seg_pred, mask) # seg랑 mask갖고 precision, recall 계산 (예전에 손으로 해봤지?기억나지?ㅇㅋㅇㅋ)
        return seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall # segmentation/ vertex 예측, segmentation/ vertex 손실, precision, recall 뱉음 


class EvalWrapper(nn.Module): # 평가 래퍼
    def forward(self, seg_pred, vertex_pred, use_argmax=True, use_uncertainty=False): # 두 예측값과 argmax쓸지말지 불확실한지아닌지 input
        vertex_pred=vertex_pred.permute(0,2,3,1) # vertex예측한걸 Bㅐ치사이즈(한번에처리하는이미지사이즈), H,W, V*2(xy니까 2개)(V는 키포인트 개수)로 순서바꿈 
        b,h,w,vn_2=vertex_pred.shape # 각 값
        vertex_pred=vertex_pred.view(b,h,w,vn_2//2,2) # (B, H, W, V, 2) 형태로 변환.
        # 예측 마스크를 argmax로 처리
        if use_argmax: # 여기서 argmax를 처리해서 mask가 이산값을 가져야 란삭 알고리즘을 돌릴 수 있대. 물체있없을 구분할수 있다고함 
            mask=torch.argmax(seg_pred,1)  # 픽셀에서 가장 높은 확률 클래스 인덱스걸 가져옴 
            # seg_pred.size = Bㅐ치사이즈, C클래스개수, H,W임 c를보고 가장 높은 확률가진 클래스를 얻는것같음. mask.size=B,H,W가 됨
        else: # 세그멘테이션결과가 확률분포일때 모든 픽셀의 확률값을 직접 활용해 더 정교한 추정을 수행하고싶을때나 예측분포가 이미명확해서 확률값자체를 활용하는게 나을때 쓴대. 예시는... 모르겠음
            mask=seg_pred # 아니면 그냥 그대로 사용 픽셀별 클래스 확률분포를 그대로 갖고있음 

        # 예측 결과의 불확실성을 고려하거나말고 RANSAC 적용 -> 키포인트 추정 
        if use_uncertainty:
            return ransac_voting_layer_v5(mask,vertex_pred,128,inlier_thresh=0.99,max_num=100) # 한 배치에 있는 모든 이미지를 다 굳 키포인트좌표,각키포인트신뢰도계산한거 모아서 리턴됨
        else:
            return ransac_voting_layer_v3(mask,vertex_pred,128,inlier_thresh=0.99,max_num=100) # 한번에 만들가설수가128개

class MotionEvalWrapper(nn.Module): # 모션 평가 래퍼 
    def forward(self, seg_pred, vertex_pred, use_argmax=True, use_uncertainty=False):
        vertex_pred=vertex_pred.permute(0,2,3,1)
        b,h,w,vn_2=vertex_pred.shape
        vertex_pred=vertex_pred.view(b,h,w,vn_2//2,2) # 각 키포인트별 x, y 벡터로 나눔.
        if use_argmax: # 최대값 쓴다만다 
            mask=torch.argmax(seg_pred,1)
        else:
            mask=seg_pred
        return ransac_motion_voting(mask, vertex_pred) # 란삭을 이용해 모션 추정 

class UncertaintyEvalWrapper(nn.Module): # 불확실성기반 평가 래퍼 
    def forward(self, seg_pred, vertex_pred, use_argmax=True):
        vertex_pred=vertex_pred.permute(0,2,3,1)
        b,h,w,vn_2=vertex_pred.shape
        vertex_pred=vertex_pred.view(b,h,w,vn_2//2,2) # 각 키포인트별 x, y 벡터로 나눔.
        if use_argmax: # 최대값사용안사용 
            mask=torch.argmax(seg_pred,1)
        else:
            mask=seg_pred
        # 란삭으로 키포인트 평균 계산 
        mean=ransac_voting_layer_v3(mask, vertex_pred, 512, inlier_thresh=0.99) # 뭐근데 v5랑 비슷할듯? 불확실성 고려 안 할 뿐이잖 어라 근데 왜이름이 불확실어쩌구
        mean, var=estimate_voting_distribution_with_mean(mask,vertex_pred,mean) #평가투표분포윗평균 에...
        return mean, var # 예측된 키포인트의 평균과 분산 반환.

def train(net, optimizer, dataloader, epoch):
    for rec in recs: rec.reset() # loss 기록 초기화
    data_time.reset() # 초기화
    batch_time.reset() # 초기화 

    train_begin=time.time() # 리얼타임 

    net.train() # 넷웤을 train모드로 전환
    size = len(dataloader) # 데이터로더의 전체 배치 개수
    end=time.time() # 현재 시간 
    for idx, data in enumerate(dataloader): # 데이터 로더에서 배치별로 데이터 가져옴 
        image, mask, vertex, vertex_weights, pose, _ = [d.cuda() for d in data] # 데이터 GPU로 로드한대
        data_time.update(time.time()-end) 

        # 네트워크 추론 및 손실 계산
        seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall = net(image, mask, vertex, vertex_weights) # 넷웤돌리면 나옴
        loss_seg, loss_vertex, precision, recall=[torch.mean(val) for val in (loss_seg, loss_vertex, precision, recall)] # 한 배치의? 평균 계산함 
        loss = loss_seg + loss_vertex * train_cfg['vertex_loss_ratio'] # 총 손실 계산 
        vals=(loss_seg,loss_vertex,precision,recall) # 튜플로 묶어서
        for rec,val in zip(recs,vals): rec.update(val) # 손실 및 메트릭 저장~ 

        optimizer.zero_grad() # 그래디언트 초기화 
        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트

        batch_time.update(time.time()-end)
        end=time.time()

        # 손실 및 메트릭 원하는 스텝마다 기록 
        if idx % train_cfg['loss_rec_step'] == 0:
            step = epoch * size + idx # 현재 스텝
            losses_batch=OrderedDict() # 삽입순서기록용딕셔너리?라는데? 손실 기록을 위해 생성햇대
            for name,rec in zip(recs_names,recs): losses_batch['train/'+name]=rec.avg # 각 메트릭 평균추가
            recorder.rec_loss_batch(losses_batch,step,epoch) # 레코더에 저장
            for rec in recs: rec.reset() # 기록 객체 초기화

            data_time.reset()
            batch_time.reset()

        # 이미지와 예측결과 원하는스텝마다 기록 
        if idx % train_cfg['img_rec_step'] == 0:
            batch_size = image.shape[0] # 배치사이즈 얼마더라 
            nrow = 5 if batch_size > 5 else batch_size # 이미지 그리드 행 개수 5거나 배치사이즈거나..
            recorder.rec_segmentation(F.softmax(seg_pred, dim=1), num_classes=2, nrow=nrow, step=step, name='train/image/seg') # 세그멘테이션결과기록
            recorder.rec_vertex(vertex_pred, vertex_weights, nrow=4, step=step, name='train/image/ver') # 벌텍스결과기록 
        
        # 한 배치 완 

    print('epoch {} training cost {} s'.format(epoch,time.time()-train_begin)) # 한 에폭 돌았음 


def val(net, dataloader, epoch, val_prefix='val', use_camera_intrinsic=False, use_motion=False): # 검증 시작
    for rec in recs: rec.reset() # 성능지표들 기록하는 객체 초기화 

    test_begin = time.time() # 시간 이제부터 걍 안 써야겠다 
    evaluator = Evaluator() # 평가객체. 생성 

    # 평가용 넷웤 : 모션이면 모션 아니면 일반평가래퍼 를 쿠다에 올림. 데이터패러렐 : 여러 GPU에서 병렬로 모델실행하도록 함 단일GPU여도 노상관
    # 평가래퍼 : 란삭기반하여 세그멘테이션/벌텍스 예측값으로 물체의 키포인트 추정ㅎㅁ
    # 모션평가래퍼 : 세그멘테이션/벌텍스 예측값으로 물체의 움직임 추정함 
    eval_net=DataParallel(EvalWrapper().cuda()) if not use_motion else DataParallel(MotionEvalWrapper().cuda()) 
    uncertain_eval_net=DataParallel(UncertaintyEvalWrapper().cuda()) # 불확실성평가넷웤
    net.eval() # 평가모드온 
    for idx, data in enumerate(dataloader): # 배치별로 반복. 다돌면 한에폭완 
        if use_camera_intrinsic: # 카메라본질적? 카메라 고유 파라미터를 사용하는 경우 Ks : 데이터와 카메라 고유 행렬(Ks)를 gpu에 올리겠다 
            image, mask, vertex, vertex_weights, pose, corner_target, Ks = [d.cuda() for d in data]
        else:
            image, mask, vertex, vertex_weights, pose, corner_target = [d.cuda() for d in data]

        with torch.no_grad(): # 평가모드니까 그래디언트 계산 x
            seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall = net(image, mask, vertex, vertex_weights) # 넷웤돌리면 나오는 값들

            loss_seg, loss_vertex, precision, recall=[torch.mean(val) for val in (loss_seg, loss_vertex, precision, recall)] #을 평균냄 

            # 평가에폭 트루, 원하는간격, 테스트모델일때 등등 에폭 상황 딱 맞으면 자세 추정 수행 
            if (train_cfg['eval_epoch']
                and epoch%train_cfg['eval_inter']==0
                and epoch>=train_cfg['eval_epoch_begin']) or args.test_model:
                if args.use_uncertainty_pnp: # 불확실성고려한 자세추정을 사용하겠다 
                    mean,cov_inv=uncertain_eval_net(seg_pred,vertex_pred) # 하면 예측된키포인트의 평균과분산반환됨 (한배치내의 )평균 및 공분산 역행렬 계산
                    mean=mean.cpu().numpy() # cpu로 보냄 다시 
                    cov_inv=cov_inv.cpu().numpy()
                else: # 고려안하겠다 
                    corner_pred=eval_net(seg_pred,vertex_pred).cpu().detach().numpy() # 키포인트좌표/신뢰도 예측모음집 (한배치내의)
                pose=pose.cpu().numpy() # 자세 데이터 cpu로 보냄 

                b=pose.shape[0] # 배치크기 
                pose_preds=[]
                for bi in range(b): # 각 이미지마다 반복 
                    intri_type='use_intrinsic' if use_camera_intrinsic else 'linemod' # 카메라 파라미터 사용함?
                    K=Ks[bi].cpu().numpy() if use_camera_intrinsic else None # 카메라 행렬... 
                    if args.use_uncertainty_pnp: # 불확실성 기반 평가... 
                        pose_preds.append(evaluator.evaluate_uncertainty(mean[bi],cov_inv[bi],pose[bi],args.linemod_cls,
                                                                         intri_type,vote_type,intri_matrix=K))
                        # evaluate_uncertainty : 불확실성기반자세추정... 예측된키포인트좌표,공분산,실제자세 등 input/ -> 불확실성고려 pnp알고리즘써서 자세예측계산 평가지표업뎃
                    else: # 그냥 일반적인 자세 평가 
                        pose_preds.append(evaluator.evaluate(corner_pred[bi],pose[bi],args.linemod_cls,intri_type,
                                                             vote_type,intri_matrix=K))
                        # evaluate : pnp문제 기반으로 pose 예측하고 평가지표 업뎃~ 


                if args.save_inter_result: # 중간결과 저장하는 경우 예측마스크,gt마스크, 입력이미지,자세데이터 저장 <- 얜 특히 피클로 저장 
                    mask_pr = torch.argmax(seg_pred, 1).cpu().detach().numpy() # 예측마스크
                    mask_gt = mask.cpu().detach().numpy() # gt
                    # assume batch size = 1
                    imsave(os.path.join(args.save_inter_dir, '{}_mask_pr.png'.format(idx)), mask_pr[0])
                    imsave(os.path.join(args.save_inter_dir, '{}_mask_gt.png'.format(idx)), mask_gt[0])
                    imsave(os.path.join(args.save_inter_dir, '{}_rgb.png'.format(idx)),
                           imagenet_to_uint8(image.cpu().detach().numpy()[0]))
                    save_pickle([pose_preds[0],pose[0]],os.path.join(args.save_inter_dir, '{}_pose.pkl'.format(idx)))

            vals=[loss_seg,loss_vertex,precision,recall] # 지금 배치의 성능 지표
            for rec,val in zip(recs,vals): rec.update(val) # 기록객체에 업뎃 
        # 한 에폭 완 

    with torch.no_grad(): # 그래디언트 계싼 x. 최종 기록하기 
        batch_size = image.shape[0]
        nrow = 5 if batch_size > 5 else batch_size # 이미지그리드행개수설정 
        # 레코더에 저장 세그멘테이션과벌텍스결과를 
        recorder.rec_segmentation(F.softmax(seg_pred, dim=1), num_classes=2, nrow=nrow,
                                  step=epoch, name='{}/image/seg'.format(val_prefix))
        recorder.rec_vertex(vertex_pred, vertex_weights, nrow=4, step=epoch, name='{}/image/ver'.format(val_prefix))

    losses_batch=OrderedDict() # 삽입순서기록용딕셔너리?라는데? 손실 기록을 위한 딕셔너리 생성 
    for name, rec in zip(recs_names, recs): losses_batch['{}/'.format(val_prefix) + name] = rec.avg # 각성능지표 평균추가
    # 여차저차 상황되면 
    if (train_cfg['eval_epoch']
        and epoch%train_cfg['eval_inter']==0
        and epoch>=train_cfg['eval_epoch_begin']) or args.test_model:
        # 자세 평가 결과 계산해서 추가 
        proj_err,add,cm=evaluator.average_precision(False)
        losses_batch['{}/scalar/projection_error'.format(val_prefix)]=proj_err
        losses_batch['{}/scalar/add'.format(val_prefix)]=add
        losses_batch['{}/scalar/cm'.format(val_prefix)]=cm
    recorder.rec_loss_batch(losses_batch, epoch, epoch, val_prefix) # 손실 기록~~ 
    for rec in recs: rec.reset() # 기록객체리셋 

    print('epoch {} {} cost {} s'.format(epoch,val_prefix,time.time()-test_begin)) # 평가에 걸린 시간

def train_net():
    # 네트워크 설정
    net=Resnet18_8s(ver_dim=vote_num*2, seg_dim=2) # resnet18 백본 사용 (vertex 예측 segmentation 예측 뱉음)
    net=NetWrapper(net) # 넷웤 래퍼 씌움
    net=DataParallel(net).cuda() # 넷웤을 GPU 병렬 처리 모드로 전환하는 거래 걍 토치에 있는 기본빵 함수 같으니까 목적만 알고 일단 넘김 

    optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr']) # Adam 최적화기 선택
    model_dir=os.path.join(cfg.MODEL_DIR,train_cfg['model_name']) # 모델 저장 경로는 이곳
    motion_model=train_cfg['motion_model'] # motion model True/False 쓸지말지 뭔데이거... 뭐지? 
    print('motion state {}'.format(motion_model)) # 쓸지말지 써둘게~ 

    # TEST 
    if args.test_model:
        torch.manual_seed(0) # 랜덤시드고정 
        begin_epoch=load_model(net.module.net, optimizer, model_dir, args.load_epoch) # 모델...로드 아 그 에폭부터 시작으로 걍줘서 val때처럼하는데 gt계산은안하나보다 

        if args.normal: # 일반적인 LineMOD 데이터셋일때 
            print('testing normal linemod ...')
            image_db = LineModImageDB(args.linemod_cls,has_render_set=False,
                                      has_fuse_set=False) # 데이터셋 객체 초기화 
            test_db = image_db.test_real_set+image_db.val_real_set # 테스트셋+검증셋 합침 
            test_set = LineModDatasetRealAug(test_db, cfg.LINEMOD, vote_type, augment=False, use_motion=motion_model) # 증강거치고
            test_sampler = SequentialSampler(test_set) # 부분적으로 섞고? 아아니다 데이터 순차 샘플링.
            test_batch_sampler = ImageSizeBatchSampler(test_sampler, train_cfg['test_batch_size'], False) # 배치샘플러
            test_loader = DataLoader(test_set, batch_sampler=test_batch_sampler, num_workers=0) # 데이터로더 생성완 
            prefix='test' if args.use_test_set else 'val' # 출력 경로를 테스트/val에 맞춤
            val(net, test_loader, begin_epoch, prefix, use_motion=motion_model) # val함수 돌림! 에. 근데 val 안에 실제포즈나 그거 계산이 있는데 뭔소리지?

        if args.occluded and args.linemod_cls in cfg.occ_linemod_cls_names: # 가림데이터셋이고 그 객체가 가림 객체에 있을때 
            print('testing occluded linemod ...')
            occ_image_db = OcclusionLineModImageDB(args.linemod_cls)
            occ_test_db = occ_image_db.test_real_set
            occ_test_set = LineModDatasetRealAug(occ_test_db, cfg.OCCLUSION_LINEMOD, vote_type,
                                                 augment=False, use_motion=motion_model)
            occ_test_sampler = SequentialSampler(occ_test_set)
            occ_test_batch_sampler = ImageSizeBatchSampler(occ_test_sampler, train_cfg['test_batch_size'], False)
            occ_test_loader = DataLoader(occ_test_set, batch_sampler=occ_test_batch_sampler, num_workers=0)
            prefix='occ_test' if args.use_test_set else 'occ_val'
            val(net, occ_test_loader, begin_epoch, prefix, use_motion=motion_model)

        if args.truncated: # 잘림데이터셋일때 
            print('testing truncated linemod ...')
            trun_image_db = TruncatedLineModImageDB(args.linemod_cls)
            print(len(trun_image_db.set))
            trun_image_set = LineModDatasetRealAug(trun_image_db.set, cfg.LINEMOD, vote_type, augment=False,
                                                   use_intrinsic=True, use_motion=motion_model)
            trun_test_sampler = SequentialSampler(trun_image_set)
            trun_test_batch_sampler = ImageSizeBatchSampler(trun_test_sampler, train_cfg['test_batch_size'], False)
            trun_test_loader = DataLoader(trun_image_set, batch_sampler=trun_test_batch_sampler, num_workers=0)
            prefix='trun_test'
            val(net, trun_test_loader, begin_epoch, prefix, True, use_motion=motion_model)

    # TRAIN
    else:
        begin_epoch=0

        # RESUME의 경우, 시작 epoch를 맞춰줌. 이전 학습 상태 로드함 
        if train_cfg['resume']: 
            begin_epoch=load_model(net.module.net, optimizer, model_dir)

        # train 데이터셋 구성. (뭔 클래스 쓸지, fuse(여러 객체 다 있는 데이터셋으로 학습 시시시작하겠냐고)랑 렌더셋(3d->2d로바꿔서가상데이터추가해둔데이터셋))
        image_db = LineModImageDB(args.linemod_cls,
                                  has_fuse_set=train_cfg['use_fuse'],
                                  has_render_set=True)

        train_db=[]
        train_db+=image_db.render_set # 렌더링된 데이터 추가
        if train_cfg['use_real_train']: # 실제 학습 데이터 추가
            train_db+=image_db.train_real_set
        if train_cfg['use_fuse']: # 혼합되 ㄴ데이터 추가 
            train_db+=image_db.fuse_set

        train_set = LineModDatasetRealAug(train_db, cfg.LINEMOD, vote_type, augment=True, cfg=train_cfg['aug_cfg'], use_motion=motion_model) # 증강 거친 데이터셋
        train_sampler = RandomSampler(train_set) # 랜덤 샘플러 => 학습 데이터 랜덤하게 순서없이 제공~ 
        train_batch_sampler = ImageSizeBatchSampler(train_sampler, train_cfg['train_batch_size'], False, cfg=train_cfg['aug_cfg']) # 이미지사이즈배치샘플러 : 데이터크기에따라 배치를 삭삭 나눠서 구성함 다양한크기의데이터를 유연하게 배치할라고 ~ 비슷한사이즈묶어둠 
        train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=12) # train 데이터 로더 구성 완

        val_db=image_db.val_real_set
        val_set = LineModDatasetRealAug(val_db, cfg.LINEMOD, vote_type, augment=False, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
        val_sampler = SequentialSampler(val_set)
        val_batch_sampler = ImageSizeBatchSampler(val_sampler, train_cfg['test_batch_size'], False, cfg=train_cfg['aug_cfg'])
        val_loader = DataLoader(val_set, batch_sampler=val_batch_sampler, num_workers=12) # val 데이터 로더 구성 완 

        if args.linemod_cls in cfg.occ_linemod_cls_names: # 가려짐데이터에있으면 하기 : 
            occ_image_db=OcclusionLineModImageDB(args.linemod_cls)
            occ_val_db=occ_image_db.test_real_set[:len(occ_image_db.test_real_set)//2]
            occ_val_set = LineModDatasetRealAug(occ_val_db, cfg.OCCLUSION_LINEMOD, vote_type, augment=False, cfg=train_cfg['aug_cfg'], use_motion=motion_model)
            occ_val_sampler = SequentialSampler(occ_val_set)
            occ_val_batch_sampler = ImageSizeBatchSampler(occ_val_sampler, train_cfg['test_batch_size'], False, cfg=train_cfg['aug_cfg'])
            occ_val_loader = DataLoader(occ_val_set, batch_sampler=occ_val_batch_sampler, num_workers=12) # 가림데이터 val 로더구성완 

        # 학습 루프 시시시작 
        for epoch in range(begin_epoch, train_cfg['epoch_num']): # 매 에폭마다 반복 
            adjust_learning_rate(optimizer,epoch,train_cfg['lr_decay_rate'],train_cfg['lr_decay_epoch']) # 학습률 조정
            train(net, optimizer, train_loader, epoch) # 학습 실행
            val(net, val_loader, epoch,use_motion=motion_model) # 검증 실행
            if args.linemod_cls in cfg.occ_linemod_cls_names: # 가려짐데이터에 있으면
                val(net, occ_val_loader, epoch, 'occ_val',use_motion=motion_model) # 가려짐데이터도 검증 실행
            save_model(net.module.net, optimizer, epoch, model_dir) # 체크포인트 저장 

# def save_dataset(dataset,prefix=''):
#     with open('assets/{}{}.txt'.format(prefix,args.linemod_cls),'w') as f:
#         for data in dataset: f.write(data['rgb_pth']+'\n')
#
# def save_poses_dataset(prefix=''):
#     print(np.asarray(poses_pr).shape)
#     np.save('assets/{}{}_pr.npy'.format(prefix,args.linemod_cls),np.asarray(poses_pr))
#     np.save('assets/{}{}_gt.npy'.format(prefix,args.linemod_cls),np.asarray(poses_gt))

if __name__ == "__main__":
    train_net()
    # save_poses_dataset('trun_')
