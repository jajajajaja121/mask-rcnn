import torch
def cal_iou(result,target):
    result=result.repeat(target.shape[0])
    target=target.repeat(result.shape[0]).permute(1,0,2)
    x_min=max(result[:,:,0],target[:,:,0])
    y_min=max(result[:,:,1],target[:,:,1])
    x_max=min(result[:,:,2],target[:,:,2])
    y_max=min(result[:,:,3],target[:,:,3])
    area=torch.clamp((y_max-y_min),0)*torch.clamp((x_max-x_min),0)
    area_result=(result[:,:,2]-result[:,:,0])*(result[:,:,3]-result[:,:,1])
    area_target=(target[:,:,2]-target[:,:,0])*(target[:,:,3]-target[:,:,1])
    union=(area_result+area_target)-area
    iou=area/union
    return iou
