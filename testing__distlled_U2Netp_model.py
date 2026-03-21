


from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.optim as optim
import time
import tqdm

from PIL import Image
import cv2


from loss.diceloss import *
from CTS_dataset_updated_for_BASNet import mydataloader
from models.ResUnet import *
from models.WideResNet import *
from models.Attention_UNet import *
from models.anamnet import *
from models.Attention_UNet_by_modifying_existing_UNet import *
from hausdorff_distance_mask import *
#%%
# loading the model.

def compute_loss(y_hat, y):
    return nn.BCELoss()(y_hat, y)

diceloss=DiceLoss()
#%%

def binary_entropy(prediction_map):
    """
    Calculate the binary entropy of a binary prediction map.

    Parameters:
    prediction_map (torch.Tensor): Binary prediction map of shape (batch_size, 1, height, width).

    Returns:
    torch.Tensor: Binary entropy value.
    """
    # Ensure input tensor is on CPU and in float format
    prediction_map = prediction_map.cpu().float()
    
    # Calculate entropy
    entropy_value = - (prediction_map * torch.log2(prediction_map + 1e-20))
    
    return entropy_value
#%%
def write_mask_on_image(input_image,input_mask,expected_color=(255,0,0)):
    
    temp=input_image
    # make a edges/boundary for the mask
    edges = cv2.Canny(input_mask,0,255)
    
    
    # replacing the mask boundary in the given image
    temp[:,:,0][edges==255]=expected_color[0]
    temp[:,:,1][edges==255]=expected_color[1]
    temp[:,:,2][edges==255]=expected_color[2]
    
    #plt.imshow(img1)
    
    #cv2.imshow('img1',img1)
    
    cv2.imwrite("masked_image.png", cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))

    return cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

def write_mask_on_image(input_image,input_mask,expected_color=(255,0,0)):
    
    temp=input_image
    # make a edges/boundary for the mask
    edges = cv2.Canny(input_mask,0,255)
    
    
    # replacing the mask boundary in the given image
    temp[:,:,0][edges==255]=expected_color[0]
    temp[:,:,1][edges==255]=expected_color[1]
    temp[:,:,2][edges==255]=expected_color[2]
    
    #plt.imshow(img1)
    
    #cv2.imshow('img1',img1)
    
    cv2.imwrite("masked_image.png", cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))

    return cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)


#%%
#plt.imshow(write_mask_on_image(input_image, input_mask,expected_color=(0,255,0)))

def cross_section_area(mask):
    temp=mask[mask==255]
    #print(temp.shape)


def another_metrics(ground_truth_mask,generated_mask):
    #print(ground_truth_mask.shape,generated_mask.shape)
    #print(np.unique(ground_truth_mask),np.unique(generated_mask))
    
    TP=np.count_nonzero(ground_truth_mask[generated_mask==255]==255)
    #print(TP)
    
    temp_ground=np.copy(ground_truth_mask)
    temp_ground[generated_mask==255]=0
    FN=np.count_nonzero(temp_ground)

    temp_generated=np.copy(generated_mask)
    temp_generated[ground_truth_mask==255]=0
    FP=np.count_nonzero(temp_generated)
    
    TN=464*352-(TP+FP+FN)
    
    #print(TP,TN,FP,FN)
    
    if(TP!=0):
        accuray=(TP+TN)/(TP+FP+FN+TN)
        precision=(TP)/(TP+FP)
        recall=(TP)/(TP+FN)
        F1score=(2*TP)/(2*TP+FP+FN)
        Threatscore=(TP)/(TP+FN+FP)
        correction_effort=(FP+FN)/(TP+FN)

    else:
        precision=0
        recall=0
        accuray=(TP+TN)/(TP+FP+FN+TN)
        F1score=0
        Threatscore=0
        correction_effort=(FP+FN)/(TP+FN)
        
        

    #print([accuray,precision,recall])
    
    return accuray,precision,recall,F1score,Threatscore,correction_effort
#%%


def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")

#%%
gpu_id=0
model_name='u2netp'



#%%


if(model_name=='u2netp'):
    from models.u2_net_model import U2NETP

    model = U2NETP(3,1)

    epoch_no=1
    experiment_name='Mar_21_01_03_PM_U2NET_to_U2NETP_distillation_100_percent_data_bce_ssim_iou_multiscale'
    PATH_for_experiments="savedModels/distillation_experiments///"


    PATH_for_experiment=PATH_for_experiments+"/"+experiment_name
    model_path=PATH_for_experiment+'/u2netp_distilled_epoch_'+str(epoch_no)+'.pth'



#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


state_dict = torch.load(model_path, map_location=device)


# Handle DataParallel
if list(state_dict.keys())[0].startswith("module."):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    state_dict = new_state_dict

model.load_state_dict(state_dict)
model.to(device)
model=model.float()
model.eval()


#%%

torch.save(model.state_dict(), 'model.pth')

#%%


# LOADING THE TEST DATA.

data_path='../data_making/aster_updated_data_nov_09_2022_with_flip/'

t_loader = mydataloader(data_path, 
                        '../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/patients_list_90_99.csv', 
                        '../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/data_with_v_h_90_99.csv')


# t_loader =  mydataloader(data_path, '../data_making/aster_updated_data_nov_09_2022_with_flip//csv_files/patients_list_100.csv',
#                           'csv_files/data_test_1_90_99_99.csv')

test_loader = DataLoader(t_loader, batch_size = 1, shuffle=False, num_workers=1)
no_test_batches=len(test_loader)

#%%
# saving the output files.
from datetime import datetime

#making directory for sving the results
print ('*******************************************************')
directory=PATH_for_experiment+'/results_'+str(epoch_no)+"/"
print('Model will be saved to  :', directory)

try:
    os.makedirs(directory)
    os.makedirs(directory+'/both_mask_on_image')

except:
    print("results are existed...")
#%%




# IT IS FOR WRITING THE MASKS
def write_masks(output_cpu,directory,iteration,is_actual):
        img=output_cpu
        img=(img).astype(np.uint8)
        img=Image.fromarray(img)
        if(is_actual):
            img.save(directory+str(iteration)+'_mask_actual.tif' )
            #print('actual mask writing...')
        else:
            img.save(directory+str(iteration)+'_mask_generated.tif' )
            #print('generated mask writing...')

# IT IS FOR WRITING THE IMAGES.

def write_images(output_cpu,directory,iteration,is_actual):
        img=output_cpu
        img=(img).astype(np.uint8)
        img=Image.fromarray(img)
        if(is_actual):
            img.save(directory+str(iteration)+'_mask_actual.jpg' )
            #print('actual mask writing...')
        else:
            img.save(directory+str(iteration)+'_mask_generated.jpg' )
            #print('generated mask writing...')


def write_masks_appened(output_cpu,directory,iteration):
        img=output_cpu[i]
        img=(img).astype(np.uint8)
        img=Image.fromarray(img)
        img.save(directory+str(i)+'_mask_on_image.tif' )
        
        
#%%

def perturb_weights(model, std=0.00):
    for param in model.parameters():
        noise = param*std
        param.data.add_(noise)

# perturb_weights(model, std=0.06)

#%%
#%%
calibration=0.004328254
testing_loss=0

accuray_list=[]
actual_cross_section_area_list=[]
computed_cross_section_area_list=[]
hausdorff_distance_list=[]
patient_video_needed=91



with torch.no_grad():
    for i, data in tqdm.tqdm(enumerate(test_loader)): 
            raw_image_file,mask_file,patient_id,image_no=data
            image_file=raw_image_file/255
            image_file=image_file.float()
            image_file=image_file.cuda(gpu_id)

            mask_file=mask_file.float()
            mask_file=mask_file.cuda(gpu_id)


            # tic()
            output, d1, d2, d3, d4, d5, d6=model(image_file)


            median_nerve_pixels=output[mask_file==255]

            # for dealing edges
            output_for_dealing_edges=output.clone()
            


            output[output>0.5]=1
            output[output<0.5]=0

            loss = diceloss(output,mask_file/255)
            testing_loss += loss.item()
            #print(1-loss.item())

            image_file=image_file.squeeze().permute(1,2,0)

            # image_file_actual[:,:,0]=image_file[:,:,0]*mask_file
            # image_file_actual[:,:,1]=image_file[:,:,1]*mask_file
            # image_file_actual[:,:,2]=image_file[:,:,2]*mask_file

            image_file_generated=torch.zeros(size=(448,320,3))
            image_file_generated[:,:,0]=image_file[:,:,0]*output
            image_file_generated[:,:,1]=image_file[:,:,1]*output
            image_file_generated[:,:,2]=image_file[:,:,2]*output
    
            # print('image_file.shape:',image_file.shape)
            # print('image_file_permuted.shape:',image_file_permuted.shape)
            # print('mask_file.shape:',mask_file.shape)
            
            # image_file_actual=image_file_actual.cpu().detach().numpy()
            # image_file_generated=image_file_generated.cpu().detach().numpy()
    
            # write_images(image_file_permuted*255, directory, i, True)
            # write_images(image_file_generated*255, directory, i, False)
            
            # For dealing about median nerve pixels....

            
            # print('\noutput_cpu',output_cpu.shape)
            # print('mask_file',mask_file.shape)
            image_file_cpu=image_file.cpu().squeeze().detach().numpy()*255
            image_file_cpu=(image_file_cpu).astype(np.uint8)
            #print(image_file_cpu.dtype)            
            # img=Image.fromarray(image_file_cpu)
            #img.save(directory+'_mask_'+str(patient_id.item())+'_'+str(image_no.item())+'_image.jpg' )
    
            
            # writing actual mask 

            #write_masks(mask_file, directory, str(patient_id.item())+'_'+str(image_no.item()), True)
    
            # writing the generated mask        
            output_cpu=output.cpu().squeeze().detach().numpy()
            output_cpu=output_cpu*255
            output_cpu=output_cpu.astype(np.uint8)
            
            mask_file_clone_for_edges=mask_file.clone()
            mask_file=mask_file.squeeze()
            mask_file=mask_file/255

            mask_file=mask_file.cpu().squeeze().detach().numpy()
            mask_file=mask_file*255
            mask_file=mask_file.astype(np.uint8)
            



            temp=np.copy(image_file_cpu)

            contours_gt, _ = cv2.findContours(mask_file, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_on_image = cv2.drawContours(image_file_cpu.copy(), contours_gt, -1, (0, 255, 0), 2)
        
            # drawing predicted mask contour on given image
            contours_pred, _ = cv2.findContours(output_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_on_image = cv2.drawContours(mask_on_image.copy(), contours_pred, -1, (0, 0, 255), 2)




            accuray,precision,recall,F1score,Threatscore,correction_effort=another_metrics(mask_file, output_cpu)
            
            
            
            cs_area_from_prediction=np.count_nonzero(output_cpu==255)*calibration
            cs_area_from_actual=np.count_nonzero(mask_file==255)*calibration
            #print('CS_AREA',cs_area_from_actual,cs_area_from_prediction)
            actual_cross_section_area_list.append(cs_area_from_actual)
            computed_cross_section_area_list.append(cs_area_from_prediction)
            # cv2.putText(img=mask_on_image, text=str(str(round(cs_area_from_prediction,4))), org=(275, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0,0,255),thickness=1)
            # cv2.putText(img=mask_on_image, text=str(str(round(cs_area_from_actual,4))), org=(275, 70), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0,255,0),thickness=1)
            #cv2.imwrite(directory+'/both_mask_on_image/'+str(patient_id.item())+'_'+str(image_no.item())+'_both_mask_on_image_'+str(round(F1score,4))+'_'+str(round(cs_area_from_actual,4))+'_'+str(round(cs_area_from_prediction,4))+'.png',mask_on_image)

            accuray_list.append([patient_id.item(),image_no.item(),1-loss.item(),accuray,precision,recall,F1score,Threatscore,correction_effort])
            hausdorff_distance_list.append(hausdorff_distance_mask(mask_file/255,output_cpu/255))
            # print('\n',patient_id.item(),image_no.item(),hausdorff_distance_mask(mask_file/255,output_cpu/255))
            
print('testing is completed.')
    

print('dice match:',1-testing_loss/no_test_batches)


print('dice match:',1-testing_loss/no_test_batches)
#%%

try:
    csv_results_path=PATH_for_experiment+'/csv_results/'
    os.makedirs(csv_results_path)

except:
    print("results are existed...")
#%%
temp=hausdorff_distance_list.copy()
if(np.inf in temp):
    # temp.remove(np.inf)
    temp = [x for x in temp if not np.isinf(x)]

temp=np.array(temp)
average_hus_distance=np.mean(temp)
std_hus_distance=np.std(temp)
print('average hausdroff distance',average_hus_distance,std_hus_distance)
accuray_list=np.array(accuray_list)
hausdorff_distance_list=np.array(hausdorff_distance_list)
#%%
#%%

import pandas as pd
avg_dice_score=accuray_list[:,2].mean()
accuray=accuray_list[:,3].mean()
precision=accuray_list[:,4].mean()
recall=accuray_list[:,5].mean()
F1score=accuray_list[:,6].mean()
Threatscore=accuray_list[:,7].mean()
correction_effort=accuray_list[:,8].mean()

#%%
dic={'patient_id':accuray_list[:,0],'image_no':accuray_list[:,1],'individual':accuray_list[:,2],'accuracy':accuray_list[:,3],'precision':accuray_list[:,4],'recall':accuray_list[:,5],'F1score':accuray_list[:,6],'Threatscore':accuray_list[:,7],'correction_effort':accuray_list[:,8],'cs_atual':actual_cross_section_area_list,'cs_computed':computed_cross_section_area_list,'hd':hausdorff_distance_list}
df = pd.DataFrame.from_dict(dic) 
df.to_csv (r''+csv_results_path+str(epoch_no)+'_'+str(round(avg_dice_score,4))+'_'+str(round(accuray,4))+'_'+str(round(precision,4))+'_'+str(round(recall,4))+"_"+str(round(average_hus_distance,4))+'.csv', index = False, header=True)

#%%
print(avg_dice_score,accuray,precision,recall,F1score,Threatscore,correction_effort)

print('#'*50)

print('avg_dice_score:',avg_dice_score)
print('precision:',precision)
print('recall:',recall)
print('hausdorff_distance:',average_hus_distance)

print('#'*100)


