#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import cv2
from pylab import text
import scipy
import scipy.integrate as integrate
from scipy.stats import gaussian_kde

#put each scenarios's characteristic into a dictionary
Scene={'S1':{},'S2':{},'S3':{},'S4':{},'S5':{},'S6':{},'S7':{},'S8':{},'S9':{}}
for i in Scene:
    Scene[i]={'P_avg_speed':-1,'P_avg_acc:':-1, 'R_avg_speed':-1,'avg_minNN_dist':-1,'P_nearR_path_eff':-1}
    
    
class  Pedestrain:
     def __init__(self):
            self.traj_x=[]
            self.traj_y=[]
            self.vel_x=[0]
            self.vel_y=[0]
            self.acc_x=[0]
            self.acc_y=[0]
            self.jerk_x=[0]
            self.jerk_y=[0]
            
            self.avg_acc=[]
            
            self.ID=-1
            self.frame=[]

            self.occ_pos=[]
            self.occ_duration=[]
            self.occ_Tcell_idx=[]
            self.cell_duration=[]

            self.occ_stayx=[]
            self.occ_stayy=[]
            self.cell_stayx=[]
            self.cell_stayy=[]

            self.simple_speed=[] #only in y dir
            self.nearestP=[] #nearest people's ID
            self.nearestD=[] #nearest people's distance

            self.bottleneck_y=[]

            self.enter_idx=-1
            self.exit_idx=-1
            
            self.eval_pass_order=-2  #should be 0,-1,1
            
            self.avg_speed=0




scn_nbr_list=[3]  #7 not included here
run_nbr_list=[1]


main_dir = '/home/akaimo/CHI2020/new_cut_video'
bottleneck_duration_file= open('bottleneck_duration.txt','w+')
pass_order_file=open("pass_order.txt","w+")
close_segments_file=open("close_segments_file.txt","w+")

avg_scn_simple_speed_bn=[] #avg speed for each scene
avg_scn_simple_acc_bn=[]
fps=12.5
dt=1/fps

for scn_nbr in scn_nbr_list:
    #define robot type
    if scn_nbr == 1 or scn_nbr == 2:
        robot_type='no robot'
    elif scn_nbr==3 or scn_nbr==4 or scn_nbr==5 or scn_nbr==6:
        robot_type='wheelchair'
    elif scn_nbr==8 or scn_nbr==9:
        robot_type='Pepper'
    else:
        robot_type='error'
        
    scn_simple_speed_bn=[] #avg run speed for each scene
    scn_simple_acc_bn=[]
    
    people_count=0
    exceed_robot=0
    give_way=0
    same=0
    allp_scn_sim_speed_bn=[]
    scn_all_efficiency=[]
    robot_speed_bn=[]
    avg_minNN_dist=[]
    
    allp_scn_simple_acc_bn=[]
    if scn_nbr == 9:
        run_nbr_list=[1,2,3]
    
    for run_nbr in run_nbr_list:
        print(scn_nbr,',',run_nbr)
        robot_ID=-1

        v=cv2.VideoCapture(main_dir + '/S%d/S%d_run%d0001-100000-undistort.mp4' %(scn_nbr, scn_nbr, run_nbr))
        v.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
        d_duration=v.get(cv2.CAP_PROP_POS_MSEC)/1000.0 #unit is s
        #print ((v.get(cv2.CAP_PROP_POS_FRAMES)))
       
        dir='new_cut_video/S%d/run%d/'%(scn_nbr, run_nbr)
        head='Extended_traj.txt'#'S%d_run%d-heads.txt'%(scn_nbr, run_nbr)
        new_txt='S%d_run%d-new.txt'%(scn_nbr, run_nbr)

        traj_list=[]
        all_list=[]
        pedestrians=[] 

        #avg_speed vs time
        avg_speed=[]
        
        #this just read number of all ped+robot & robot_ID
        with open(dir+new_txt) as f:
            for line in f.readlines():
                all_list.append(line.split(' '))
                
        p_num=int(all_list[1][3]) #no. of pedestrians+robot
        robot_ID=int(all_list[2][4]) 

        #this read extended trajectories(all trajectories have same time duration)
        with open(dir+head) as f:
            for line in f.readlines():
                traj_list.append(line.split(' '))
                
        #read all pedestrians & robot trajetcories into Pedestrians class
        for num in range(1,p_num+1):
            initialized=False
            for row in range(1,len(traj_list)):
                if (int(traj_list[row][0])==num):
                    if initialized==False:
                        p=Pedestrain() 
                        initialized=True

                    p.ID=(int(traj_list[row][0]))
                    p.frame.append(int(traj_list[row][1]))
                    p.traj_x.append(float(traj_list[row][2]))
                    p.traj_y.append(float(traj_list[row][3]))
            if initialized:         
                pedestrians.append(p)
                
      
        #calculate instaneous vel for each pedestrians
        for p in pedestrians:
            for j in range(1,len(p.traj_x)):
                p.vel_x.append((p.traj_x[j]-p.traj_x[j-1])/dt)
                p.vel_y.append((p.traj_y[j]-p.traj_y[j-1])/dt)

        #calculate instant acc for each pedestrian:
        for p in pedestrians:
            for i in range(1,len(p.vel_x)):
                p.acc_x.append((p.vel_x[i]-p.vel_x[i-1])/dt)
                p.acc_y.append((p.vel_y[i]-p.vel_y[i-1])/dt)
                
        #calculate instant jerk
        for p in pedestrians:
            for i in range(1,len(p.acc_x)):
                p.jerk_x.append((p.acc_x[i]-p.acc_x[i-1])/dt)
                p.jerk_y.append((p.acc_y[i]-p.acc_y[i-1])/dt)
        
        
        #plot most simple diagrams--for each run, each pedestrians's trajectory/inst velocity/ inst acc
        #trajetcory
        plt.figure()
        for p in pedestrians:
            plt.plot(p.traj_x,p.traj_y)
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.title("S%d_run%d_trajetcories"%(scn_nbr,run_nbr))
        
        plt.savefig(main_dir+"/S%d_run%d_trajectories"%(scn_nbr,run_nbr))
        
        #vel_x
        plt.subplot(1,2,1)
        for p in pedestrians:
            plt.plot(p.traj_x,p.vel_x)  #vel_x vs x
        plt.xlabel('x(m)')
        plt.ylabel('vel_x(m/s)')
        plt.title("S%d_run%d_vel_x"%(scn_nbr,run_nbr))
        #plt.savefig(main_dir+"S%d_run%d_vel_x"%(scn_nbr,run_nbr))
        
        #vel_y
        plt.subplot(1,2,2)
        for p in pedestrians:
            plt.plot(p.traj_y,p.vel_y)  #vel_x vs x
        plt.xlabel('y(m)')
        plt.ylabel('vel_y(m/s)')
        plt.title("S%d_run%d_vel_y"%(scn_nbr,run_nbr))
        plt.tight_layout()
        plt.savefig(main_dir+"/S%d_run%d_vel"%(scn_nbr,run_nbr))
        
        #acc_x
        plt.subplot(1,2,1)
        for p in pedestrians:
            plt.plot(p.traj_x,p.acc_x)  #acc_x vs x
        plt.xlabel('x(m)')
        plt.ylabel('acc_x(m2/s)')
        plt.title("S%d_run%d_acc_x"%(scn_nbr,run_nbr))
        #plt.savefig(main_dir+"S%d_run%d_vel_x"%(scn_nbr,run_nbr))
        
        #vel_x
        plt.subplot(1,2,2)
        for p in pedestrians:
            plt.plot(p.traj_y,p.acc_y)  #acc_y vs x
        plt.xlabel('y(m)')
        plt.ylabel('acc_y(m/s)')
        plt.title("S%d_run%d_acc_y"%(scn_nbr,run_nbr))
        plt.tight_layout()
        plt.savefig(main_dir+"/S%d_run%d_acc"%(scn_nbr,run_nbr))
        
        #jerk_x
        plt.subplot(1,2,1)
        for p in pedestrians:
            plt.plot(p.traj_x,p.jerk_x)  #vel_x vs x
        plt.xlabel('x(m)')
        plt.ylabel('jerk_x(m/s)')
        plt.title("S%d_run%d_jerk_x"%(scn_nbr,run_nbr))
        #plt.savefig(main_dir+"S%d_run%d_vel_x"%(scn_nbr,run_nbr))
        
        #vel_x
        plt.subplot(1,2,2)
        for p in pedestrians:
            plt.plot(p.traj_y,p.jerk_y)  #vel_x vs x
        plt.xlabel('y(m)')
        plt.ylabel('jerk_y(m/s)')
        plt.title("S%d_run%d_jerk_y"%(scn_nbr,run_nbr))
        plt.tight_layout()
        plt.savefig(main_dir+"/S%d_run%d_jerk"%(scn_nbr,run_nbr))
        
        
        

        #HERE ONLY LOOK AT BOTTLENECK SECTION 

        #decide which direction of flow
        #direction=0 means from up to down (y decrease)
        #direction=1 means from down to up (y increase)
        #NOTE: THIS WILL BE DIFFERENT BASED ON WALKING DIRECTION
        before_bn_y=[(0,-4),(-8,-12)]
        at_bn_y=[(-4,-6.5),(-5.5,-8)]
        after_bn_y=[(-6.5,-12),(0,-5.5)]

        for i in range(0,len(pedestrians)):
            pedestrians[i].nearestP=[]
            pedestrians[i].bottleneck_y=[]

        if pedestrians[0].traj_y[0]>pedestrians[0].traj_y[-1]:
            direction=0
        else:
            direction=1

        #find out distance to NN
        for p_target in range(0,len(pedestrians)):
            for y in pedestrians[p_target].traj_y:
                 if (direction==0 and at_bn_y[0][0]>=y and at_bn_y[0][1]<y) or (direction==1 and at_bn_y[1][0]>=y and at_bn_y[1][1]<y):
                    pedestrians[p_target].bottleneck_y.append(y)
                    #at bottleneck
                    min_dist=100000
                    search_x=100000
                    search_y=100000

                    for p_search in range(0,len(pedestrians)):
                        target_x=pedestrians[p_target].traj_x[pedestrians[p_target].traj_y.index(y)]
                        target_y=y
                        if p_search!= p_target:
                            search_x=pedestrians[p_search].traj_x[pedestrians[p_target].traj_y.index(y)]
                            search_y=pedestrians[p_search].traj_y[pedestrians[p_target].traj_y.index(y)]

                        dist=np.sqrt((target_x-search_x)**2+(target_y-search_y)**2)
                        if dist<min_dist:
                            min_dist=dist
                            min_nbr=p_search

                    pedestrians[p_target].nearestP.append((min_nbr,min_dist))
            avg_minNN_dist.append(np.min(list(np.array(pedestrians[p_target].nearestP)[:,1]))) #min distance to other ped/robot during crossing

        #for all pedestrians, plot nearest_dist vs traj_y
        plt.figure()
        for p in pedestrians:
            print("bottleneck length ",len(p.bottleneck_y),"nearestP length ",len(p.nearestP))
            plt.scatter(p.bottleneck_y,list(np.array(p.nearestP)[:,1]))
       
        plt.xlabel('y(m)')
        plt.ylabel('nearestDist(m)')
        plt.title('S%d_run%d_nearestDist VS bottleneck_y' %(scn_nbr,run_nbr))

        plt.savefig(dir+'S%d_run%d_nearestDist VS bottleneck_y' %(scn_nbr,run_nbr))
        plt.show()

        ###########################################################################
        #calculate time the first person enter the area and the last person out of the area(robot doesn't count)
        #normalized by no.of people

        #read all extend_traj(robot ID is the first one), calculate avg duration for each scenario, write to file

        for p in pedestrians:
            if direction==0:
                all_idx=[idx for idx in range(len(p.traj_y)) if p.traj_y[idx] > -6.5 and p.traj_y[idx] <= -4]
            else:
                all_idx=[idx for idx in range(len(p.traj_y)) if p.traj_y[idx] > -8 and p.traj_y[idx] <= -5.5]

            p.enter_idx=np.min(all_idx)
            p.exit_idx=np.max(all_idx)

        first_enter_frame=np.min([p.enter_idx for p in pedestrians if p.ID!=robot_ID])   #exclude robot
        last_exit_frame=np.max([p.exit_idx for p in pedestrians  if p.ID!=robot_ID])      #exclude robot

      
        bottleneck_duration=(last_exit_frame-first_enter_frame)/(fps*p_num) #in s

        print("S",scn_nbr,"_run",run_nbr," bottleneck_duration is:",bottleneck_duration)
        bottleneck_duration_file.write("S"+str(scn_nbr)+"_run"+str(run_nbr)+" bottleneck_duration is:"+str(bottleneck_duration)+", pedestrian+robot num: "+str(p_num)+", robot: "+robot_type+"\n")

        
      #######################################################################################
        

        #evaluate passing order
        #define prev_order and its obeservation area
        if direction==0:
            prev_obs_line=-5
            obs_line=-6 #this is the exact line where the gate is(need to check)
        else:
            prev_obs_line=-7
            obs_line=-6

        prev_order=[]
        prev_arrive_time=[]

        def takeSecond(elem):
            return elem[1]

        for p in pedestrians:

            if direction==0:
                temp=[t for t in p.frame if p.traj_y[t]<=prev_obs_line] 

                prev_arrive_time.append((p.ID,temp[0]))
            else:
                temp=[t for t in p.frame if p.traj_y[t]>=prev_obs_line]
                prev_arrive_time.append((p.ID,temp[0]))  

        prev_arrive_time.sort(key=takeSecond) 
        prev_order=list(np.array(prev_arrive_time)[:,0])

        #calculate actual passing order 
        pass_order=[]
        pass_arrive_time=[]


        for p in pedestrians:

            if direction==0:
                temp=[t for t in p.frame if p.traj_y[t]<=obs_line] 

                pass_arrive_time.append((p.ID,temp[0]))
            else:
                temp=[t for t in p.frame if p.traj_y[t]>=obs_line]
                pass_arrive_time.append((p.ID,temp[0]))  

        pass_arrive_time.sort(key=takeSecond) 
        pass_order=list(np.array(pass_arrive_time)[:,0])

        print("prev order:",prev_order," pass order:",pass_order)

        #evaluate passing order wrt to robot
        #eval_pass_order=[]
        #0 indicate no change, 1 indicate reverse-robot first to people first, -1 indicate reverse--people first to robot first

        if robot_ID == -1: #no robot case, compare with people starting from same position? (but different speed to robot?) or compare all pairs
           #THIS IS TEMP SOLUTION--NOT CORRECT
            eval_ID=pedestrians[0].ID  
        else:
            eval_ID=robot_ID

        for p in pedestrians:
            if (p.ID != eval_ID):
                if (prev_order.index(eval_ID)<prev_order.index(p.ID)) and (pass_order.index(eval_ID)>pass_order.index(p.ID)):
                    p.eval_pass_order=1
                    exceed_robot+=1
                elif (prev_order.index(eval_ID)>prev_order.index(p.ID)) and (pass_order.index(eval_ID)<pass_order.index(p.ID)):
                    p.eval_pass_order=-1
                    give_way+=1
                else:
                    p.eval_pass_order=0
                    same+=1
                    
                people_count+=1

            pass_order_file.write("S"+str(scn_nbr)+"_run"+str(run_nbr)+" pedestrians "+str(p.ID)+" pass order is "+str(p.eval_pass_order)+",robot: "+robot_type+"\n")

            
        ########################################################
        #only in bottleneck is interested!!!!before and after is not! IT depends on the video!!! 
        if direction==0:
            y_before=before_bn_y[0]
            y_at=at_bn_y[0]
            y_after=after_bn_y[0]
        else:
            y_before=before_bn_y[1]
            y_at=at_bn_y[1]
            y_after=after_bn_y[1]

        #calculate avg speed at three section--only interested in the mid section
        for p in pedestrians:
            p.simple_speed=[]
            before_bn_disty=[]
            at_bn_disty=[]
            after_bn_disty=[]
            before_bn_frame=[]
            at_bn_frame=[]
            after_bn_frame=[]
            
            at_bn_speed=[]
            at_bn_acc=[]
            
            for i in range(0,len(p.traj_y)):
                #before bottleneck
                if y_before[0]>=p.traj_y[i] and y_before[1]<p.traj_y[i]:
                    before_bn_disty.append((p.traj_x[i],p.traj_y[i]))
                    before_bn_frame.append(p.frame[i])
                #at bottleneck
                elif y_at[0]>=p.traj_y[i] and y_at[1]<p.traj_y[i]:
                    at_bn_disty.append((p.traj_x[i],p.traj_y[i])) 
                    at_bn_frame.append(p.frame[i])
                    at_bn_speed.append(np.sqrt((p.traj_y[i]-p.traj_y[i-1])**2+(p.traj_x[i]-p.traj_x[i-1])**2)/dt)
                    at_bn_acc.append((at_bn_speed[-1]-(np.sqrt((p.traj_y[i-1]-p.traj_y[i-2])**2+(p.traj_x[i-1]-p.traj_x[i-2])**2)/dt))/dt)
                #after bottleneck   
                elif y_after[0]>=p.traj_y[i] and y_after[1]<p.traj_y[i]:
                    after_bn_disty.append((p.traj_x[i],p.traj_y[i]))
                    after_bn_frame.append(p.frame[i])
                    
            #calculate pedestrians avg speed,acc
            p.avg_acc=(at_bn_speed[-1]-at_bn_speed[0])/((np.max(before_bn_frame)-np.min(before_bn_frame))/fps)
           
            #calculate avg speed for three sections--stored in p.simple_speed
            if len(before_bn_disty)!=0:        
                p.simple_speed.append(np.sqrt((np.max(np.array(before_bn_disty)[:,0])-np.min(np.array(before_bn_disty)[:,0]))**2+(np.max(np.array(before_bn_disty)[:,1])-np.min(np.array(before_bn_disty)[:,1]))**2)/((np.max(before_bn_frame)-np.min(before_bn_frame))/fps))
            if len(at_bn_disty)!=0:     
                p.simple_speed.append(np.sqrt((np.max(np.array(at_bn_disty)[:,0])-np.min(np.array(at_bn_disty)[:,0]))**2+(np.max(np.array(at_bn_disty)[:,1])-np.min(np.array(at_bn_disty)[:,1]))**2)/((np.max(at_bn_frame)-np.min(at_bn_frame))/fps))
            if len(after_bn_disty)!=0:     
                p.simple_speed.append(np.sqrt((np.max(np.array(after_bn_disty)[:,0])-np.min(np.array(after_bn_disty)[:,0]))**2+(np.max(np.array(after_bn_disty)[:,1])-np.min(np.array(after_bn_disty)[:,1]))**2)/((np.max(after_bn_frame)-np.min(after_bn_frame))/fps))

        
        #ONLY INTERESTED IN BOTTLENECK SPEED!
        allp_simple_speed_bn=[]
        #allp_simple_acc_bn=[]
        for p in pedestrians:
            if p.ID != robot_ID: #extract all avg speed,acc for pedestrians!!
                allp_simple_speed_bn.append(p.simple_speed[1])
                allp_scn_sim_speed_bn.append(p.simple_speed[1]) #add all pedestrians avg speed for one scene
                allp_scn_simple_acc_bn.append(p.avg_acc)
            else:
                robot_speed_bn.append(p.simple_speed[1]) #robot avg speed through bottleneck
            
        if robot_ID !=-1:
            ped_num=p_num-1
        else:
            ped_num=p_num
         
        #robot is not considered!!
        scn_simple_speed_bn.append(np.sum(allp_simple_speed_bn)/ped_num) #average speed for all pedestrians at bottleneck
        ############################################################3
       
        #find out segment trajectories for robot&ped in close distance
        #record the ped ID number which is closest to robot throught out robot's traj at each time

        if robot_ID !=-1:
            #pedestrians trajectories segments which are closet to robot & corresponding robot segment
            robot_nearestID=[]
            for p in pedestrians:
                if p.ID == robot_ID:
                    target_x=p.traj_x
                    target_y=p.traj_y
            
            for t in range(len(pedestrians[0].frame)):
                min_dist=100000  #set a threshold? 
                search_x=100000
                search_y=100000
                min_ID=-1

                for p in pedestrians:
                    if p.ID!=robot_ID:
                        search_x=p.traj_x[t]  
                        search_y=p.traj_y[t]
                       
                        dist=np.sqrt((target_x[t]-search_x)**2+(target_y[t]-search_y)**2)
                    
                    if dist<min_dist:
                        min_dist=dist
                        min_ID=p.ID

                if min_ID !=-1:
                    robot_nearestID.append(min_ID)
            print(robot_nearestID)

            #create new segments--closest to robot
            #to analyse vel, acc etc --only make sense when the segment is in bottleneck? 
            close_segment=[]

            for t in range(int(v.get(cv2.CAP_PROP_POS_FRAMES))):
                for p in pedestrians:
                    if robot_nearestID[t]==p.ID:
                        close_segment.append([p.traj_x[t],p.traj_y[t],t])
            j=0
            len_count=0
            close_segment_split=[[close_segment[0]]]

            #robot segment
            robot_x=[i.traj_x for i in pedestrians if i.ID==robot_ID]
            robot_y=[i.traj_y for i in pedestrians if i.ID==robot_ID]


            robot_segment_split=[[[robot_x[0][0],robot_y[0][0],0]]] 
            
            for i in range(1,len(robot_nearestID)):

                if robot_nearestID[i] == robot_nearestID[i-1]:
                    len_count+=1
                    close_segment_split[j].append(close_segment[i])
                    robot_segment_split[j].append([robot_x[0][i],robot_y[0][i],i]) #i is time(frame here)

                else:
                    close_segment_split.append([])
                    robot_segment_split.append([])
                    j+=1

            #only interested in those that initeraction time > 1s--12 frames
            #for one run
            close_segment_split=[i for i in close_segment_split if len(i)>=11]
            robot_segment_split=[i for i in robot_segment_split if len(i)>=11]
          
            #calculate avg vel,acc,dist for those segments
            acc_x_seg=[]
            acc_y_seg=[]
            vel_x_seg=[]
            vel_y_seg=[]
            min_dist_seg=[]

            #calculate average vel,accelaration (y direction) per segment 
            for i in close_segment_split:
                ins_vel_x=[0]
                ins_vel_y=[0]

                vel_x_seg.append((list(np.array(i)[:,0])[0]-list(np.array(i)[:,0])[-1])/(len(i)/fps))
                vel_y_seg.append((list(np.array(i)[:,1])[0]-list(np.array(i)[:,1])[-1])/(len(i)/fps))


                for j in range(1,len(i)):
                    ins_vel_x.append((i[j][0]-i[j-1][0])/(1/fps))
                    ins_vel_y.append((i[j][1]-i[j-1][1])/(1/fps))

                acc_x_seg.append((ins_vel_x[-1]-ins_vel_x[1])/(len(i)/fps))
                acc_y_seg.append((ins_vel_y[-1]-ins_vel_y[1])/(len(i)/fps))

            #calculate minimum distance per segment (with robot)
            for i in range(len(close_segment_split)):
                min_dist=10000
                for j in range(len(close_segment_split[i])):
                    ped_x=np.array(close_segment_split[i])[j,0]
                    ped_y=np.array(close_segment_split[i])[j,1]
                    robot_x=np.array(robot_segment_split[i])[j,0]
                    robot_y=np.array(robot_segment_split[i])[j,1]

                    dist=np.sqrt((ped_x-robot_x)**2+(ped_y-robot_y)**2)
                    if dist<min_dist:
                        min_dist=dist
                min_dist_seg.append(min_dist)

            print("min_dist",min_dist_seg)       

          

            #evaluate those segment are in which region
            plt.figure()
            for i in range(len(close_segment_split)):
                plt.scatter(np.array(close_segment_split[i])[:,0],np.array(close_segment_split[i])[:,1])
            for i in range(len(robot_segment_split)):
                plt.scatter(np.array(robot_segment_split[i])[:,0],np.array(robot_segment_split[i])[:,1],color='black')
                
            plt.plot(range(12),[-6]*12)  
            plt.xlabel('x(m)')
            plt.ylabel('y(m)')
            plt.title('S%d_run%d_traj nearest to robot' %(scn_nbr,run_nbr))       
            
            if direction==0:
                text(0.5, -4, 'before')
                text(0.5, -8, 'after')
            else:
                text(0.5, -8, 'before')
                text(0.5, -4, 'after')
                
            plt.savefig(main_dir+"/S%d/run%d/close_traj_segments" %(scn_nbr,run_nbr))
            plt.show()
            #need to add legend

            close_segments_file.write("S"+str(scn_nbr)+"_run"+str(run_nbr)+" pedestrains closet segment to robot is: "+str(close_segment_split)+",average acc_x is: "+str(acc_x_seg)+",avegrate acc_y is: "+str(acc_y_seg)+"min dist is: "+str(min_dist_seg))

            ####################################################################
            #estimate local density (equation from ref1)

            #find nearest neighbour--distance for each pedestrians
            #traj_y vs nearest_dist
            for p_target in pedestrians:
                p_target.nearestD=[]
                for y in p_target.traj_y:
                    #if p is in bottleneck (y=-4,-6.5)
                    #at bottleneck
                    min_dist=100000
                    search_x=100000
                    search_y=100000

                    for p_search in pedestrians:
                        target_x=p_target.traj_x[p_target.traj_y.index(y)]
                        target_y=y
                        if p_search!= p_target:
                            search_x=p_search.traj_x[p_target.traj_y.index(y)]
                            search_y=p_search.traj_y[p_target.traj_y.index(y)]

                        dist=np.sqrt((target_x-search_x)**2+(target_y-search_y)**2)
                        #print("dist:",dist,"target_x:",target_x)
                        if dist<min_dist:
                            min_dist=dist
                            min_nbr=p_search
                            
                    if(min_dist==0):
                        min_dist=0.000001 #prevent division by 0

                    p_target.nearestD.append(min_dist) #tuple(min_nbr,dist)


            #4 interested timestamps
            val_t_list=[0,int(v.get(cv2.CAP_PROP_POS_FRAMES)/4),int(v.get(cv2.CAP_PROP_POS_FRAMES)/3),int(v.get(cv2.CAP_PROP_POS_FRAMES)/2)]           #t here is frame!!!NOT TIME IN S!!
            a=0.7 #smooth factor

            all_density=[]
            all_speed=[]
            all_flow=[]
            for val_t in val_t_list:
                #distance to nearest neighbour
                
                #1/2PI
                #p_t_x_y=lambda x,y,t:0.5*np.pi*np.sum([1/((a*p.nearestD[t])**2)*np.exp(-((p.traj_x[t]-x)**2+(p.traj_y[t]-y)**2)/(2*(a*p.nearestD[t])**2)) for p in pedestrians])
                #p_valt_x_y=lambda x,y,t=val_t:p_t_x_y(x,y,t)  #assign val_t to t, function  become x,y as variable
                
                #1/PI
                p_t_x_y=lambda x,y,t:np.sum([1/(np.pi*a**2)*np.exp(-((p.traj_x[t]-x)**2+(p.traj_y[t]-y)**2)/a**2) for p in pedestrians])
                p_valt_x_y=lambda x,y,t=val_t:p_t_x_y(x,y,t)  #assign val_t to t, function  become x,y as variable
                #print(p_valt_x_y)
                range_X=np.linspace(0,12,100)
                range_Y=np.linspace(0,-12,100)
                density=[[]]
                for i in range(len(range_X)):
                    for j in range(len(range_Y)-1,-1,-1):
                        density[i].append(p_valt_x_y(range_X[i],range_Y[j]))
                    if i!=len(range_X)-1:
                        density.append([])


                fig, ax = plt.subplots(figsize=(12, 12))

                heatmap = ax.pcolor((np.array(density)/fps).transpose(), cmap='hot')  #here it converts to s
                cbar = plt.colorbar(heatmap)
           
                ax.set_xlabel('x(m)')
                ax.set_ylabel('y(m)')
                ax.set_xticks([0,len(range_X)])
                ax.set_yticks([0,len(range_Y)])
                ax.set_xticklabels([str(range_X[0]),str(range_X[-1])])
                ax.set_yticklabels([str(range_Y[-1]),str(range_Y[0])])
           

                ax.set_title("S%d_run%d_local_crowds_density" %(scn_nbr,run_nbr))

                plt.savefig(main_dir+"/S%d/run%d/S%d_run%d_local_crowds_density_t=%d" %(scn_nbr,run_nbr,scn_nbr,scn_nbr,val_t))
                plt.show()
                
                #local speed 
                v_t_x_y=lambda x,y,t:np.sum([np.sqrt(p.vel_y[t]**2)*(1/(np.pi*a))*np.exp(-((p.traj_x[t]-x)**2+(p.traj_y[t]-y)**2)/(a**2)) for p in pedestrians])/np.sum([(1/(np.pi*a))*np.exp(-((p.traj_x[t]-x)**2+(p.traj_y[t]-y)**2)/(a**2)) for p in pedestrians])
                v_valt_x_y=lambda x,y,t=val_t:v_t_x_y(x,y,t)  
          
                range_X=np.linspace(0,12,100)
                range_Y=np.linspace(0,-12,100)
                speed=[[]]
                for i in range(len(range_X)):
                    for j in range(len(range_Y)-1,-1,-1):
                        speed[i].append(v_valt_x_y(range_X[i],range_Y[j]))
                    if i!=len(range_X)-1:
                        speed.append([])

                fig, ax = plt.subplots(figsize=(12, 12))

                heatmap = ax.pcolor((np.array(speed)/fps).transpose(), cmap='hot') #here convert to s
                cbar = plt.colorbar(heatmap)
              
                ax.set_xlabel('x(m)')
                ax.set_ylabel('y(m)')
                ax.set_xticks([0,len(range_X)])
                ax.set_yticks([0,len(range_Y)])
                ax.set_xticklabels([str(range_X[0]),str(range_X[-1])])
                ax.set_yticklabels([str(range_Y[-1]),str(range_Y[0])])


                ax.set_title("S%d_run%d_local_crowds_speed" %(scn_nbr,run_nbr))

                plt.savefig(main_dir+"/S%d/run%d/S%d_run%d_local_crowds_speed_t=%d" %(scn_nbr,run_nbr,scn_nbr,scn_nbr,val_t))
                plt.show()
                
               #estimate local flow q=p*v
     
                q_t_x_y=lambda x,y,t:p_t_x_y(x,y,t)*v_t_x_y(x,y,t)
                q_valt_x_y=lambda x,y,t=val_t:q_t_x_y(x,y,t) 


                range_X=np.linspace(0,12,100)
                range_Y=np.linspace(0,-12,100)
                flow=[[]]
                for i in range(len(range_X)):
                    for j in range(len(range_Y)-1,-1,-1):
                        flow[i].append(q_valt_x_y(range_X[i],range_Y[j]))
                    if i!=len(range_X)-1:
                        flow.append([])


                fig, ax = plt.subplots(figsize=(12, 12))

                heatmap = ax.pcolor((np.array(flow)/fps).transpose(), cmap='hot') #here convert to s

                cbar = plt.colorbar(heatmap)
              
                ax.set_xlabel('x(m)')
                ax.set_ylabel('y(m)')
                ax.set_xticks([0,len(range_X)])
                ax.set_yticks([0,len(range_Y)])
                ax.set_xticklabels([str(range_X[0]),str(range_X[-1])])
                ax.set_yticklabels([str(range_Y[-1]),str(range_Y[0])])


                ax.set_title("S%d_run%d_local_crowds_flow" %(scn_nbr,run_nbr))

                plt.savefig(main_dir+"/S%d/run%d/S%d_run%d_local_crowds_flow_t=%d" %(scn_nbr,run_nbr,scn_nbr,scn_nbr,val_t))
                plt.show()
                
                if val_t == int(v.get(cv2.CAP_PROP_POS_FRAMES)/4):
                    all_density.append(density)
                    all_speed.append(speed)
                    all_flow.append(flow)
              
            #plot FD---is it correct??? (PLOT RESULT FOR ALL X,Y,T POINTS?)
            plt.subplot(1,3,1)
            plt.scatter(np.ravel(all_density)/fps,np.ravel(all_flow)/fps)
            plt.xlabel('density')
            plt.ylabel('flow')

            plt.subplot(1,3,2)
            plt.scatter(np.ravel(all_density)/fps,np.ravel(all_speed)/fps)
            plt.xlabel('density')
            plt.ylabel('speed')

            plt.subplot(1,3,3)
            plt.scatter(np.ravel(all_flow)/fps,np.ravel(all_speed)/fps)
            plt.xlabel('flow')
            plt.ylabel('speed')
            plt.savefig(main_dir+"/S%d/run%d/S%d_run%d_local_FD_diagram" %(scn_nbr,run_nbr,scn_nbr,scn_nbr))
            plt.tight_layout()
            plt.show()
                
            #################################################################################
            #simple path efficiency (distance/actual segment length)  NEED TO CONSIDER PEOPLE'S SWAY MOTION AND GATE ?!
            path_efficiency=[]
        
            for seg in close_segment_split:
                end_point=seg[-1]
                start_point=seg[0]
                Euclidean_dist=np.sqrt((start_point[0]-end_point[0])**2+(start_point[1]-end_point[1])**2)
                seg_length=0
                for s in range(1,len(seg)):

                    seg_length+=np.sqrt((seg[s][0]-seg[s-1][0])**2+(seg[s][1]-seg[s-1][1])**2)

                if seg_length!=0:    
                    path_efficiency.append(Euclidean_dist/seg_length)
                    scn_all_efficiency.append(Euclidean_dist/seg_length)

            print(path_efficiency)
           

    #process characterstic for each scene;
    #################################################
    Scene['S'+str(scn_nbr)]={'P_avg_speed':allp_scn_sim_speed_bn,'P_avg_acc':allp_scn_simple_acc_bn,'R_avg_speed':robot_speed_bn,'avg_minNN_dist':avg_minNN_dist,'P_nearR_path_eff':scn_all_efficiency}
    
     
        
    avg_scn_simple_speed_bn.append(np.sum(scn_simple_speed_bn)/len(scn_simple_speed_bn))  #avegrage speed for passing the bottleneck-for each scene
  
        
    plt.figure()
    plot_X=['people give way','people exceed robot']
    plot_Y=[give_way/people_count,exceed_robot/people_count]
    plt.plot(plot_X,plot_Y)
    plt.title('S%d_passing_order'%(scn_nbr))
    plt.savefig(main_dir+"/S%d/pass_order" %(scn_nbr))
    
    
    #plot distribution of average pedestrians speed for passing the bottleneck
    plt.figure()
    x=allp_scn_sim_speed_bn
    plt.hist(x,bins=50)
    plt.gca().set(title='S%d_run%d_PEDspeed_Frequency Histogram'%(scn_nbr,run_nbr), ylabel='Frequency');
    
    plt.savefig(main_dir+"/S%d/avg_pedestrians_speed" %(scn_nbr))
    
    #robot speed distribution
    if robot_ID !=-1:
        plt.figure()
        x=robot_speed_bn
        plt.hist(x,bins=50)
        plt.gca().set(title='S%d_run%d_ROBOTspeed_Frequency Histogram'%(scn_nbr,run_nbr), ylabel='Frequency');

        plt.savefig(main_dir+"/S%d/avg_robot_speed" %(scn_nbr))
    
    #histogram of path efficiecy for each scneario
    plt.figure()
    x=scn_all_efficiency
    plt.hist(x,bins=50)
    plt.gca().set(title='S%d_run%d_Efficiency_Frequency Histogram'%(scn_nbr,run_nbr), ylabel='Frequency');
    
    plt.savefig(main_dir+"/S%d/avg_efficiency" %(scn_nbr))
    

plt.figure()
plot_X=['S'+str(i) for i in scn_nbr_list]
plot_Y=avg_scn_simple_speed_bn
plt.plot(plot_X,plot_Y)
plt.show()
    
        
pass_order_file.close()        
bottleneck_duration_file.close() 
close_segments_file.close()

###########################################################

#compare S1/S3/S8
#box plot for PEDspeed 
#compare the case without robot/wheelchair/pepper--wide gate (WC,PEPPER-low speed)


scn_138=['S1','S3','S8']
scn_246=['S2','S4','S6']

for i in scn_138:
    scn_avg_speed=np.sum(Scene[i]['P_avg_speed'])/len(Scene[i]['P_avg_speed'])
    scn_speed_std=np.std(Scene[i]['P_avg_speed'])
#print(scn_avg_speed,",",scn_speed_std)
plt.figure()
plt.boxplot([Scene['S1']['P_avg_speed'],Scene['S3']['P_avg_speed'],Scene['S8']['P_avg_speed']],labels = ['S1','S3','S8'],showmeans=True)    
for i in range(3):
    plt.text(i, scn_avg_speed[i]+0.5, str(scn_avg_speed[i]), horizontalalignment='center',  color='b', weight='semibold',fontsize=12)  
    
plt.title('Average Pedestrians Speed for S1/S3/S8')
plt.savefig(main_dir+'/avg PEDspeed for S1_S3_S8')

#added--avg acc
for i in scn_138:
    scn_avg_speed=np.sum(Scene[i]['P_avg_acc'])/len(Scene[i]['P_avg_acc'])
    scn_speed_std=np.std(Scene[i]['P_avg_acc'])
#print(scn_avg_speed,",",scn_speed_std)
plt.figure()
plt.boxplot([Scene['S1']['P_avg_acc'],Scene['S3']['P_avg_acc'],Scene['S8']['P_avg_acc']],labels = ['S1','S3','S8'],showmeans=True)    
for i in range(3):
    plt.text(i, scn_avg_speed[i]+0.5, str(scn_avg_speed[i]), horizontalalignment='center',  color='b', weight='semibold',fontsize=12)  
    
plt.title('Average Pedestrians Acceleration for S1/S3/S8')
plt.savefig(main_dir+'/avg PEDacc for S1_S3_S8')


#box plot for ROBOTspeed 
#compare the case without robot/wheelchair/pepper--wide gate (WC,PEPPER-low speed)
for i in scn_138:
    scn_avg_speed=np.sum(Scene[i]['R_avg_speed'])/len(Scene[i]['R_avg_speed'])
    scn_speed_std=np.std(Scene[i]['R_avg_speed'])
#print(scn_avg_speed,",",scn_speed_std)
plt.figure()
plt.boxplot([Scene['S3']['R_avg_speed'],Scene['S8']['R_avg_speed']],labels = ['S3','S8'],showmeans=True)    
for i in range(2):
    plt.text(i, scn_avg_speed[i+1]+0.5, str(scn_avg_speed[i+1]), horizontalalignment='center',  color='b', weight='semibold',fontsize=12)  
           
plt.title('Average Robot Speed for S3/S8')
plt.savefig(main_dir+'/avg ROBOTspeed for S3_S8')

#box plot for minNNdist
#compare the case without robot/wheelchair/pepper--wide gate (WC,PEPPER-low speed)
for i in scn_138:
    scn_avg_speed=np.sum(Scene[i]['avg_minNN_dist'])/len(Scene[i]['avg_minNN_dist'])
    scn_speed_std=np.std(Scene[i]['avg_minNN_dist'])
#print(scn_avg_speed,",",scn_speed_std)
plt.figure()
plt.boxplot([Scene['S1']['avg_minNN_dist'],Scene['S3']['avg_minNN_dist'],Scene['S8']['avg_minNN_dist']],labels = ['S1','S3','S8'],showmeans=True)    
for i in range(3):
    plt.text(i, scn_avg_speed[i]+0.5, str(scn_avg_speed[i]), horizontalalignment='center',  color='b', weight='semibold',fontsize=12)  
            
plt.title('Average Nearest Neighbour Distance for S1/S3/S8')
plt.savefig(main_dir+'/avg min NN distance for S1_S3_S8')

#box plot for path efficiency
#compare the case without robot/wheelchair/pepper--wide gate (WC,PEPPER-low speed)
for i in scn_138:
    scn_avg_speed=np.sum(Scene[i]['P_nearR_path_eff'])/len(Scene[i]['P_nearR_path_eff'])
    scn_speed_std=np.std(Scene[i]['P_nearR_path_eff'])
#print(scn_avg_speed,",",scn_speed_std)
plt.figure()
plt.boxplot([Scene['S3']['P_nearR_path_eff'],Scene['S8']['P_nearR_path_eff']],labels = ['S3','S8'],showmeans=True)    
for i in range(3):
    plt.text(i, scn_avg_speed[i+1]+0.5, str(scn_avg_speed[i+1]), horizontalalignment='center',  color='b', weight='semibold',fontsize=12)  
            
plt.title('Average Pedestrians(close to the robot)Path Effciency for S3/S8')
plt.savefig(main_dir+'/avg path eff for S3_S8')


# In[34]:


#S1/S3/S8---see how robot&its type effect pedestrian dynamics (robot low speed)
#S1/S4/S9---see how robot&its type effect pedestrian dynamics (robot high speed)
#S2/S5/S6---see how WC & WC speed affect pedestrian dynamics (narrow gate)
#S1/S3/S4---see how WC & WC speed affect pedestrian dynamics (wide gate)
#S1/S2---S3/S5---see how WC & GATE WIDTH affect pedestrian dynamics (WC low speed)
#S1/S2---S4/S6---see how WC & GATE WIDTH affect pedestrian dynamics (WC high speed)

#Evaluation:
#pedetsrians speed
#min distance to neighbour
#path efficiency near robot (if there is robot)

#S1-no robot, wide
#S2-no robot, narrow
#S3-WC, wide, slow
#S4-WC, wide, fast
#S5-WC, narrow, slow
#S6-WC, narrow, fast
#S7--emergency
#S8-Pepper, wide, slow
#S9-Pepper, wide, fast


# In[6]:


#p-test
from scipy import stats

p_test=open(main_dir+'/p_test.txt','a+')


comp=[['S1','S3'],['S1','S8'],['S3','S8'],['S1','S9'],['S4','S9'],['S2','S5'],['S2','S6'],['S5','S6'],['S1','S4'],['S1','S2'],['S3','S5'],['S4','S6']]

for i in range(len(comp)):
    #two tailed t test for pedestrians avg speed
    
    p_test.write(str(comp[i][0])+" & " + str(comp[i][1])+" P_avg_speed "+str(stats.ttest_ind(Scene[comp[i][0]]['P_avg_speed'],Scene[comp[i][1]]['P_avg_speed']))+"\n")
    #two tailed t test for min distance
    p_test.write(str(comp[i][0])+" & " + str(comp[i][1])+" avg_minNN_dist "+str(stats.ttest_ind(Scene[comp[i][0]]['avg_minNN_dist'],Scene[comp[i][1]]['avg_minNN_dist']))+"\n")

    #two tailed t test for path efficiency (with robot)
    p_test.write(str(comp[i][0])+" & " + str(comp[i][1])+" P_nearR_path_eff "+str(stats.ttest_ind(Scene[comp[i][0]]['P_nearR_path_eff'],Scene[comp[i][1]]['P_nearR_path_eff']))+"\n")

    #S1/S3/S8---see how robot&its type effect pedestrian dynamics (robot low speed)
    #S1/S4/S9---see how robot&its type effect pedestrian dynamics (robot high speed)
    #S2/S5/S6---see how WC & WC speed affect pedestrian dynamics (narrow gate)
    #S1/S3/S4---see how WC & WC speed affect pedestrian dynamics (wide gate)
    #S1/S2---S3/S5---see how WC & GATE WIDTH affect pedestrian dynamics (WC low speed)
    #S1/S2---S4/S6---see how WC & GATE WIDTH affect pedestrian dynamics (WC high speed)

