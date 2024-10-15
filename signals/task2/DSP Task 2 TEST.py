#!/usr/bin/env python
# coding: utf-8

# In[5]:


def ReadSignalFile(file_name):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    return expected_indices,expected_samples


# In[6]:


def AddSignalSamplesAreEqual(userFirstSignal,userSecondSignal,Your_indices,Your_samples):
    if(userFirstSignal=='Signal1.txt' and userSecondSignal=='Signal2.txt'):
        file_name="" # write here path of signal1+signal2
    else if(userFirstSignal=='Signal1.txt' and userSecondSignal=='Signal3.txt'):
        file_name="" # write here path of signal1+signal3
    expected_indices,expected_samples=ReadSignalFile(file_name)          
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Addition Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Addition Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Addition Test case failed, your signal have different values from the expected one") 
            return
    print("Addition Test case passed successfully")


# In[ ]:


def SubSignalSamplesAreEqual(userFirstSignal,userSecondSignal,Your_indices,Your_samples):
    if(userFirstSignal=='Signal1.txt' and userSecondSignal=='Signal2.txt'):
        file_name="" # write here path of signal1-signal2
    else if(userFirstSignal=='Signal1.txt' and userSecondSignal=='Signal3.txt'):
        file_name="" # write here path of signal1-signal3
        
    expected_indices,expected_samples=ReadSignalFile(file_name)   
    
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Subtraction Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Subtraction Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Subtraction Test case failed, your signal have different values from the expected one") 
            return
    print("Subtraction Test case passed successfully")


# In[ ]:


def NormalizeSignal(MinRange,MaxRange,Your_indices,Your_samples):
    if(MinRange==-1 and MaxRange==1):
        file_name="" # write here path of normalize signal 1 output.txt
    else if(MinRange==-1 and MaxRange==1):
        file_name="" # write here path of normalize signal 2 output.txt
        
    expected_indices,expected_samples=ReadSignalFile(file_name)      
    
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Normalization Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Normalization Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Normalization Test case failed, your signal have different values from the expected one") 
            return
    print("Normalization Test case passed successfully")


# In[ ]:


def MultiplySignalByConst(User_Const,Your_indices,Your_samples):
    if(User_Const==5):
        file_name="" # write here path of MultiplySignalByConstant-Signal1 - by 5.txt
    else if(User_Const==10):
        file_name="" # write here path of MultiplySignalByConstant-Signal2 - by 10.txt
        
    expected_indices,expected_samples=ReadSignalFile(file_name)      
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Multiply by "+User_Const.str()+ " Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Multiply by "+User_Const.str()+" Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Multiply by "+User_Const.str()+" Test case failed, your signal have different values from the expected one") 
            return
    print("Multiply by "+User_Const.str()+" Test case passed successfully")


# In[ ]:


def ShiftSignalByConst(Shift_value,Your_indices,Your_samples):
    if(Shift_value==500):
        file_name="" # write here path of output shifting by add 500.txt
    else if(Shift_value==-500):
        file_name="" # write here path of output shifting by minus 500.txt
        
    expected_indices,expected_samples=ReadSignalFile(file_name)      
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Shift by "+Shift_value.str()+" Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Shift by "+Shift_value.str()+" Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Shift by "+Shift_value.str()+" Test case failed, your signal have different values from the expected one") 
            return
    print("Shift by "+Shift_value.str()+" Test case passed successfully")


# In[ ]:


# use this twice one for Accumlation and one for Squaring
# Task name when call ACC or SQU
def SignalSamplesAreEqual(TaskName,file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
         print(TaskName+" Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print(TaskName+" Test case failed, your signal have different indicies from the expected one") 
            return             
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print(TaskName+" Test case failed, your signal have different values from the expected one") 
            return
    print(TaskName+" Test case passed successfully")

