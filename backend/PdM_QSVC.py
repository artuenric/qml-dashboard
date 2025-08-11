#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit.visualization.pulse_v2.interface import device_info
from qiskit import*
from qiskit.circuit import*
from qiskit.visualization import*
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import RealAmplitudes,  TwoLocal, ZZFeatureMap, EfficientSU2
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.algorithms.classifiers import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel, TrainableFidelityQuantumKernel, BaseKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import SPSA, ADAM


# In[2]:


#===================================================================================================================================
import seaborn as sns
from openpyxl import Workbook, load_workbook
import string

#===================================================================================================================================
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import*
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
#==================================================================================================================================
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
#===================================================================================================================================
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd 
import random
import os


# In[3]:


from qiskit_aer import Aer

simulator = Aer.get_backend('aer_simulator')
simulator.set_options(device='GPU', blocking_qubits=23, blocking_enable=True)


# In[4]:


def kernel(feature_Map_, n_features, entanglement,batch_size, simulator, reps = 1, initial_state=  None, savefig = False, insert_barriers = True  ,parameter_prefix = "$\\theta$"):
	
    sampler = Sampler()
    

    fidelity = ComputeUncompute(sampler = sampler)
    
    if feature_Map_ == 'ZZFeatureMap':
        feature_Map = ZZFeatureMap(feature_dimension = n_features, reps= reps, entanglement = entanglement, parameter_prefix = parameter_prefix, insert_barriers=insert_barriers)
        qKernel =  TrainableFidelityQuantumKernel(feature_map = feature_Map,  fidelity=None, training_parameters=None)
        if savefig == True:
            feature_Map.decompose().draw('mpl')
            plt.savefig("compilacoes/Experimento_2/Circuitos/" + "ZZfeatureMap" + f"{n_features} qubits" + f" {entanglement}"  + ".png")
        else:
            None
        
    elif feature_Map_ == 'RealAmplitudes':
        feature_Map = RealAmplitudes(num_qubits = int(n_features/2), entanglement = entanglement, reps=reps, parameter_prefix=parameter_prefix, insert_barriers=insert_barriers, initial_state = initial_state)
        qKernel = BaseKernel(feature_map = feature_Map,  fidelity=None, training_parameters=None)
        if savefig == True:
            feature_Map.decompose().draw('mpl')
            plt.savefig("compilacoes/Experimento_2/Circuitos/"+"Real Amplitudes" + f"{n_features/2} qubits" + f" {entanglement}"  + ".png")
        else:
            None
    elif feature_Map_ == 'EfficientSU2':
        feature_Map = EfficientSU2(num_qubits = int(n_features/4), su2_gates = None, entanglement = entanglement, reps=reps, parameter_prefix=parameter_prefix, insert_barriers=insert_barriers, initial_state = initial_state)
        qKernel = TrainableFidelityQuantumKernel(feature_map = feature_Map,  fidelity=None, training_parameters=None)
        if savefig == True:
            feature_Map.decompose().draw('mpl')
            plt.savefig("compilacoes/Experimento_2/Circuitos/"+"EfficientSU2" + f"{n_features/4} qubits" + f" {entanglement}"  + ".png")
        else:
            None
    
    else:
        print("Feature Map desconhecido")
    
    return feature_Map, qKernel



def resultsQ(y_true,y_pred, feature_map, n_features, entanglement, average ,fail, savefig = False):
    conf_matrix_Q = confusion_matrix(y_true, y_pred)
    figQ, axQ = plt.subplots(figsize=(5, 5))
    axQ.matshow(conf_matrix_Q, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix_Q.shape[0]):
        for j in range(conf_matrix_Q.shape[1]):
            axQ.text(x=j, y=i,s=conf_matrix_Q[i, j], va='center', ha='center', size='xx-large')
    
    if feature_map == 'ZZFeatureMap':
        n_qbits = n_features
    elif feature_map == 'RealAmplitudes':
        n_qbits = int(n_features/2)
    elif feature_map == 'EfficientSU2':
        n_qbits = int(n_features/4)
    
    #title = "Kernell: Quântico " + feature_map + " " + str(n_features)+ " features" +  " com " + str(n_qbits) + " qubits " + str(entanglement)
    title ="Kernell: Quântico " + str(feature_map)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(title)
    resultQ = pd.DataFrame({"Accuracy": round(accuracy_score(y_true, y_pred)*100,3),"Precision": round(precision_score(y_true, y_pred, average=average)*100,3),"Recall": round(recall_score(y_true, y_pred, average=average)*100,3),"F1_Score": round(f1_score(y_true, y_pred,average=average)*100,3),}, index = range(4)).head(1)
  
    
    if savefig == True:
        plt.savefig('compilacoes/Experimento_2/Matrizes_de_confusao/Quantum/' + feature_map + " " + str(n_features)+ " features" +  " com"  + " qubits " + str(entanglement) + str(fail))
        resultQ = pd.DataFrame({"Accuracy": round(accuracy_score(y_true, y_pred)*100,3),"Precision": round(precision_score(y_true, y_pred, average=average)*100,3),"Recall": round(recall_score(y_true, y_pred, average=average)*100,3),"F1_Score": round(f1_score(y_true, y_pred, average=average)*100,3),}, index = range(4)).head(1).to_csv('compilacoes/Experimento_1/Resultados/Quantum/' + feature_map + " " + str(n_features)+ " features" +  " com"  + " qubits " + str(entanglement) + str(fail))
    

    return resultQ


def resultsC(y_true,y_pred, feature_map, fail, average, savefig = False):
    conf_matrix_C = confusion_matrix(y_true, y_pred)
    figC, axC = plt.subplots(figsize=(5, 5))
    axC.matshow(conf_matrix_C, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix_C.shape[0]):
        for j in range(conf_matrix_C.shape[1]):
            axC.text(x=j, y=i,s=conf_matrix_C[i, j], va='center', ha='center', size='xx-large')
    
    
    title ="Kernell: Clássico " + str(feature_map)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(title)
    resultC = pd.DataFrame({"Accuracy": round(accuracy_score(y_true, y_pred)*100,3),"Precision": round(precision_score(y_true, y_pred, average=average)*100,3),"Recall": round(recall_score(y_true, y_pred, average=average)*100,3),"F1_Score": round(f1_score(y_true, y_pred, average=average)*100,3),}, index = range(4)).head(1)
    
    if savefig == True:
        plt.savefig('compilacoes/Experimento_2/Matrizes_de_confusao/Classical/' + "Kernel_Classico " + str(feature_map) + str(fail))
        resultC = pd.DataFrame({"Accuracy": round(accuracy_score(y_true, y_pred)*100,3),"Precision": round(precision_score(y_true, y_pred, average=average)*100,3),"Recall": round(recall_score(y_true, y_pred, average=average)*100,3),"F1_Score": round(f1_score(y_true, y_pred, average=average)*100,3),}, index = range(4)).head(1).to_csv('compilacoes/Experimento_1/Resultados/Classical/' + "Kernel_Classico" + str(feature_map) + str(fail) )
        
    
    return resultC


def feature_selectionRFE(max_iter,data,X, y, numbers_features):
    std = StandardScaler()
    X = std.fit_transform(X)
    model = LogisticRegression(max_iter=max_iter)
    rfe = RFE(model,n_features_to_select=numbers_features)
    fit = rfe.fit(X,y)
    cols = fit.get_support(indices=True)
    features = data.iloc[:,cols]
    
    return features, cols

def distribution(df_train,savefig = False):
    plt.figure(figsize=[5,5])
    plt.pie(df_train['Fault'].value_counts(), autopct='%.1f', labels=df_train['Fault'].value_counts().keys())
    plt.title(f"Distribution of classes {list(df_train['Fault'].value_counts().keys())}")
    if savefig == savefig:
        plt.savefig("compilacoes/Experimento_2/Distribuicao_treino_test/" + f"Distribution_of_class {list(df_train['Fault'].value_counts().keys())}")
    
    
    return plt.show()




# In[5]:


#======================================================= Parâmetros dos experimentos============================================================================
filename = "df_combine.csv"
path = os.path.join(os.getcwd(), 'compilacoes/data_exp_prepare')
df_combine = pd.read_csv(os.path.join(path, filename), low_memory=False)
df_combine.Fault.value_counts()
fault= [['AF', ]]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
savefig = False # Salvar Figuras
random_state = 80
numbers_features = 32 # Numeros de Features para serem selecionadas
batch_size = 400 # Tamanho dos batchs para treino
qLayers = 1  # Número de camadas do circuito quântico
perc_F_NF = 0.40  # Relação de falha com não falha
res_C = []
res_Q  = []
result_kernel = []
i = 0
#=================================================================================================================================================================

print(df_combine.shape)
df_nf_ = df_combine[df_combine.Fault=='NF']
df_f = df_combine[df_combine.Fault !='NF']

num_NF = (df_f['Fault'].value_counts()[fault[i]] - perc_F_NF*df_f['Fault'].value_counts()[fault[i]])/perc_F_NF
num_NF = int(round(num_NF, 0))

df_nf = df_combine[df_combine.Fault=='NF'].sample(num_NF, replace=True ,random_state=random_state) # SELECIONANDO 300 SAMPLES DE CASOS DE NÃO FALHA
df_combine = pd.concat((df_nf, df_f), axis=0).reset_index(drop=True)

df = df_combine.drop(columns=['DateTime_x', 'Time', 'Error', 'WEC: ava. windspeed', 
                                    'WEC: ava. available P from wind',
                                    'WEC: ava. available P technical reasons',
                                    'WEC: ava. Available P force majeure reasons',
                                    'WEC: ava. Available P force external reasons',
                                    'WEC: max. windspeed', 'WEC: min. windspeed', 
                                    'WEC: Operating Hours', 
                                    'WEC: Production minutes', 'DateTime_y'])


#df= df[(df['Fault'] == 'NF') | (df['Fault'] == fault[0][0])| (df['Fault'] == fault[0][1])| (df['Fault'] == fault[0][2])| (df['Fault'] == fault[0][3])| (df['Fault'] == fault[0][4])]
df= df[(df['Fault'] == 'NF') | (df['Fault'] == fault[0][0])]
df_f.Fault.value_counts().plot.pie(autopct = '%.2f%%', title = 'Faults Distribution')


# In[6]:


resultQ = []
resultC = []

df = df.sample(len(df))
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
le = LabelEncoder()
y = le.fit_transform(y)

for i in range(len(y)):
    if y[i] == 0:
        y[i] = 1
    elif y[i] == 1:
        y[i] = 0
        
        
df_combine.Fault.value_counts()


# In[7]:



distribution(df)
X_RFE = feature_selectionRFE(max_iter=500,data = df, X=X, y=y, numbers_features=numbers_features)
X = X_RFE[0] # Redução de features para 32
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state= random_state)

print("Quantidade de DADOS DE FALHA/NAOFALHA", df.Fault.value_counts())
print("\n")
print("Quantidade de DADOS TOTAL", df.Fault.shape[0])
print(f"Porcentagem de DADOS PARA TREINO {(1-test_size)*100}%")
print(f"Porcentagem de DADOS PARA TESTE {(test_size)*100}%")
print(f"FORMATO X_train: {X_train.shape}\nFORMATO X_test: {X_test.shape}\nFormato y_train: {y_train.shape}\nFormato y_test: {y_test.shape}")
n_features = X.shape[1]
print("Quantidade de FEATURES:", n_features)




# In[8]:


# Scalling
std = StandardScaler()
X_train_prep = pd.DataFrame(std.fit_transform(X_train))
X_test_prep = pd.DataFrame(std.fit_transform(X_test))


# In[9]:


classical_kernels = 'linear' # Define o kernel clássico
svc_classical = SVC(C=2,kernel=classical_kernels, class_weight={0:40})
svc_classical.fit(X_train_prep, y_train)
y_Cpreds = svc_classical.predict(X_test_prep)
resultC.append(resultsC(y_test, y_Cpreds, feature_map=classical_kernels, fail='MF', average='binary', savefig=False )) # Plota os resultados do kernel clássico


print("kernel Clássico")
display(resultC)


# In[10]:


#quantum_kernels = 'ZZFeatureMap' # Define o kernel quântico
quantum_kernels = 'EfficientSU2' # Define o kernel quântico
entanglements = 'full'
entanglement = entanglements

feature_map, qKernel = kernel(feature_Map_ = quantum_kernels, n_features = n_features, entanglement = entanglement, batch_size=batch_size, simulator = 'statevector_simulator', reps = qLayers, savefig=False)
svc_quantum = SVC(C=2,kernel = qKernel.evaluate, class_weight={0:40})
svc_quantum.fit(X_train_prep, y_train)
y_Qpreds = svc_quantum.predict(X_test_prep)

resultQ.append(resultsQ(y_test, y_Qpreds, feature_map=quantum_kernels, fail='AF', entanglement=entanglement, n_features = n_features, average='binary', savefig=False )) # Função plota os resultados do kernel quântico
print("kernel Quântica")
display(resultQ)


# In[ ]:

# Ao final do notebook:
accuracy = 0.89  # substitua pelo cálculo real
with open("result.txt", "w") as f:
    f.write(f"QSVC Accuracy: {accuracy*100:.2f}%")



