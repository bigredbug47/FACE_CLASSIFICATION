#FUNCTION: RENAME MULTIPLE FILE
def rename_file(path,name):
    import os
    import shutil
    import glob
    path = path+"/"+name
    i = 1
    for filename in glob.glob(os.path.join(path, '*.jpg')):
        if i%1000 == 0:
            print("Rename file "+str(i))
        new_name=path+"/"+name+"_"+str(i)+".png"
        shutil.move(filename, new_name)
        i=i+1

def move_file(path_img,path_feature):
    import os
    import shutil
    import glob
    for filename in glob.glob(os.path.join(path_img, '*.png')):
        for filename_2 in glob.glob(os.path.join(path_feature, '*.mat')):
            if os.path.splitext(os.path.basename(filename))[0] == os.path.splitext(os.path.basename(filename_2))[0]:
                new_name=path_img+"/done/"+os.path.basename(filename)
                shutil.move(filename, new_name)

#FUNCTION: RANDOM TRAINING DATA
def random_file(path_file):
    import random
    num_rand = random.sample(range(1,8000),7999) 
    #FACE
    file_train_txt = open(path_file+"/train.txt","w") 
    file_train_lb = open(path_file+"/lbtrain.txt","w")
    file_test_txt = open(path_file+"/test.txt","w") 
    file_test_lb = open(path_file+"/lbtest.txt","w")
    for i in num_rand[0:5600]: 
        file_train_txt.write("face_"+str(i)+"\n")
        file_train_lb.write("1\n")
    for i in num_rand[5600:7999]: 
        file_test_txt.write("face_"+str(i)+"\n")
        file_test_lb.write("1\n")
    file_train_txt.close() 
    file_train_lb.close() 
    file_test_txt.close() 
    file_test_lb.close()

    #NONFACE
    num_rand = random.sample(range(1,8275),8274)
    file_train_txt = open(path_file+"/train.txt","a") 
    file_train_lb = open(path_file+"/lbtrain.txt","a")
    file_test_txt = open(path_file+"/test.txt","a") 
    file_test_lb = open(path_file+"/lbtest.txt","a")
    for i in num_rand[0:5792]:
        file_train_txt.write("nonface_"+str(i)+"\n")
        file_train_lb.write("0\n")
    for i in num_rand[5792:8274]:
        file_test_txt.write("nonface_"+str(i)+"\n")
        file_test_lb.write("0\n")
    file_train_txt.close() 
    file_train_lb.close() 
    file_test_txt.close() 
    file_test_lb.close()
    print("DONE WRITE FILE")

#FUNCTION: EXTRACT FEATURE VGG16
def extract_fea_vgg16(path_fea,path_img):
    from vgg16 import VGG16
    from keras.preprocessing import image
    from keras.models import Model
    from imagenet_utils import preprocess_input
    import numpy as np
    import glob
    import os
    vgg16model = VGG16(weights='imagenet', include_top=True);
    model = Model(inputs=vgg16model.input, outputs=vgg16model.get_layer('fc2').output)
    i=1
    for filename in glob.glob(os.path.join(path_img, '*.png')):
        name_save=(filename.replace(path_img,"")).replace(".png","")
        file_feature = open(path_fea+"/"+name_save+".mat","wb") 
        img_path=filename
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        np.savetxt(file_feature, features)
        file_feature.close()
        print("Step "+str(i)+" done")
        i=i+1
    print("DONE EXTRACT FEATURE")

    
#FUNCTION: INITIALIZE INPUT
def initialize_input(path_training,path_features,file_name):
    import os
    import numpy as np
    print("Initialize input")
    input_file = open(path_training + "/"+file_name,"r")
    if file_name == "test.txt" or file_name == "train.txt":
        arr_training = []
        for line in input_file.readlines():
            for i in line.split():
                temp=[]
                inp=open(path_features + "/" + str(i) + ".mat","r")
                for subline in inp.readlines():
                # loop over the elemets, split by whitespace
                    for subi in subline.split():
                        # convert to float and append to the list
                        temp.append(float(subi))
                if len(arr_training) == 0:
                    arr_training.append(temp)
                else:
                    arr_training=np.vstack([arr_training,temp]) 
                inp.close()
        input_file.close()
        print("DONE TRAIN")
        return arr_training
    elif file_name == "lbtest.txt" or file_name == "lbtrain.txt":
        arr_lbtest=[]
        temp=[]
        for line in input_file.readlines():
            for i in line.split():
                if len(arr_lbtest) == 0:
                    temp.append(int(i))
                    arr_lbtest.append(temp)
                else:
                    temp=[]
                    temp.append(int(i))
                    arr_lbtest=np.vstack([arr_lbtest,temp])
        input_file.close()
        print("DONE TEST")
        return arr_lbtest

#FUNCTION: MERGE DATA TO 1 MATRIX
def get_training_input(path_training,path_features):
    arr_test = initialize_input(path_training,path_features,"test.txt")
    arr_lbtest = initialize_input(path_training,path_features,"lbtest.txt")
    arr_train = initialize_input(path_training,path_features,"train.txt")
    arr_lbtrain = initialize_input(path_training,path_features,"lbtrain.txt")
    return [
        arr_test,
        arr_lbtest,
        arr_train,
        arr_lbtrain
    ]

#FUNCTION: LINEAR REGRESSION MODEL
def sklearn_linear_model(x_train,lbx_train,x_test,lbx_test,path_result):
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error
    x_train=x_train.tolist()
    x_test=x_test.tolist()
    lbx_train=(lbx_train.ravel()).tolist()
    lbx_test=(lbx_test.ravel()).tolist()
    
    file_result_txt = open(path_result+"/result.txt","w") 
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(x_train, lbx_train)
    # Make predictions using the testing set
    x_pred = regr.predict(x_test)
    print("Mean squared error: %.2f"
              % mean_squared_error(lbx_test, x_pred))
    result = 1 - mean_squared_error(lbx_test, x_pred)
    
    file_result_txt.write("LINEAR REGRESSION MODEL:\n")
    file_result_txt.write("Accuracy: "+str(result)+"\n")
    file_result_txt.close()


#FUNCTION: SVM MODEL
def sklearn_svm(x_train,lbx_train,x_test,lbx_test,kernel_type,kernel_input,path_result):
    from sklearn import svm
    from sklearn.metrics import mean_squared_error
    x_train=x_train.tolist()
    x_test=x_test.tolist()
    lbx_train=(lbx_train.ravel()).tolist()
    lbx_test=(lbx_test.ravel()).tolist()
    file_result_txt = open(path_result+"/result.txt","a") 
    
    if kernel_type == "nonlinear":
        clf = svm.NuSVC()
        clf.fit(x_train, lbx_train)
        x_pred = clf.predict(x_test)
         # The mean squared error
        print("NuSVC Kernel :")
        print("Mean squared error: %.2f"
              % mean_squared_error(lbx_test, x_pred))
        result = 1 - mean_squared_error(lbx_test, x_pred)
        # Explained variance score: 1 is perfect prediction
        file_result_txt.write("SVM NUSVC MODEL:\n")
        file_result_txt.write("Accuracy: "+str(result)+"\n")
        file_result_txt.close()
    else:
        clf = svm.SVC(kernel=kernel_input)
        clf.fit(x_train, lbx_train)
        x_pred = clf.predict(x_test)
        # The mean squared error
        print("SVC Kernel " + kernel_input + " :")
        print("Mean squared error: %.2f"
              % mean_squared_error(lbx_test, x_pred))
        result = 1 - mean_squared_error(lbx_test, x_pred)
        file_result_txt.write("SVM MODEL "+"WITH "+kernel_input+" KERNEL"+"\n")
        file_result_txt.write("Accuracy: "+str(result)+"\n")
        file_result_txt.close()


#MAIN 
my_own_path="D:/CAMI/DIP/NMTGMT/face_detection/"
images_dataset_path=my_own_path+"/dataset"
#sub_name_1="face"
sub_name_2="nonface"

path_img = images_dataset_path+"/"+sub_name_2
path_fea = my_own_path+"/feature"

#move_file(path_img,path_fea)
path_training = my_own_path+"/annotation/db1"
path_training_1 = my_own_path+"/annotation/db2"
path_training_2 = my_own_path+"/annotation/db3"

path_result = my_own_path+"/result/db1"
path_result_1 = my_own_path+"/result/db2"
path_result_2 = my_own_path+"/result/db3"

# RENAME FILE IN STANDARD FORMAT
#rename_file(images_dataset_path,sub_name_1)
#rename_file(images_dataset_path,sub_name_2)


# EXTRACT FEATURE VGG16
#extract_fea_vgg16(path_fea,path_img)


# RANDOM TRAINING DATASET AND TEST DATASET
# random_file(path_training)
# random_file(path_training_1)
# random_file(path_training_2)


#Parameters have 4 elements: 
#         arr_test(X,Y) - X: number of samples, Y: features in sample (n,m)
#         arr_lbtest(X,Y) - X:number of samples, Y: Label of sample (n,1)
#         arr_train(X,Y)
#         arr_lbtrain(X,Y)
parameters = get_training_input(path_training_1,path_fea)
x_test,lbx_test,x_train,lbx_train = parameters
sklearn_linear_model(x_train,lbx_train,x_test,lbx_test,path_result_1)
sklearn_svm(x_train,lbx_train,x_test,lbx_test,"linear","linear",path_result_1)
sklearn_svm(x_train,lbx_train,x_test,lbx_test,"linear","rbf",path_result_1)
sklearn_svm(x_train,lbx_train,x_test,lbx_test,"linear","sigmoid",path_result_1)
sklearn_svm(x_train,lbx_train,x_test,lbx_test,"nonlinear","linear",path_result_1)

print("DONE")