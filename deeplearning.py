import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract as pt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import imutils
import tensorflow as tf


# LOAD YOLO MODEL
INPUT_WIDTH =  640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model = tf.keras.models.load_model('./models/network.h5')
model_w_h = tf.keras.models.load_model('./models/w_h_model.h5')
model_7_9 =  tf.keras.models.load_model('./models/7_9_model.h5')
model_y_v =  tf.keras.models.load_model('./models/v_y_model.h5')
#model = tf.keras.models.load_model('./static/models/network_k.h5')





def get_detections(img,net):
    # CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image,detections):
    # FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    # NMS
    index = np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)).flatten()
    
    return boxes_np, confidences_np, index


import os
def save_text(filename,text):
    name,ext = os.path.splitext(filename)
    with open('./static/predict/{}.txt'.format(name),mode='w') as f:
        f.write(text)
    f.close()

def save_roi(filename,text):
    name,ext = os.path.splitext(filename)
    with open('./static/roi/{}.txt'.format(name),mode='w') as f:
        f.write(text)
    f.close()


def save_preprocessing(filename,text):
    name,ext = os.path.splitext(filename)
    with open('./static/preprocessing/{}.txt'.format(name),mode='w') as f:
        f.write(text)
    f.close()


def save_segmentation(filename,text):
    name,ext = os.path.splitext(filename)
    with open('./static/segmentation/{}.txt'.format(name),mode='w') as f:
        f.write(text)
    f.close()






def extract_text(image,bbox,path,filename):


    img = np.array(load_img(path))

    x,y,w,h = bbox
    
    roi = image[y:y+h, x:x+w]
    if 0 in roi.shape:
        return ''
    else:

        #aqui

        """ 
        roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY)
        magic_color = apply_brightness_contrast(gray,brightness=40,contrast=70)
        #text = pt.image_to_string(magic_color)
         """
        

        
      #ANTERIOR METODO  #plate_number= reconocimiento_caracteres(roi,path,filename)
        plate_number = reconocimiento_caracteres_othermodels(roi,path,filename)
        #text = pt.image_to_string(magic_color,lang='eng',config='--psm 6')
        text = plate_number.strip()


        roi_or = img[y:y+h,x:x+w]
        roi_bgr_or = cv2.cvtColor(roi_or,cv2.COLOR_RGB2BGR)
        gray_or = cv2.cvtColor(roi_bgr_or,cv2.COLOR_BGR2GRAY)

        #guardando ROI, roi en imagen original, la otra imagen tiene un preprocesamiento 
        #al enviarlo a pt.image_to_string
        cv2.imwrite('./static/roi/{}'.format(filename),roi_bgr_or)
        #cv2.imwrite('./static/preprocessed/{}'.format(filename),roi_bgr)
        #print(text)
        save_text(filename,text)
        save_roi(filename,text)
        save_preprocessing(filename,text)
        save_segmentation(filename,text)

        
        return text

def drawings(image,boxes_np,confidences_np,index,path,filename):

    
    #debemos tener un listado de la siguiente forma

    #cada elemento del objeto 

    ''' 
    placa = {
      "number_plate": "P234GHT",
      "X": 12,
      "Y": 13,
      "W": 124
      "H": 250
    }
    '''   

    text_list_json=[]


    # drawings
    valor = 1
    text_list = []
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'placa: {:.0f}%'.format(bb_conf*100)
         #filename_n = str(ind)+'_'+ filename; #varias placas al mismo tiempo



        filename_n = str(valor) +'_'+filename;
        valor = valor + 1
        license_text = extract_text(image,boxes_np[ind],path,filename_n)

        resultado = {
             "license_plate": license_text,
             "x": x,
             "y": y,
             "w":w,
             "h":h
             # retornar tambien score
             }

        text_list_json.append(resultado)




        # convert into JSON:
     #   y = json.dumps(resultado)
        # the result is a JSON string:
        #print(y)
        


          #bgr
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),5)  #bounding box placa
        cv2.rectangle(image,(x,y-30),(x+w,y),(0,0,255),-5)  # prob. bounding box  -1 
        cv2.rectangle(image,(x,y+h+20),(x+w,y+h+30+20),(0,0,0),-1)   # número leido ()



     #ESTA SECCIÒN DE CÓDIGO SERVIRA PARA ANOTAR EL NÙMERO CORRESPONDIENTE (AÙN EN PROCESO)

        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)   # texto prob.
        cv2.putText(image,license_text,(x+10,y+h+37),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)  # texto número leìdo


   
        
        text_list.append(license_text)



    return image,  text_list , text_list_json


# predictions
def yolo_predictions(img,net,path,filename):
    ## step-1: detections
    input_image, detections = get_detections(img,net)
    ## step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    ## step-3: Drawings
    result_img, text, list_json = drawings(img,boxes_np,confidences_np,index,path,filename) # add path and filenaem
    return result_img, text , list_json


def object_detection(path,filename):
    # read image
    image = cv2.imread(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    result_img, text_list, list_json = yolo_predictions(image,net,path,filename)  # add path and filenaem
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)
    return text_list, list_json

#OCR MODELO PROVISIONAL

def OCR(path,filename):
    img = np.array(load_img(path))
    cods = object_detection(path,filename)
    xmin ,xmax,ymin,ymax = cods[0]
    roi = img[ymin:ymax,xmin:xmax]
    roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY)
   # magic_color = apply_brightness_contrast(gray,brightness=40,contrast=70)
    cv2.imwrite('./static/roi/{}'.format(filename),roi_bgr)
    print(text)
    save_text(filename,text)
    return text

"""
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf
"""


#/////////////////////////////////////////////////////////////////////////////
#MODELO OCR 



#preprocesamiento
def preprocesamiento_img(img,path,filename):
  #gray->blur->adaptive thresold -> inversion -> dilatation(op)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (3, 3), 0)
  adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
  invertion = 255 - adaptive

  cv2.imwrite('./static/preprocessing/{}'.format(filename),invertion)

  #imagen = np.array(load_img(path))

  #dilation = cv2.dilate(edges, np.ones((3,3)))  

##############guardar localmente esta imagen
  return gray, invertion


def preprocesamiento_img_thresh(img,path,filename):
  #gray->blur->adaptive thresold -> inversion -> dilatation(op)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (3, 3), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

  cv2.imwrite('./static/preprocessing/{}'.format(filename),thresh)


  return gray, thresh


#obtener contornos (ayuda para posterior bounding boxes)
def find_contours(img):
  conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  conts = imutils.grab_contours(conts)
  conts = sort_contours(conts, method = 'left-to-right')[0]
  return conts

#conts = find_contours(invertion.copy())


#segmentar caracteres

def show_segmentation(img,conts,gray,path,filename):
  min_w, max_w = 4, 160
  min_h, max_h = 14, 140
  #para remover contornos innecesarios (lineas horizontales)
  alto_imagen = img.shape[0]
  ancho_imagen = img.shape[1]

  max_w= int(ancho_imagen/6)
  max_h = int(alto_imagen)

  img_copy = img.copy()
  for c in conts:
  #print(c)
    (x, y, w, h) = cv2.boundingRect(c)
  #print(x, y, w, h)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
      roi = gray[y:y+h, x:x+w]
    #cv2_imshow(roi)
      thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      #cv2_imshow(thresh)
      cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0,0,255), 1) #segmentación de caracteres


  cv2.imwrite('./static/segmentation/{}'.format(filename),img_copy)

    #cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 100, 0), 2)
  #plt.imshow(img_copy)
  #cv2.imshow("segmentation", img_copy)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()

  ##############guardar localmente esta imagen

#procesa los caracteres detecados

def extract_roi(img,x ,y,w,h, margin=2 ):
  roi = img[y - margin:y + h, x - margin:x + w + margin]
  return roi

def thresholding(img):
  thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  return thresh

def extract_roi_o(img,x,y,w,h): #
  roi = img[y:y + h, x:x + w]
  return roi


def resize_img(img, w, h):
  if w > h:
    resized = imutils.resize(img, width = 28)
  else:
    resized = imutils.resize(img, height = 28)

  (h, w) = resized.shape
  dX = int(max(0, 28 - w) / 2.0)
  dY = int(max(0, 28 - h) / 2.0)

  filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT, value = (0,0,0))
  filled = cv2.resize(filled, (28,28))
  return filled

def normalization(img):
  img = img.astype('float32') / 255.0
  img = np.expand_dims(img, axis = -1)
  return img



def process_box(gray, x, y, w, h):
  roi = extract_roi_o(gray,x,y,w,h)
  #cv2_imshow(roi), aqui se debe guardar la imagen para w y para 7 
  thresh = thresholding(roi)
  (h, w) = thresh.shape
  resized = resize_img(thresh, w, h)
  #plt.imshow(resized)
  normalized = normalization(resized)
  return normalized, x, y , w, h


def process_box_no_return(gray, x, y, w, h,filename,path,character):

  #path = 'C:/Users/user/Documents/proyecto_matricula/YOLOv5_OCR/web_app/static//or_binary/7_9/roi/'+ format(filename) 
  #roi = extract_roi(gray,x,y,w,h)
  roi = extract_roi_o(gray,x,y,w,h)
  cv2.imwrite('./static/for_binary/all_letters/roi/{}'.format(filename),roi)

  character_r=character
  


  if(character== '7' or character == '9'):
      ruta='./static/for_binary/7_9/roi/'+format(filename)
      cv2.imwrite(ruta,roi)
      img7_9= cv2.imread(ruta)

     # cv2.imwrite('./static/for_binary/7_9/roi/{}'.format(filename),roi)
    #  img7_9 = cv2.imread(path)
    #  #nuevo_character = (img7_9)
      character_r = diferenciar_7_9(img7_9)
      #print(character_r)
      

  
  if(character == 'H'):   # if(character== 'W' or character == 'H'):

      ruta='./static/for_binary/W_H/roi/'+format(filename)
      cv2.imwrite(ruta,roi)
      imgH_W= cv2.imread(ruta)
      character_r = diferenciar_H_W(imgH_W)
      #print(character_r)

  if(character== 'V'):  #  if(character== 'V' or character == 'Y'):


      ruta='./static/for_binary/V_Y/roi/'+format(filename)
      cv2.imwrite(ruta,roi)
      imgV_Y= cv2.imread(ruta)
      character_r = diferenciar_V_Y(imgV_Y)



  #cv2_imshow(roi)
  thresh = thresholding(roi)
  cv2.imwrite('./static/for_binary/all_letters/thresh/{}'.format(filename),thresh)


  if(character== '7' or character == '9'):
      cv2.imwrite('./static/for_binary/7_9/thresh/{}'.format(filename),thresh)

  
  if(character == 'H'):
      cv2.imwrite('./static/for_binary/W_H/thresh/{}'.format(filename),thresh)

  
     
   #   ruta='./static/for_binary/W_H/thresh/'+format(filename)
   #   cv2.imwrite(ruta,thresh)
   #   imgH_W= cv2.imread(ruta)
   #   character_r = diferenciar_H_W(imgH_W)

  

  
  if(character== 'V'):  
      cv2.imwrite('./static/for_binary/V_Y/thresh/{}'.format(filename),thresh)


  #cv2_imshow(thresh)
 # thresh = cv2.dilate(thresh, np.ones((3,3))) #this
  (h, w) = thresh.shape
  resized = resize_img(thresh, w, h)
  cv2.imwrite('./static/for_binary/all_letters/resize/{}'.format(filename),resized)

  if(character== '7' or character == '9'):
      cv2.imwrite('./static/for_binary/7_9/resize/{}'.format(filename),resized)

  
  if(character == 'H'):
      cv2.imwrite('./static/for_binary/W_H/resize/{}'.format(filename),resized)
      

      ''' 

      ruta='./static/for_binary/W_H/resize/'+format(filename)
      cv2.imwrite(ruta,resized)
      imgH_W= cv2.imread(ruta)
      character_r = diferenciar_H_W(imgH_W)
      '''

  

  if(character== 'V'):
      cv2.imwrite('./static/for_binary/V_Y/resize/{}'.format(filename),resized)



 
  #cv2_imshow(resized)
  normalized = normalization(resized)
  return character_r

  
 





def procesando_caracteres_detectados(img,path,filename):
  #print('Imagen de Placa:')
  #plt.imshow(img)
  characters = []
  gray, invertion = preprocesamiento_img(img,path,filename)
  #print('\n')
  #print('Resultado de preprocesamiento de imagen:')
  #plt.imshow(invertion)
  conts = find_contours(invertion.copy())

  #print('\n')
  #print('Resultado de segmentación de caracteres:')
  show_segmentation(img,conts,gray,path,filename)

  min_w, max_w = 4, 160
  min_h, max_h = 14, 140
  #print('\n')
  #print('Caracteres segmentados que se envian al modelo:')
  for c in conts:
  #print(c)
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
      normalized, xn,yn,wn,hn = process_box(gray, x, y, w, h)
      characters.append((normalized, (xn, yn, wn, hn)))


  boxes = [box[1] for box in characters]
  pixels = np.array([pixel[0] for pixel in characters], dtype = 'float32')

  return boxes, pixels , img

def procesando_caracteres_detectados_gray(img,path,filename):
  #print('Imagen de Placa:')
  #plt.imshow(img)
  characters = []
  gray, invertion = preprocesamiento_img_thresh(img,path,filename)
  #print('\n')
  #print('Resultado de preprocesamiento de imagen:')
  #plt.imshow(invertion)
  conts = find_contours(invertion.copy())

  #print('\n')
  #print('Resultado de segmentación de caracteres:')
  show_segmentation(img,conts,gray,path,filename)

  min_w, max_w = 4, 160
  min_h, max_h = 14, 140

  alto_imagen = img.shape[0]
  ancho_imagen = img.shape[1]

  max_w= int(ancho_imagen/6)
  max_h = int(alto_imagen)

  #print('\n')
  #print('Caracteres segmentados que se envian al modelo:')
  for c in conts:
  #print(c)
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
      normalized, xn,yn,wn,hn = process_box(gray, x, y, w, h)
      characters.append((normalized, (xn, yn, wn, hn)))


  boxes = [box[1] for box in characters]
  pixels = np.array([pixel[0] for pixel in characters], dtype = 'float32')

  return boxes, pixels , img , gray


def semantica(number_plate):
   placa_corregida=''
   n_letrainicial=''
   n_partenumerica=''
   n_parteconsonante=''
   if (len(number_plate)==7):

       
     #  print('\n')
     #  print('7caracteres')
       n_letrainicial1=number_plate[0]
       n_partenumerica1=number_plate[1:4]
       n_parteconsonante1=number_plate[4:7]

       #print('\n')
       #print(n_letrainicial)
       #str.isalpha().
       if (n_letrainicial1=='P'):
         n_letrainicial='P'
       elif (n_letrainicial1=='0'):
         n_letrainicial='O'
       elif (n_letrainicial1=='D'):
         n_letrainicial='O'
       elif not((n_letrainicial1=='C')or(n_letrainicial1=='O')or(n_letrainicial1=='A')or(n_letrainicial1=='M')):
         n_letrainicial='P'
       else:
         n_letrainicial=n_letrainicial1
       
      # print(n_letrainicial)
     #  print('+')

       for numero in n_partenumerica1 :
         #print(numero)
         if(numero.isnumeric()):
           n_partenumerica=n_partenumerica+numero
         else:
           if(numero=='I'):
             n_partenumerica=n_partenumerica+'1'
           elif(numero=='O'):
             n_partenumerica=n_partenumerica+'0'
           elif(numero=='D'):
             n_partenumerica=n_partenumerica+'0'
           elif(numero=='S'):
             n_partenumerica=n_partenumerica+'5'        
           elif(numero=='B'):
             n_partenumerica=n_partenumerica+'8'        
           elif(numero=='Z'):
             n_partenumerica=n_partenumerica+'2'
           elif(numero=='J'):
             n_partenumerica=n_partenumerica+'3'  #AGREGAR ESTO AL COLAB
           elif(numero=='G'):
             n_partenumerica=n_partenumerica+'6'  #AGREGAR ESTO AL COLAB
           elif(numero=='T'):                         
             n_partenumerica=n_partenumerica+'7' 
           else:
             n_partenumerica=n_partenumerica+numero
                

       #print(n_partenumerica)
      # print('+')

       for consonante in n_parteconsonante1 :
           if(consonante=='I'):
             n_parteconsonante=n_parteconsonante+'L'
           elif(consonante=='O'):
             n_parteconsonante=n_parteconsonante+'D'
           elif(consonante=='0'):
             n_parteconsonante=n_parteconsonante+'D'
           elif(consonante=='5'):
             n_parteconsonante=n_parteconsonante+'S'        
           elif(consonante=='8'):
             n_parteconsonante=n_parteconsonante+'B'        
           elif(consonante=='2'):
             n_parteconsonante=n_parteconsonante+'Z'
           elif(consonante=='U'):
             n_parteconsonante=n_parteconsonante+'V'
           elif(consonante=='E'):
             n_parteconsonante=n_parteconsonante+'F'
           elif(consonante=='3'):
             n_parteconsonante=n_parteconsonante+'J'  #AGREGAR ESTO AL COLALB
           elif(consonante=='6'):
             n_parteconsonante=n_parteconsonante+'G'  #AGREGAR ESTO AL COLALB
           elif(consonante=='7'):
             n_parteconsonante=n_parteconsonante+'T' 
           else:
             n_parteconsonante = n_parteconsonante +consonante

      # print(n_parteconsonante)

       placa_corregida = n_letrainicial + n_partenumerica+n_parteconsonante


        
      # print('\n')  
      # print('Placa corregida:'+placa_corregida)
       return placa_corregida

     
   else:
      # print('\n')
      # print('maomenos')
      # print('Placa sin corregir:'+number_plate)
       return number_plate


def reconocimiento_caracteres(img,path,filename):

  boxes,pixels,img = procesando_caracteres_detectados(img,path,filename)

  digits = '0123456789'
  letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  characters_list = digits + letters
  characters_list = [l for l in characters_list]


  predictions = model.predict(pixels) #para este punto debe estar el modelo importado en flask

  numberplate=''
  for (prediction, (x, y, w, h)) in zip(predictions, boxes):
    i = np.argmax(prediction)
    #print(i)
    probability = prediction[i]
    #print(probability)
    character = characters_list[i]
    #print(character)

    #  cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255,100,0), 2) #2
    # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0,0,255), 1) #2

    #dimensiones de imagen no permiten que se muestre el texto de cada bounding box
    #cv2.putText(img_copy, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)
    print(character, ' -> ', probability * 100)
    numberplate= numberplate + character

  #escribir en el filesystem el numero de placa, (preferiblemente en las 2 carpetas roi + prediction)
  numberplate = semantica(numberplate)

  return numberplate


def reconocimiento_caracteres_othermodels(img,path,filename):

  boxes,pixels,img,gray = procesando_caracteres_detectados_gray(img,path,filename)

  digits = '0123456789'
  letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  characters_list = digits + letters
  characters_list = [l for l in characters_list]


  predictions = model.predict(pixels) #para este punto debe estar el modelo importado en flask
  print('--------------------------------')
  caracter = 1
  numberplate=''
  for (prediction, (x, y, w, h)) in zip(predictions, boxes):
    i = np.argmax(prediction)
    #print(i)
    probability = prediction[i]
    #print(probability)
    character = characters_list[i]
   # process_box_no_return(gray, x, y, w, h)

    #print(character)

    #  cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255,100,0), 2) #2
    # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0,0,255), 1) #2

    #dimensiones de imagen no permiten que se muestre el texto de cada bounding box
    #cv2.putText(img_copy, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)
    print(character, ' -> ', probability * 100)

    filename_n = str(caracter) +'_'+filename;
    caracter = caracter + 1
    #print(path)
    
    #DEPENDIENDO DE LETRA QUE SE GUARDE, MANDAR A LOS DE CLASIFICACION BINARIA
    #REVISAR BIEN CUAL GUARDAR.
    character_mod = process_box_no_return(gray, x, y, w, h,filename_n,path,character)
    numberplate= numberplate + character_mod



  #escribir en el filesystem el numero de placa, (preferiblemente en las 2 carpetas roi + prediction)
  numberplate = semantica(numberplate)
  return numberplate



#MODELO PARA DIFERENCIAR 7 DE 9
def diferenciar_7_9(img_7_9):

 #   print(img_7_9.shape)
    img_7_9 = cv2.resize(img_7_9, (75,100))
    img_7_9 = img_7_9.astype('float32') / 255.0
    img_7_9 = np.reshape(img_7_9, (1,100,75,3))
    prediction=''
    result = model_7_9.predict(img_7_9)
    if result[0][0] >= 0.5:
        prediction = '9'
        probability = result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)
        
    else:
        prediction = '7'
        probability = 1 - result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)
    
    return prediction


#MODELO PARA DIFERENCIAR H DE W


def diferenciar_H_W(img_H_W):

   # print(img_H_W.shape)
    img_H_W = cv2.resize(img_H_W, (75,100))
    img_H_W = img_H_W.astype('float32') / 255.0
    img_H_W = np.reshape(img_H_W, (1,100,75,3))
    prediction=''
    result = model_w_h.predict(img_H_W)
    print(str(result[0][0]))
    if result[0][0] >= 0.05: #0.5  -> 0.1  -> 0.05
        prediction = 'W'
        probability = result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)
        
    else:
        prediction = 'H'
        probability = 1 - result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)
    
    return prediction


def diferenciar_V_Y(img_Y_V):
   # print(img_H_W.shape)
    img_Y_V = cv2.resize(img_Y_V, (75,100))
    img_Y_V = img_Y_V.astype('float32') / 255.0
    img_Y_V = np.reshape(img_Y_V, (1,100,75,3))
    prediction=''
    result = model_y_v.predict(img_Y_V)
    if result[0][0] >= 0.5:
        prediction = 'Y'
        probability = result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)
        
    else:
        prediction = 'V'
        probability = 1 - result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)
    
    return prediction




  #cv2_imshow(img_w)


def obtener_numero_placa(img):
  #revisar si se define abajo o en parametros , se enviara la ruta de la imagen
 # numero_placa= reconocimiento_caracteres(img)
  numero_placa= reconocimiento_caracteres2_corregir(img)
  print('\n')
  print('Placa:'+numero_placa)
  return numero_placa






def obtener_numero_placa(img):
  #revisar si se define abajo o en parametros , se enviara la ruta de la imagen
  numero_placa= reconocimiento_caracteres(img)
  print('\n')
  print('Placa:'+numero_placa)