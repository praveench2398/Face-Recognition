import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from face_recognition import load_image_file,face_locations,face_encodings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class Fr_functionality:
    
    def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        
        X = []
        y = []
        
        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                
                continue
            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                
                image = load_image_file(img_path)
                face_bounding_boxes =face_locations(image)
                
                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if                                          len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

       
        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
        return knn_clf
    
    def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
        
        if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
            
            raise Exception("Invalid image path: {}".format(X_img_path))

        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
        
        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        # Load image file and find face locations
        X_img =load_image_file(X_img_path)
        X_face_locations =face_locations(X_img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings =face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in                          zip(knn_clf.predict(faces_encodings),X_face_locations, are_matches)]
    
    
    def show_prediction_labels_on_image(img_path, predictions):
        pil_image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(pil_image)
        for name, (top, right, bottom, left) in predictions:
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            name = name.encode("UTF-8")
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        del draw
        plt.imshow(pil_image)

class Face_recognition( Fr_functionality):
    
    def __init__(self,Input_Image_file,model_save_path,val_Image_file):
        
        self.Input_Image_file=Input_Image_file
        
        self.model_save_path=model_save_path
        
        self.val_Image_file=val_Image_file
    
    def recognition(self):
        
        classifier=Fr_functionality.train(self.Input_Image_file,
                                         model_save_path=self.model_save_path+'trained_knn_model.clf',n_neighbors=2)
        print("Training complete!")
        
        # STEP 2: Using the trained classifier, make predictions for unknown images
        for image_file in os.listdir(self.val_Image_file):
            
            full_file_path = os.path.join(self.val_Image_file,image_file)
            
            print("Looking for faces in {}".format(image_file))
            
            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = Fr_functionality.predict(full_file_path, model_path=self.model_save_path+'trained_knn_model.clf')

            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                
                print("- Found {} at ({}, {})".format(name, left, top))
                
        Fr_functionality.show_prediction_labels_on_image(os.path.join(self.val_Image_file, image_file), predictions)
        
if __name__ == "__main__":
    
    Input_Image_file=input('Enter the Image file directory for training:')
       
    model_save_path=input('Enter the save path for model you are training:') 
    
    val_Image_file=input('Enter the Image file directory for validation:')
        
    a=Face_recognition(Input_Image_file,model_save_path,val_Image_file)
    
    b=a.recognition()
     
