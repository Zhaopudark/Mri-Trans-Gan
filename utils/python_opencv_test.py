import cv2 
print(cv2.__version__)
img = cv2.imread("./Implementations/ali.png")
cv2.imshow("Image",img)
cv2.waitKey (0)  
cv2.destroyAllWindows()