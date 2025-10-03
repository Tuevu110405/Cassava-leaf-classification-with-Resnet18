# Cassava-leaf-classification-with-Resnet18

## Performance

### Performance of model trained with CrossEntropyLoss on test 

```
                precision   recall    f1-score   

         cbb       0.68      0.10      0.17       
        cbsd       0.64      0.84      0.73       
         cgm       0.75      0.37      0.49       
         cmd       0.81      0.93      0.87       
     healthy       0.58      0.48      0.52       

    accuracy                           0.74      
   macro avg       0.69      0.54      0.56      
weighted avg       0.73      0.74      0.70      
```

### Performance of model trained with FocalLoss on test 
```
                precision    recall  f1-score   

         cbb       0.49      0.52      0.50       
        cbsd       0.72      0.75      0.73       
         cgm       0.63      0.43      0.51       
         cmd       0.85      0.86      0.85       
     healthy       0.49      0.70      0.57       

    accuracy                           0.73      
   macro avg       0.64      0.65      0.63      
weighted avg       0.74      0.73      0.73      
```

### Performance of model trained with FocalLoss&WeigtLoss on test 

```
                precision   recall  f1-score   

         cbb       0.58      0.12      0.20       
        cbsd       0.55      0.78      0.65       
         cgm       0.36      0.06      0.10       
         cmd       0.72      0.91      0.80       
     healthy       0.80      0.08      0.14       

    accuracy                           0.65      
   macro avg       0.60      0.39      0.38      
weighted avg       0.62      0.65      0.58            
```