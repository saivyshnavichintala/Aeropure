[1mdiff --git a/Regression.py b/Regression.py[m
[1mindex c8f3d23..99969c0 100644[m
[1m--- a/Regression.py[m
[1m+++ b/Regression.py[m
[36m@@ -4,14 +4,15 @@[m [mfrom sklearn.model_selection import train_test_split[m
 [m
 from sklearn.linear_model import ([m
     LinearRegression,[m
[32m+[m[32m    Lasso,[m
[32m+[m[32m    Ridge,[m
     LogisticRegression[m
 )[m
 [m
 from sklearn.metrics import ([m
     mean_squared_error,[m
     r2_score,[m
[31m-    accuracy_score,[m
[31m-    classification_report[m
[32m+[m[32m    accuracy_score[m
 )[m
 [m
 [m
[36m@@ -120,11 +121,92 @@[m [mdef run_regression(df):[m
         predictions[m
     )[m
 [m
[31m-    results["linear_mse"] = round(mse, 3)[m
[31m-    results["linear_r2"] = round(r2, 3)[m
[32m+[m[32m    results["linear_mse"] = round([m
[32m+[m[32m        mse,[m
[32m+[m[32m        3[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    results["linear_r2"] = round([m
[32m+[m[32m        r2,[m
[32m+[m[32m        3[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    # =================================================[m
[32m+[m[32m    # L1 REGRESSION - LASSO[m
[32m+[m[32m    # =================================================[m
[32m+[m
[32m+[m[32m    l1_model = Lasso([m
[32m+[m[32m        alpha=1.0[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    l1_model.fit([m
[32m+[m[32m        X_train,[m
[32m+[m[32m        y_train[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    predictions = l1_model.predict([m
[32m+[m[32m        X_test[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    mse = mean_squared_error([m
[32m+[m[32m        y_test,[m
[32m+[m[32m        predictions[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    r2 = r2_score([m
[32m+[m[32m        y_test,[m
[32m+[m[32m        predictions[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    results["l1_mse"] = round([m
[32m+[m[32m        mse,[m
[32m+[m[32m        3[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    results["l1_r2"] = round([m
[32m+[m[32m        r2,[m
[32m+[m[32m        3[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    # =================================================[m
[32m+[m[32m    # L2 REGRESSION - RIDGE[m
[32m+[m[32m    # =================================================[m
[32m+[m
[32m+[m[32m    l2_model = Ridge([m
[32m+[m[32m        alpha=1.0[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    l2_model.fit([m
[32m+[m[32m        X_train,[m
[32m+[m[32m        y_train[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    predictions = l2_model.predict([m
[32m+[m[32m        X_test[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    mse = mean_squared_error([m
[32m+[m[32m        y_test,[m
[32m+[m[32m        predictions[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    r2 = r2_score([m
[32m+[m[32m        y_test,[m
[32m+[m[32m        predictions[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    results["l2_mse"] = round([m
[32m+[m[32m        mse,[m
[32m+[m[32m        3[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    results["l2_r2"] = round([m
[32m+[m[32m        r2,[m
[32m+[m[32m        3[m
[32m+[m[32m    )[m
 [m
     # =================================================[m
[31m-    # LOGISTIC REGRESSION[m
[32m+[m[32m    # LOGISTIC REGRESSION - CLASSIFICATION[m
     # =================================================[m
 [m
     # Create binary air-quality class[m
