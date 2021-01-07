from libraries import *
from params import *

class AutoML(object):

    X_train, X_test, y_train, y_test = None, None, None, None

    def __init__(self, df, target, type, dummy = True, device_type = "CPU"):
        self.df = df
        self.cols = df.columns.tolist()
        self.target = target
        self.type = type
        self.cat_vars = []
        self.model = None
        self.device_type = device_type
        self.dummy = dummy

    def __repr__(self):
        return repr(self.df.head())

    def class_dist(self):
        plt.figure(figsize=(10,8))
        ax = sns.countplot(y = self.target, data = self.df, orient = 'v')
        plt.title("Distribution of classes")

        total = len(self.df[self.target])
        for p in ax.patches:
            percentage = '{:.5f}%'.format(100 * p.get_width()/total)
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height()/2
            ax.annotate(percentage, (x,y))

        # plt.show()

    def extract_cols(self):  
        cat_variables = [var for var in self.cols if self.df[var].dtype == 'O']
        self.cat_vars = [s for s in cat_variables if s != self.target]


    def class_splitting(self):
        self.cols = [cols for cols in self.cols if cols != self.target]
        num_variables = [var for var in self.df.columns if self.df[var].dtype != 'O']
        # print("Numeric Columns : ", num_variables)
        num_vars = [s for s in num_variables if s != self.target]
        
        for cols in num_vars:
            self.df[cols].fillna(value=self.df[cols].mean(), inplace=True)
        self.df.dropna(inplace=True)
        
        X = self.df[self.cols]
        y = self.df[self.target]

        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size= 0.3, random_state=42)

        for col in self.cat_vars:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        if self.dummy:
            for col in self.cat_vars:
                dummy = pd.get_dummies(pd.Series(X_train[col].astype(str)), prefix=col+"_", drop_first=True)
                dummy.index = X_train.index
                X_train = pd.concat([X_train, dummy], axis=1, sort=False)

                dummy = pd.get_dummies(pd.Series(X_test[col].astype(str)), prefix=col+"_", drop_first=True)
                dummy.index = X_test.index
                X_test = pd.concat([X_test, dummy], axis=1, sort=False)

            X_train.drop(columns=self.cat_vars, inplace=True, axis=1)
            X_test.drop(columns=self.cat_vars, inplace=True, axis=1)

            missedCol = list(set(X_train.columns) - set(X_train.columns).intersection(set(X_test.columns)))
            for col in missedCol:
                X_test[col] = [0] * len(X_test)

            X_train, X_test = X_train.align(X_test, join='left', axis=1)

        self.lb = LabelEncoder()
        y_train = self.lb.fit_transform(y_train)
        y_test = self.lb.transform(y_test)

        return X_train, X_test, y_train, y_test, self.classes, self.lb

    def reg_splitting(self):
        self.cols = [cols for cols in self.cols if cols != self.target]
        num_variables = [var for var in self.df.columns if self.df[var].dtype != 'O']
        print("Numeric Columns : ", num_variables)
        num_vars = [s for s in num_variables if s != self.target]
        
        for cols in num_vars:
            self.df[cols].fillna(value=self.df[cols].mean(), inplace=True)
        self.df.dropna(inplace=True)
        
        X = self.df[self.cols]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

        for col in self.cat_vars:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        if self.dummy:
            for col in self.cat_vars:
                dummy = pd.get_dummies(pd.Series(X_train[col].astype(str)), prefix=col+"_", drop_first=True)
                dummy.index = X_train.index
                X_train = pd.concat([X_train, dummy], axis=1, sort=False)

                dummy = pd.get_dummies(pd.Series(X_test[col].astype(str)), prefix=col+"_", drop_first=True)
                dummy.index = X_test.index
                X_test = pd.concat([X_test, dummy], axis=1, sort=False)

            X_train.drop(columns=self.cat_vars, inplace=True, axis=1)
            X_test.drop(columns=self.cat_vars, inplace=True, axis=1)

            missedCol = list(set(X_train.columns) - set(X_train.columns).intersection(set(X_test.columns)))
            for col in missedCol:
                X_test[col] = [0] * len(X_test)

            X_train, X_test = X_train.align(X_test, join='left', axis=1)

        return X_train, X_test, y_train, y_test

    def roc_auc_metric(self):
        if self.n_classes == 2:
            probs = self.model.predict_proba(self.X_test)

            ns_probs = [0 for _ in range(len(self.y_test))]
            ns_auc = roc_auc_score(self.y_test, ns_probs)

            ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
            ls_fpr, ls_tpr, _ = roc_curve(self.y_test, probs[:,1])

            plt.figure(figsize=(10,8))
            plt.plot(ns_fpr, ns_tpr, linestyle="--", label="Random Prediction")
            plt.plot(ls_fpr, ls_tpr, linestyle="--", label="Binary Modeling")

            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            plt.show()

        else:
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(self.n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            colors = cycle(["blue", "green", "red", "magenta", "cyan", "yellow", "black", "#eeefff", "0.55"])
            plt.figure(figsize=(10,8))

            for i, color, cls in zip(range(self.n_classes), colors, classes):
                plt.plot(fpr[i], tpr[i], linewidth=3, color=color, label="ROC Curve of class{0} (area={1:0.2f})".format(cls, roc_auc[i]))

            plt.plot([0,1], [0,1], 'k--')
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])

            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC for Multi-Class data")
            plt.legend(loc="lower right")
            plt.show()

    def thresh_col(self):
        counts = {}
        col_names = []
        col_vals = []
        for cols in self.cat_vars:
            col_names.append(cols)
            col_vals.append(df[cols].nunique())

        counts = dict(zip(col_names, col_vals))
        keymax = max(counts, key=counts.get)
        return keymax

    def set_threshold_downsample(self):
        if self.df[self.target].nunique() == 2:
            threshold = min(1.5 * min(self.df[self.target].value_counts()), max(self.df[self.target].value_counts()))
            print(threshold)
            return int(threshold)

    def undersample(self, dataframe, column, basis_col, count, sample_count):
        
        above = pd.DataFrame()
        below = pd.DataFrame()
        data = pd.DataFrame()
        df = dataframe.copy(deep=True)
        col = column
        
        max_val = df[basis_col].value_counts().head(1).values[0]
        n = min(max_val, df[basis_col].value_counts().min())
        stratified = df.groupby(basis_col).apply(lambda x: x.sample(n))
        stratified.index = stratified.index.droplevel(0)
        
        above_count = pd.DataFrame(df[col].value_counts() > count)
        below_count = pd.DataFrame(df[col].value_counts() < count)
        
        to_sample = above_count[above_count[col] == True].index.tolist()
        not_sample = below_count[below_count[col] == True].index.tolist()
        
        for val in to_sample: # Sample with the given sample_count value 
            above = above.append(df[df[col] == val].sample(n = sample_count, 
                                                        replace = True,
                                                        random_state=1))
        
        for val in not_sample:
            below = below.append(df[df[col] == val]) # Without Sampling for values less than given count
            
        data = pd.concat([above, stratified, below])
        return data

    def plot_local_shap(self, expected_value, shap_values, X_test, index_to_plot, feature_names, 
                        matplotlib=True, figsize=(15,3), link='identity', text_rotation=90, save=True,
                        filename='"shap_local_plot.png'):

                        f = plt.figure()

                        cols = self.X_train.columns
                        pred = self.model.predict(self.X_test.iloc[index_to_plot:index_to_plot+1, :])[0]

                        feature_names_ext = []

                        for i,j in zip(feature_names, X_test.iloc[index_to_plot, :]):
                            temp = i + " = " + str(j)
                            feature_names_ext.append(temp)

                        print(feature_names_ext)
                        shap.plots.waterfall_plot(expected_value,
                                            shap_values = shap_values[index_to_plot],
                                            features = self.X_test[index_to_plot],
                                            feature_names=feature_names_ext,
                                            max_display=15)
                        f.savefig(filename, bbox_inches='tight', dpi=600)



    def catboost_class_cv(self, params, param_names):

        params = dict(zip(param_names, params))
        
        self.extract_cols() # Extract the cat_features

        if self.df[self.target].nunique() == 2:
            down_threshold = self.set_threshold_downsample()
            self.df = self.undersample(self.df, self.target, self.thresh_col(), down_threshold, down_threshold)
            # classify.class_dist()

        self.X_train, self.X_test, self.y_train, self.y_test, self.classes , self.lb = self.class_splitting()

        if len(self.classes) == 2:
            self.objective = "CrossEntropy"
        else:
            self.objective = "MultiClass"

        params['loss_function'] = self.objective
        params['eval_metric'] = 'Accuracy'

        if self.dummy:
            cv_dataset = Pool(data=self.X_train,
                            label = self.y_train)
        else:
            cv_dataset = Pool(data=self.X_train,
                            label = self.y_train,
                            cat_features = self.cat_vars)


        scores = cv(cv_dataset, params, fold_count=3, iterations=200, verbose=100, early_stopping_rounds=50)

        return - 1 * scores['test-Accuracy-mean'].max()


    def classification(self):

        param_names = ['depth', 'learning_rate',
                       'bagging_temperature', 'l2_leaf_reg',
                       'border_count']

        optimization_func = partial(self.catboost_class_cv,
                                    param_names = param_names)
        
        result = gp_minimize(optimization_func,
                             dimensions = hyperparameter_space,
                             n_random_starts = 3,
                             n_jobs=-1,
                             n_calls= 5,
                             verbose=True,
                             random_state=42)

        params = dict(zip(param_names, result.x))
        print(params)

        self.model = CatBoostClassifier(
            iterations=200,
            task_type=self.device_type,
            bagging_temperature=params["bagging_temperature"],
            learning_rate=params['learning_rate'],
            depth=int(params['depth']),
            l2_leaf_reg=params['l2_leaf_reg'],
            border_count = int(params["border_count"]),
            random_seed=42,
            verbose=200,
            thread_count=-1,
            loss_function=self.objective,
            eval_metric='Accuracy'
        )

        if self.dummy:
            self.model.fit(self.X_train, self.y_train,
                        early_stopping_rounds=50, 
                        plot=False)
        else:
            self.model.fit(self.X_train, self.y_train, 
                        cat_features=self.cat_vars, 
                        early_stopping_rounds=50, 
                        plot=False)

        pred = self.model.predict(self.X_test)
        proba = self.model.predict_proba(self.X_test)

        importances = pd.DataFrame(self.model.get_feature_importance(prettified=True), columns = ["Feature Id", "Importances"])
        print()
        print(importances.head(20))
        print()
        print("Most Important Feature:", importances["Feature Id"][0])

        if self.n_classes > 2:
            multi_class = 'ovo'
            roc_value = roc_auc_score(self.y_test, proba[: , 1], multi_class=multi_class)

        else:
            multi_class = 'raise'
            roc_value = roc_auc_score(self.y_test, proba[: , 1])
        
        print("The ROC-AUC of the model is:", roc_value)
        print("Accuracy of the model is :", accuracy_score(self.y_test, pred))
        #self.roc_auc_metric()

        if self.n_classes > 2:
            if self.dummy:
                shap_values = self.model.get_feature_importance(Pool(self.X_test, self.y_test), type='ShapValues')
            else:
                shap_values = self.model.get_feature_importance(Pool(self.X_test, self.y_test, cat_features = self.cat_vars), type='ShapValues')
            original_shape = shap_values.shape
            print("Original Shap Shape:", original_shape)

            print("Original SHAP values", shap_values.head())

            shap_values_reshaped = shap_values.transpose(original_shape[1], original_shape[0], original_shape[-1]) #(1,0,2) # (samples, classes, features)
            shap_values = shap_values_reshaped[:, :, :-1] # (samples, features)
            expected_value = shap_values[:, -1][0]

            shap.summary_plot(list(shap_values), features = self.X_train, class_names = self.classes, plot_type='bar')

        else:
            if self.dummy:
                shap_values = self.model.get_feature_importance(Pool(self.X_test, self.y_test), type='ShapValues')
            else:
                shap_values = self.model.get_feature_importance(Pool(self.X_test, self.y_test, cat_features = self.cat_vars), type='ShapValues')
            
            expected_value = shap_values[:, -1][0]
            shap_values = shap_values[:, :-1]
            shap.summary_plot(shap_values, features = self.X_test, class_names = self.classes.tolist(), plot_type='bar')

        index_to_plot = np.random.randint(0, self.X_test.shape[0]-2)
        print("Plotting Index :", index_to_plot)
        self.plot_local_shap(expected_value, shap_values, self.X_test, index_to_plot, self.X_train.columns)

        print("Classification Model Finished Running")

    def catboost_reg_cv(self, params, param_names):
    
        params = dict(zip(param_names, params))
        
        self.extract_cols() # Extract the cat_features
        self.X_train, self.X_test, self.y_train, self.y_test = self.reg_splitting()

        params['loss_function'] = 'RMSE'

        if self.dummy:
            cv_dataset = Pool(data=self.X_train,
                            label = self.y_train)
        else:
            cv_dataset = Pool(data=self.X_train,
                            label = self.y_train,
                            cat_features = self.cat_vars)

        scores = cv(cv_dataset, params, fold_count=3, iterations=200, verbose=100, early_stopping_rounds=50)
        return scores['test-RMSE-mean'].max()


    def regression(self):
        print("Regression Initiated...!!!")
        param_names = ['depth', 'learning_rate', 'bagging_temperature']

        optimization_func = partial(self.catboost_reg_cv,
                                    param_names = param_names)
        
        result = gp_minimize(optimization_func,
                             dimensions = reg_hyperparameter_space,
                             n_random_starts = 3,
                             n_jobs=-1,
                             n_calls= 5,
                             verbose=True,
                             random_state=42)

        params = dict(zip(param_names, result.x))
        print(params)

        self.model = CatBoostRegressor(
            iterations=1000,
            task_type=self.device_type,
            bagging_temperature=params["bagging_temperature"],
            learning_rate=params['learning_rate'],
            depth=int(params['depth']),
            thread_count=-1,
            random_seed=42,
            verbose=100,
            loss_function='RMSE'
        )

        self.model.fit(self.X_train, self.y_train, 
                       cat_features=self.cat_vars, 
                       early_stopping_rounds=50)
        
        pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, pred))
        r2 = r2_score(self.y_test, pred)
        print("R2 Score is :", r2)
        print("RMSE :", rmse)
        print()

        importances = pd.DataFrame(self.model.get_feature_importance(prettified=True), columns = ["Feature Id", "Importances"])
        print()
        print(importances.head(20))
        print()
        print("Most Important Feature:", importances["Feature Id"][0])

        if self.dummy:
            shap_values = self.model.get_feature_importance(Pool(self.X_test, self.y_test), type = "ShapValues")
        else:
            shap_values = self.model.get_feature_importance(Pool(self.X_test, self.y_test, cat_features=self.cat_vars), type = "ShapValues")
        shap_values = shap_values[:,:-1]

        shap.summary_plot(shap_values, self.X_test)
        
        print("Regression Model Ran Successfully.....!!!")


if __name__ == "__main__":
    
    # start = time.time()
    # df = pd.read_csv(r"C:\Users\Abhishek\Downloads\adult.csv")
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # #df = pd.read_csv(r"C:\Users\Abhishek\Desktop\Git Projects\AutoML\data\WearableComputing.csv")
    # classify = AutoML(df, "income", "classification", "GPU")
    # #classify = AutoML(df, "classe", "classification", "GPU")
    # classify.class_dist()
    # classify.classification()

    # # df = pd.read_csv(r"C:\AutoML\data\housing.csv")
    # # reg = AutoML(df, "medv", "regression", "CPU")
    # # reg.regression()
    # stop = time.time()
    # total_time = (stop - start)

    # print(f"Time taken for process is {total_time/60} mins.")

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--type", type=str)
    parser.add_argument("--deviceType", type=str)

    args = parser.parse_args()
    df = pd.read_csv(args.path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if args.type == "classification":
        classify = AutoML(df, args.target, args.type, args.deviceType)
        # classify.class_dist()
        classify.classification()

    elif args.type == "regression":
        reg = AutoML(df, args.target, args.type, args.deviceType)
        reg.regression()

    else:
        ValueError("Invalid argument passed")

    stop = time.time()
    total_time = (stop - start)
    print(f"Time taken for process is {total_time/60} mins.")
    # python app.py --path C:\Users\Abhishek\Downloads\adult.csv --target income --type classification --deviceType GPU
    # python app.py --path C:\Users\Abhishek\Downloads\housing.csv --target medv --type regression --deviceType GPU
