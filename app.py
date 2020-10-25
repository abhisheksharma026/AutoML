from libraries import *
from params import *

class AutoML(object):

    X_train, X_test, y_train, y_test = None, None, None, None

    def __init__(self, df, target, type, device_type = "CPU"):
        self.df = df
        self.cols = df.columns.tolist()
        self.target = target
        self.type = type
        self.cat_vars = []
        self.model = None
        self.device_type = device_type

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

        plt.show()

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

        cv_dataset = Pool(data=self.X_train,
                          label = self.y_train,
                          cat_features = self.cat_vars)

        scores = cv(cv_dataset, params, fold_count=3, iterations=100, verbose=100, early_stopping_rounds=5)

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
                             n_calls= 7,
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
            verbose=100,
            loss_function=self.objective,
            eval_metric='Accuracy'
        )

        self.model.fit(self.X_train, self.y_train, cat_features=self.cat_vars, early_stopping_rounds=5)
        pred = self.model.predict(self.X_test)
        proba = self.model.predict_proba(self.X_test)

        if self.n_classes > 2:
            multi_class = 'ovo'
            roc_value = roc_auc_score(self.y_test, proba[: , 1], multi_class=multi_class)

        else:
            multi_class = 'raise'
            roc_value = roc_auc_score(self.y_test, proba[: , 1])
            print("The ROC-AUC of the model is:", roc_value)
            roc_value = roc_auc_score(self.y_test, proba[: , 1])
        
        print("The ROC-AUC of the model is:", roc_value)
        #self.roc_auc_metric()

        if self.n_classes > 2:
            shap_values = self.model.get_feature_importance(Pool(self.X_test, self.y_test, cat_features = self.cat_vars), type='ShapValues')
            original_shape = shap_values.shape
            print("Original Shape:", original_shape)
            shap_values_reshaped = shap_values.reshape(original_shape[1], original_shape[0], original_shape[-1])
            shap_values = shap_values_reshaped[:, :, :-1]

            shap.summary_plot(list(shap_values), features = self.X_train, class_names = self.classes, plot_type='bar')

        else:
            shap_values = self.model.get_feature_importance(Pool(self.X_test, self.y_test, cat_features = self.cat_vars), type='ShapValues')
            shap_values = shap_values[:, :-1]
            shap.summary_plot(shap_values, features = self.X_test, class_names = self.classes.tolist(), plot_type='bar')

        print("Model Finished Running")

    def catboost_reg_cv(self, params, param_names):
    
        params = dict(zip(param_names, params))
        
        self.extract_cols() # Extract the cat_features
        self.X_train, self.X_test, self.y_train, self.y_test = self.reg_splitting()

        params['loss_function'] = 'RMSE'

        cv_dataset = Pool(data=self.X_train,
                          label = self.y_train,
                          cat_features = self.cat_vars)

        scores = cv(cv_dataset, params, fold_count=3, iterations=100, verbose=100, early_stopping_rounds=5)
        print(scores)
        return scores['test-RMSE-mean'].max()


    def regression(self):
        print("Regression Initiated...!!!")
        param_names = ['depth', 'learning_rate', 'bagging_temperature']

        optimization_func = partial(self.catboost_reg_cv,
                                    param_names = param_names)
        
        result = gp_minimize(optimization_func,
                             dimensions = reg_hyperparameter_space,
                             n_random_starts = 3,
                             n_calls= 7,
                             verbose=True,
                             random_state=42)

        params = dict(zip(param_names, result.x))
        print(params)

        self.model = CatBoostRegressor(
            iterations=200,
            task_type=self.device_type,
            bagging_temperature=params["bagging_temperature"],
            learning_rate=params['learning_rate'],
            depth=int(params['depth']),
            random_seed=42,
            verbose=100,
            loss_function='RMSE'
        )

        self.model.fit(self.X_train, self.y_train, cat_features=self.cat_vars, early_stopping_rounds=5)
        
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
        shap_values = self.model.get_feature_importance(Pool(self.X_test, self.y_test, cat_features=self.cat_vars), type = "ShapValues")
        shap_values = shap_values[:,:-1]

        shap.summary_plot(shap_values, self.X_test)
        
        print("Regression Model Ran Successfully.....!!!")


if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv(r"C:\AutoML\data\adult.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    #df = pd.read_csv(r"C:\Users\Abhishek\Desktop\Git Projects\AutoML\data\WearableComputing.csv")
    classify = AutoML(df, "income", "classification", "CPU")
    #classify = AutoML(df, "classe", "classification", "GPU")
    classify.class_dist()
    classify.classification()

    # df = pd.read_csv(r"C:\AutoML\data\housing.csv")
    # reg = AutoML(df, "medv", "regression", "CPU")
    # reg.regression()
    stop = time.time()
    total_time = (stop - start)

    print(f"Time taken for process is {total_time/60} mins.")

