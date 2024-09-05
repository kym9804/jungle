import warnings
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.linear_model import Lasso

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class RULPredictionModel:
    def plot_predictions(self,y_true, y_pred, title):
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label='True Values')
        plt.plot(y_pred, label='Predicted Values', linestyle='--')
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('RUL')
        plt.legend()
        plt.show()
    def __init__(self, train_path, test_path, rul_path):
        self.train_path = train_path
        self.test_path = test_path
        self.rul_path=rul_path
        self.column_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
    def apply_smoothing(data, window_size=5):
        return data.rolling(window=window_size, min_periods=1).mean()
    def load_data(self):
        # 데이터 로드
        self.df_train_FD001 = pd.read_csv(self.train_path, header=None)
        self.df_test_FD001 = pd.read_csv(self.test_path, header=None)
        self.df_RUL_FD001 = pd.read_csv(self.rul_path, header=None)
        self.df_train_FD001.columns = self.column_names
        self.df_test_FD001.columns = self.column_names
    def add_rul_column(self):
        max_cycle_per_id = self.df_train_FD001.groupby('id')['cycle'].max().reset_index()
        max_cycle_per_id.columns = ['id', 'max_cycle']
        self.df_train_FD001 = pd.merge(self.df_train_FD001, max_cycle_per_id, on='id')
        self.df_train_FD001['RUL'] = self.df_train_FD001['max_cycle'] - self.df_train_FD001['cycle']
        self.df_train_FD001 = self.df_train_FD001.drop(columns=['max_cycle'])
        self.df_RUL_FD001.columns = ['RUL']
        self.df_test_FD001 = self.df_test_FD001.groupby('id').last().reset_index()
        self.df_test_FD001['RUL'] = self.df_RUL_FD001['RUL']
    def preprocess_data_base(self,n_components=5):
        self.pre_df_train_base = self.df_train_FD001.drop(
            columns=['id', 'setting1', 'setting2', 'setting3'])
        self.pre_df_test_base = self.df_test_FD001.drop(
            columns=['id', 'setting1', 'setting2', 'setting3'])

        self.X_train_base = self.pre_df_train_base.drop(columns=['RUL'])
        self.y_train_base = self.pre_df_train_base['RUL'].clip(upper=125)
        self.X_test_base = self.pre_df_test_base.drop(columns=['RUL'])
        self.y_test_base = self.pre_df_test_base['RUL'].clip(upper=125)
        # # 성능 악화 문제로(overfitting) 제외
        # self.X_train_smoothed_strong = self.X_train_strong.rolling(window=5, min_periods=1).mean()
        # self.X_test_smoothed_strong = self.X_test_strong.rolling(window=5, min_periods=1).mean()

    def preprocess_data_weak(self,n_components=5):
        self.pre_df_train_weak = self.df_train_FD001.drop(
            columns=['id', 'setting1', 'setting2', 'setting3', 'sensor1', 'sensor5', 'sensor6', 'sensor9', 'sensor10',
                     'sensor14', 'sensor16', 'sensor18', 'sensor19'])
        self.pre_df_test_weak = self.df_test_FD001.drop(
            columns=['id', 'setting1', 'setting2', 'setting3', 'sensor1', 'sensor5', 'sensor6', 'sensor9', 'sensor10',
                     'sensor14', 'sensor16', 'sensor18', 'sensor19'])

        self.X_train_weak = self.pre_df_train_weak.drop(columns=['RUL'])
        self.y_train_weak = self.pre_df_train_weak['RUL'].clip(upper=125)
        self.X_test_weak = self.pre_df_test_weak.drop(columns=['RUL'])
        self.y_test_weak = self.pre_df_test_weak['RUL'].clip(upper=125)


        self.scaler = StandardScaler()
        self.X_train_scaled_weak = self.scaler.fit_transform(self.X_train_weak)
        self.X_test_scaled_weak = self.scaler.transform(self.X_test_weak)
        self.X_train_scaled_df_weak = pd.DataFrame(self.X_train_scaled_weak, columns=self.X_train_weak.columns)
        self.X_test_scaled_df_weak = pd.DataFrame(self.X_test_scaled_weak, columns=self.X_test_weak.columns)
        # PCA 적용
        pca = PCA(n_components=n_components)
        self.X_train_pca_weak = pca.fit_transform(self.X_train_scaled_df_weak)
        self.X_test_pca_weak = pca.transform(self.X_test_scaled_df_weak)

    def preprocess_data_strong(self, n_components=5):
        selected_columns = ['cycle','sensor2', 'sensor4', 'sensor7', 'sensor11', 'sensor12', 'sensor15', 'sensor17', 'sensor20',
                            'sensor21','RUL']

        self.pre_df_train_strong = self.df_train_FD001[selected_columns]
        self.pre_df_test_strong = self.df_test_FD001[selected_columns]

        self.X_train_strong = self.pre_df_train_strong.drop(columns=['RUL'])
        self.y_train_strong = self.pre_df_train_strong['RUL'].clip(upper=125)
        self.X_test_strong = self.pre_df_test_strong.drop(columns=['RUL'])
        self.y_test_strong = self.pre_df_test_strong['RUL'].clip(upper=125)
        # # 성능 악화 문제로(overfitting) 제외
        # self.X_train_smoothed_strong = self.X_train_strong.rolling(window=5, min_periods=1).mean()
        # self.X_test_smoothed_strong = self.X_test_strong.rolling(window=5, min_periods=1).mean()

        self.scaler = StandardScaler()
        self.X_train_scaled_strong = self.scaler.fit_transform(self.X_train_strong)
        self.X_test_scaled_strong = self.scaler.transform(self.X_test_strong)
        self.X_train_scaled_df_strong = pd.DataFrame(self.X_train_scaled_strong, columns=self.X_train_strong.columns)
        self.X_test_scaled_df_strong = pd.DataFrame(self.X_test_scaled_strong, columns=self.X_test_strong.columns)
        # PCA 적용
        pca = PCA(n_components=n_components)
        self.X_train_pca_strong = pca.fit_transform(self.X_train_scaled_df_strong)
        self.X_test_pca_strong = pca.transform(self.X_test_scaled_df_strong)

    def XGB_weak(self):
        def objective(trial):
            param = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'lambda': trial.suggest_float('lambda', 0, 5),
                'alpha': trial.suggest_float('alpha', 0, 5)
            }

            model = xgb.XGBRegressor(**param)
            model.fit(self.X_train_pca_weak, self.y_train_weak, eval_set=[(self.X_test_pca_weak, self.y_test_weak)], early_stopping_rounds=10, verbose=False)

            preds = model.predict(self.X_test_pca_weak)
            mse = mean_squared_error(self.y_test_weak, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적 하이퍼파라미터 출력
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        model = xgb.XGBRegressor(**best_params)
        model.fit(self.X_train_pca_weak, self.y_train_weak)

        # 예측
        y_pred_train = model.predict(self.X_train_pca_weak)
        y_pred_test = model.predict(self.X_test_pca_weak)

        # 평가
        mse_train = mean_squared_error(self.y_train_weak, y_pred_train)
        r2_train = r2_score(self.y_train_weak, y_pred_train)
        mse_test = mean_squared_error(self.y_test_weak, y_pred_test)
        r2_test = r2_score(self.y_test_weak, y_pred_test)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_pred_train[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_pred_test[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"PCA-XGB-weak - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"PCA-XGB-weak - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"PCA- XGB-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- XGB-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
        self.plot_predictions(self.y_test_weak, y_pred_test, 'PCA-XGB-weak Predictions vs True Values')

        def objective(trial):
            param = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'lambda': trial.suggest_float('lambda', 0, 5),
                'alpha': trial.suggest_float('alpha', 0, 5)
            }

            model = xgb.XGBRegressor(**param)
            model.fit(self.X_train_scaled_df_weak, self.y_train_weak, eval_set=[(self.X_test_scaled_df_weak, self.y_test_weak)], early_stopping_rounds=10, verbose=False)

            preds = model.predict(self.X_test_scaled_df_weak)
            mse = mean_squared_error(self.y_test_weak, preds)
            return mse
        # 최적 하이퍼파라미터 출력
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        model = xgb.XGBRegressor(**best_params)
        model.fit(self.X_train_scaled_df_weak, self.y_train_weak)

        # 예측
        y_pred_train = model.predict(self.X_train_scaled_df_weak)
        y_pred_test = model.predict(self.X_test_scaled_df_weak)

        # 평가
        mse_train = mean_squared_error(self.y_train_weak, y_pred_train)
        r2_train = r2_score(self.y_train_weak, y_pred_train)
        mse_test = mean_squared_error(self.y_test_weak, y_pred_test)
        r2_test = r2_score(self.y_test_weak, y_pred_test)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_pred_train[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_pred_test[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"No PCA-XGB-weak - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"No PCA-XGB-weak - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"No PCA- XGB-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- XGB-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
        # 중요도 시각화
        self.plot_predictions(self.y_test_weak, y_pred_test, 'No PCA-XGB-weak Predictions vs True Values')
    def RF_reg_weak(self):
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }

            model = RandomForestRegressor(**param, random_state=42)
            model.fit(self.X_train_pca_weak, self.y_train_weak)

            preds = model.predict(self.X_test_pca_weak)
            mse = mean_squared_error(self.y_test_weak, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적 하이퍼파라미터 출력
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        rf_model = RandomForestRegressor(**best_params, random_state=42)
        rf_model.fit(self.X_train_pca_weak, self.y_train_weak)

        # 예측
        y_pred_train_rf = rf_model.predict(self.X_train_pca_weak)
        y_pred_test_rf = rf_model.predict(self.X_test_pca_weak)

        # 평가
        mse_train_rf = mean_squared_error(self.y_train_weak, y_pred_train_rf)
        r2_train_rf = r2_score(self.y_train_weak, y_pred_train_rf)
        mse_test_rf = mean_squared_error(self.y_test_weak, y_pred_test_rf)
        r2_test_rf = r2_score(self.y_test_weak, y_pred_test_rf)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_pred_train_rf[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_pred_test_rf[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"PCA - Random Forest-weak - Train MSE: {mse_train_rf}, Train R^2: {r2_train_rf}")
        print(f"PCA - Random Forest-weak - Test MSE: {mse_test_rf}, Test R^2: {r2_test_rf}")
        print(f"PCA- Random Forest-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Random Forest-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
        self.plot_predictions(self.y_test_weak, y_pred_test_rf, 'PCA-Random Forest-weak Predictions vs True Values')
        # 중요도 시각화

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }

            model = RandomForestRegressor(**param, random_state=42)
            model.fit(self.X_train_scaled_df_weak, self.y_train_weak)

            preds = model.predict(self.X_test_scaled_df_weak)
            mse = mean_squared_error(self.y_test_weak, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적 하이퍼파라미터 출력
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        rf_model = RandomForestRegressor(**best_params, random_state=42)
        rf_model.fit(self.X_train_scaled_df_weak, self.y_train_weak)

        # 예측
        y_pred_train_rf = rf_model.predict(self.X_train_scaled_df_weak)
        y_pred_test_rf = rf_model.predict(self.X_test_scaled_df_weak)

        # 평가
        mse_train_rf = mean_squared_error(self.y_train_weak, y_pred_train_rf)
        r2_train_rf = r2_score(self.y_train_weak, y_pred_train_rf)
        mse_test_rf = mean_squared_error(self.y_test_weak, y_pred_test_rf)
        r2_test_rf = r2_score(self.y_test_weak, y_pred_test_rf)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_pred_train_rf[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_pred_test_rf[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"No PCA - Random Forest-weak - Train MSE: {mse_train_rf}, Train R^2: {r2_train_rf}")
        print(f"No PCA - Random Forest-weak - Test MSE: {mse_test_rf}, Test R^2: {r2_test_rf}")
        print(f"No PCA- Random Forest-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Random Forest-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

        # 중요도 시각화
        self.plot_predictions(self.y_test_weak, y_pred_test_rf, 'No PCA-Random Forest-weak Predictions vs True Values')

    def Ln_reg_weak(self):
        def objective(trial):
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(self.X_train_pca_weak, self.y_train_weak)
            y_pred = model.predict(self.X_train_pca_weak)
            mse = mean_squared_error(self.y_train_weak, y_pred)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = LinearRegression(**best_params)
        model.fit(self.X_train_pca_weak, self.y_train_weak)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_pca_weak)
        y_test_pred = model.predict(self.X_test_pca_weak)

        # MSE와 R^2 계산
        mse_train_lr = mean_squared_error(self.y_train_weak, y_train_pred)
        r2_train_lr = r2_score(self.y_train_weak, y_train_pred)
        mse_test_lr = mean_squared_error(self.y_test_weak, y_test_pred)
        r2_test_lr = r2_score(self.y_test_weak, y_test_pred)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"PCA - Linear Regression-weak - Train MSE: {mse_train_lr}, Train R^2: {r2_train_lr}")
        print(f"PCA - Linear Regression-weak - Test MSE: {mse_test_lr}, Test R^2: {r2_test_lr}")
        print(f"PCA- Linear Regression-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Linear Regression-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

        def objective(trial):
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(self.X_train_scaled_df_weak, self.y_train_weak)
            y_pred = model.predict(self.X_train_scaled_df_weak)
            mse = mean_squared_error(self.y_train_weak, y_pred)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = LinearRegression(**best_params)
        model.fit(self.X_train_scaled_df_weak, self.y_train_weak)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_scaled_df_weak)
        y_test_pred = model.predict(self.X_test_scaled_df_weak)

        # MSE와 R^2 계산
        mse_train_lr = mean_squared_error(self.y_train_weak, y_train_pred)
        r2_train_lr = r2_score(self.y_train_weak, y_train_pred)
        mse_test_lr = mean_squared_error(self.y_test_weak, y_test_pred)
        r2_test_lr = r2_score(self.y_test_weak, y_test_pred)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"No PCA - Linear Regression-weak - Train MSE: {mse_train_lr}, Train R^2: {r2_train_lr}")
        print(f"No PCA - Linear Regression-weak - Test MSE: {mse_test_lr}, Test R^2: {r2_test_lr}")
        print(f"No PCA- Linear Regression-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Linear Regression-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
    def Rd_reg_weak(self):
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.1, 10.0)
            model = Ridge(alpha=alpha)
            model.fit(self.X_train_pca_weak, self.y_train_weak)
            y_pred = model.predict(self.X_train_pca_weak)
            mse = mean_squared_error(self.y_train_weak, y_pred)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        print("Best parameters:", best_params)

        # 최적 파라미터로 모델 훈련 및 평가
        best_model = Ridge(**best_params)
        best_model.fit(self.X_train_pca_weak, self.y_train_weak)

        y_train_pred = best_model.predict(self.X_train_pca_weak)
        mse_train = mean_squared_error(self.y_train_weak, y_train_pred)
        r2_train = r2_score(self.y_train_weak, y_train_pred)

        y_test_pred = best_model.predict(self.X_test_pca_weak)
        mse_test = mean_squared_error(self.y_test_weak, y_test_pred)
        r2_test = r2_score(self.y_test_weak, y_test_pred)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"PCA- Ridge-weak - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"PCA- Ridge-weak - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"PCA- Ridge-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Ridge-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.1, 10.0)
            model = Ridge(alpha=alpha)
            model.fit(self.X_train_scaled_df_weak, self.y_train_weak)
            y_pred = model.predict(self.X_train_scaled_df_weak)
            mse = mean_squared_error(self.y_train_weak, y_pred)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        print("Best parameters:", best_params)

        # 최적 파라미터로 모델 훈련 및 평가
        best_model = Ridge(**best_params)
        best_model.fit(self.X_train_scaled_df_weak, self.y_train_weak)

        y_train_pred = best_model.predict(self.X_train_scaled_df_weak)
        mse_train = mean_squared_error(self.y_train_weak, y_train_pred)
        r2_train = r2_score(self.y_train_weak, y_train_pred)

        y_test_pred = best_model.predict(self.X_test_scaled_df_weak)
        mse_test = mean_squared_error(self.y_test_weak, y_test_pred)
        r2_test = r2_score(self.y_test_weak, y_test_pred)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"No PCA- Ridge-weak - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"NO PCA- Ridge-weak - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"NO PCA- Ridge-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Ridge-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

    def Rs_reg_weak(self):
        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            model = Lasso(**param, random_state=42)
            model.fit(self.X_train_pca_weak, self.y_train_weak)

            preds = model.predict(self.X_test_pca_weak)
            mse = mean_squared_error(self.y_test_weak, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = Lasso(**best_params, random_state=42)
        model.fit(self.X_train_pca_weak, self.y_train_weak)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_pca_weak)
        y_test_pred = model.predict(self.X_test_pca_weak)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_weak, y_train_pred)
        r2_train = r2_score(self.y_train_weak, y_train_pred)
        mse_test = mean_squared_error(self.y_test_weak, y_test_pred)
        r2_test = r2_score(self.y_test_weak, y_test_pred)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"PCA- Rasso-weak - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"PCA- Rasso-weak - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"PCA- Rasso-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Rasso-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            from sklearn.linear_model import Lasso
            model = Lasso(**param, random_state=42)
            model.fit(self.X_train_scaled_weak, self.y_train_weak)

            preds = model.predict(self.X_test_scaled_weak)
            mse = mean_squared_error(self.y_test_weak, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = Lasso(**best_params, random_state=42)
        model.fit(self.X_train_scaled_df_weak, self.y_train_weak)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_scaled_df_weak)
        y_test_pred = model.predict(self.X_test_scaled_df_weak)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_weak, y_train_pred)
        r2_train = r2_score(self.y_train_weak, y_train_pred)
        mse_test = mean_squared_error(self.y_test_weak, y_test_pred)
        r2_test = r2_score(self.y_test_weak, y_test_pred)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"No PCA- Rasso-weak - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"No PCA- Rasso-weak - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"No PCA- Rasso-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Rasso-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

    def Els_reg_weak(self):
        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            model = ElasticNet(**param, random_state=42)
            model.fit(self.X_train_pca_weak, self.y_train_weak)

            preds = model.predict(self.X_test_pca_weak)
            mse = mean_squared_error(self.y_test_weak, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        print("Best parameters:", best_params)

        model = ElasticNet(**best_params, random_state=42)
        model.fit(self.X_train_pca_weak, self.y_train_weak)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_pca_weak)
        y_test_pred = model.predict(self.X_test_pca_weak)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_weak, y_train_pred)
        r2_train = r2_score(self.y_train_weak, y_train_pred)
        mse_test = mean_squared_error(self.y_test_weak, y_test_pred)
        r2_test = r2_score(self.y_test_weak, y_test_pred)
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]
        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"PCA- Elasticnet-weak - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"PCA- Elasticnet-weak - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"PCA- Elasticnet-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Elasticnet-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            model = ElasticNet(**param, random_state=42)
            model.fit(self.X_train_scaled_df_weak, self.y_train_weak)

            preds = model.predict(self.X_test_scaled_df_weak)
            mse = mean_squared_error(self.y_test_weak, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        print("Best parameters:", best_params)

        model = ElasticNet(**best_params, random_state=42)
        model.fit(self.X_train_scaled_df_weak, self.y_train_weak)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_scaled_df_weak)
        y_test_pred = model.predict(self.X_test_scaled_df_weak)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_weak, y_train_pred)
        r2_train = r2_score(self.y_train_weak, y_train_pred)
        mse_test = mean_squared_error(self.y_test_weak, y_test_pred)
        r2_test = r2_score(self.y_test_weak, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_weak != 0
        train_error_ratio = np.abs(self.y_train_weak[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_weak[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_weak != 0
        test_error_ratio = np.abs(self.y_test_weak[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_weak[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)



        print(f"PCA- Elasticnet-weak - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"No PCA- Elasticnet-weak - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"No PCA- Elasticnet-weak - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Elasticnet-weak - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

    def XGB_strong(self):
        def objective(trial):
            param = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'lambda': trial.suggest_float('lambda', 0, 5),
                'alpha': trial.suggest_float('alpha', 0, 5)
            }

            model = xgb.XGBRegressor(**param)
            model.fit(self.X_train_pca_strong, self.y_train_strong,
                      eval_set=[(self.X_test_pca_strong, self.y_test_strong)], early_stopping_rounds=10, verbose=False)

            preds = model.predict(self.X_test_pca_strong)
            mse = mean_squared_error(self.y_test_strong, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적 하이퍼파라미터 출력
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        model = xgb.XGBRegressor(**best_params)
        model.fit(self.X_train_pca_strong, self.y_train_strong)

        # 예측
        y_pred_train = model.predict(self.X_train_pca_strong)
        y_pred_test = model.predict(self.X_test_pca_strong)

        # 평가
        mse_train = mean_squared_error(self.y_train_strong, y_pred_train)
        r2_train = r2_score(self.y_train_strong, y_pred_train)
        mse_test = mean_squared_error(self.y_test_strong, y_pred_test)
        r2_test = r2_score(self.y_test_strong, y_pred_test)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(
            self.y_train_strong[non_zero_indices_train] - y_pred_train[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_pred_test[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"PCA-XGB-strong - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"PCA-XGB-strong - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"PCA- XGB-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- XGB-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
        self.plot_predictions(self.y_test_strong, y_pred_test, 'PCA-XGB-strong Predictions vs True Values')

        def objective(trial):
            param = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'lambda': trial.suggest_float('lambda', 0, 5),
                'alpha': trial.suggest_float('alpha', 0, 5)
            }

            model = xgb.XGBRegressor(**param)
            model.fit(self.X_train_scaled_df_strong, self.y_train_strong,
                      eval_set=[(self.X_test_scaled_df_strong, self.y_test_strong)], early_stopping_rounds=10, verbose=False)

            preds = model.predict(self.X_test_scaled_strong)
            mse = mean_squared_error(self.y_test_strong, preds)
            return mse
        # 최적 하이퍼파라미터 출력
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        model = xgb.XGBRegressor(**best_params)
        model.fit(self.X_train_scaled_df_strong, self.y_train_strong)

        # 예측
        y_pred_train = model.predict(self.X_train_scaled_df_strong)
        y_pred_test = model.predict(self.X_test_scaled_df_strong)

        # 평가
        mse_train = mean_squared_error(self.y_train_strong, y_pred_train)
        r2_train = r2_score(self.y_train_strong, y_pred_train)
        mse_test = mean_squared_error(self.y_test_strong, y_pred_test)
        r2_test = r2_score(self.y_test_strong, y_pred_test)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(
            self.y_train_strong[non_zero_indices_train] - y_pred_train[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_pred_test[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"No PCA-XGB-strong - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"No PCA-XGB-strong - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"No PCA- XGB-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- XGB-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
        self.plot_predictions(self.y_test_strong, y_pred_test, 'No PCA-XGB-strong Predictions vs True Values')


    def RF_reg_strong(self):
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }

            model = RandomForestRegressor(**param, random_state=42)
            model.fit(self.X_train_pca_strong, self.y_train_strong)

            preds = model.predict(self.X_test_pca_strong)
            mse = mean_squared_error(self.y_test_strong, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적 하이퍼파라미터 출력
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        rf_model = RandomForestRegressor(**best_params, random_state=42)
        rf_model.fit(self.X_train_pca_strong, self.y_train_strong)

        # 예측
        y_pred_train_rf = rf_model.predict(self.X_train_pca_strong)
        y_pred_test_rf = rf_model.predict(self.X_test_pca_strong)

        # 평가
        mse_train_rf = mean_squared_error(self.y_train_strong, y_pred_train_rf)
        r2_train_rf = r2_score(self.y_train_strong, y_pred_train_rf)
        mse_test_rf = mean_squared_error(self.y_test_strong, y_pred_test_rf)
        r2_test_rf = r2_score(self.y_test_strong, y_pred_test_rf)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(
            self.y_train_strong[non_zero_indices_train] - y_pred_train_rf[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_pred_test_rf[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"PCA - Random Forest-strong - Train MSE: {mse_train_rf}, Train R^2: {r2_train_rf}")
        print(f"PCA - Random Forest-strong - Test MSE: {mse_test_rf}, Test R^2: {r2_test_rf}")
        print(f"PCA- Random Forest-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Random Forest-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
        self.plot_predictions(self.y_test_strong, y_pred_test_rf, 'PCA- Random Forest-strong Predictions vs True Values')


        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }

            model = RandomForestRegressor(**param, random_state=42)
            model.fit(self.X_train_scaled_df_strong, self.y_train_strong)

            preds = model.predict(self.X_test_scaled_df_strong)
            mse = mean_squared_error(self.y_test_strong, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적 하이퍼파라미터 출력
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        rf_model = RandomForestRegressor(**best_params, random_state=42)
        rf_model.fit(self.X_train_scaled_df_strong, self.y_train_strong)

        # 예측
        y_pred_train_rf = rf_model.predict(self.X_train_scaled_df_strong)
        y_pred_test_rf = rf_model.predict(self.X_test_scaled_df_strong)

        # 평가
        mse_train_rf = mean_squared_error(self.y_train_strong, y_pred_train_rf)
        r2_train_rf = r2_score(self.y_train_strong, y_pred_train_rf)
        mse_test_rf = mean_squared_error(self.y_test_strong, y_pred_test_rf)
        r2_test_rf = r2_score(self.y_test_strong, y_pred_test_rf)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(self.y_train_strong[non_zero_indices_train] - y_pred_train_rf[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_pred_test_rf[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"No PCA - Random Forest-strong - Train MSE: {mse_train_rf}, Train R^2: {r2_train_rf}")
        print(f"No PCA - Random Forest-strong - Test MSE: {mse_test_rf}, Test R^2: {r2_test_rf}")
        print(f"No PCA- Random Forest-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Random Forest-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
        self.plot_predictions(self.y_test_strong, y_pred_test_rf, 'No PCA- Random Forest-strong Predictions vs True Values')

    def Ln_reg_strong(self):
        def objective(trial):
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(self.X_train_pca_strong, self.y_train_strong)
            y_pred = model.predict(self.X_train_pca_strong)
            mse = mean_squared_error(self.y_train_strong, y_pred)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = LinearRegression(**best_params)
        model.fit(self.X_train_pca_strong, self.y_train_strong)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_pca_strong)
        y_test_pred = model.predict(self.X_test_pca_strong)

        # MSE와 R^2 계산
        mse_train_lr = mean_squared_error(self.y_train_strong, y_train_pred)
        r2_train_lr = r2_score(self.y_train_strong, y_train_pred)
        mse_test_lr = mean_squared_error(self.y_test_strong, y_test_pred)
        r2_test_lr = r2_score(self.y_test_strong, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(self.y_train_strong[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"PCA - Linear Regression-strong - Train MSE: {mse_train_lr}, Train R^2: {r2_train_lr}")
        print(f"PCA - Linear Regression-strong - Test MSE: {mse_test_lr}, Test R^2: {r2_test_lr}")
        print(f"PCA- Linear Regression-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Linear Regression-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        def objective(trial):
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(self.X_train_scaled_df_strong, self.y_train_strong)
            y_pred = model.predict(self.X_test_scaled_df_strong)
            mse = mean_squared_error(self.y_train_strong, y_pred)
            return mse
        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = LinearRegression(**best_params)
        model.fit(self.X_train_scaled_df_strong, self.y_train_strong)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_scaled_strong)
        y_test_pred = model.predict(self.X_test_scaled_df_strong)

        # MSE와 R^2 계산
        mse_train_lr = mean_squared_error(self.y_train_strong, y_train_pred)
        r2_train_lr = r2_score(self.y_train_strong, y_train_pred)
        mse_test_lr = mean_squared_error(self.y_test_strong, y_test_pred)
        r2_test_lr = r2_score(self.y_test_strong, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(self.y_train_strong[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"No PCA - Linear Regression-strong - Train MSE: {mse_train_lr}, Train R^2: {r2_train_lr}")
        print(f"No PCA - Linear Regression-strong - Test MSE: {mse_test_lr}, Test R^2: {r2_test_lr}")
        print(f"No PCA- Linear Regression-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Linear Regression-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

    def Rd_reg_strong(self):
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.1, 10.0)
            model = Ridge(alpha=alpha)
            model.fit(self.X_train_pca_strong, self.y_train_strong)
            y_pred = model.predict(self.X_train_pca_strong)
            mse = mean_squared_error(self.y_train_strong, y_pred)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        print("Best parameters:", best_params)

        # 최적 파라미터로 모델 훈련 및 평가
        best_model = Ridge(**best_params)
        best_model.fit(self.X_train_pca_strong, self.y_train_strong)

        y_train_pred = best_model.predict(self.X_train_pca_strong)
        mse_train = mean_squared_error(self.y_train_strong, y_train_pred)
        r2_train = r2_score(self.y_train_strong, y_train_pred)

        y_test_pred = best_model.predict(self.X_test_pca_strong)
        mse_test = mean_squared_error(self.y_test_strong, y_test_pred)
        r2_test = r2_score(self.y_test_strong, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(self.y_train_strong[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"PCA- Ridge-strong - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"PCA- Ridge-strong - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"PCA- Ridge-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Ridge-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.1, 10.0)
            model = Ridge(alpha=alpha)
            model.fit(self.X_train_scaled_df_strong, self.y_train_strong)
            y_pred = model.predict(self.X_train_scaled_df_strong)
            mse = mean_squared_error(self.y_train_strong, y_pred)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        print("Best parameters:", best_params)

        # 최적 파라미터로 모델 훈련 및 평가
        best_model = Ridge(**best_params)
        best_model.fit(self.X_train_scaled_df_strong, self.y_train_strong)

        y_train_pred = best_model.predict(self.X_train_scaled_df_strong)
        mse_train = mean_squared_error(self.y_train_strong, y_train_pred)
        r2_train = r2_score(self.y_train_strong, y_train_pred)

        y_test_pred = best_model.predict(self.X_test_scaled_df_strong)
        mse_test = mean_squared_error(self.y_test_strong, y_test_pred)
        r2_test = r2_score(self.y_test_strong, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(self.y_train_strong[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"No PCA- Ridge-strong - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"NO PCA- Ridge-strong - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"NO PCA- Ridge-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Ridge-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

    def Rs_reg_strong(self):
        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            model = Lasso(**param, random_state=42)
            model.fit(self.X_train_pca_strong, self.y_train_strong)

            preds = model.predict(self.X_test_pca_strong)
            mse = mean_squared_error(self.y_test_strong, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = Lasso(**best_params, random_state=42)
        model.fit(self.X_train_pca_strong, self.y_train_strong)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_pca_strong)
        y_test_pred = model.predict(self.X_test_pca_strong)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_strong, y_train_pred)
        r2_train = r2_score(self.y_train_strong, y_train_pred)
        mse_test = mean_squared_error(self.y_test_strong, y_test_pred)
        r2_test = r2_score(self.y_test_strong, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(self.y_train_strong[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"PCA- Rasso-strong - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"PCA- Rasso-strong - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"PCA- Rasso-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Rasso-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            from sklearn.linear_model import Lasso
            model = Lasso(**param, random_state=42)
            model.fit(self.X_train_scaled_strong, self.y_train_strong)

            preds = model.predict(self.X_test_scaled_strong)
            mse = mean_squared_error(self.y_test_strong, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = Lasso(**best_params, random_state=42)
        model.fit(self.X_train_scaled_df_strong, self.y_train_strong)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_scaled_df_strong)
        y_test_pred = model.predict(self.X_test_scaled_df_strong)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_strong, y_train_pred)
        r2_train = r2_score(self.y_train_strong, y_train_pred)
        mse_test = mean_squared_error(self.y_test_strong, y_test_pred)
        r2_test = r2_score(self.y_test_strong, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(self.y_train_strong[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"No PCA- Rasso-strong - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"No PCA- Rasso-strong - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"No PCA- Rasso-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Rasso-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

    def Els_reg_strong(self):
        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            model = ElasticNet(**param, random_state=42)
            model.fit(self.X_train_pca_strong, self.y_train_strong)

            preds = model.predict(self.X_test_pca_strong)
            mse = mean_squared_error(self.y_test_strong, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        print("Best parameters:", best_params)

        model = ElasticNet(**best_params, random_state=42)
        model.fit(self.X_train_pca_strong, self.y_train_strong)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_pca_strong)
        y_test_pred = model.predict(self.X_test_pca_strong)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_strong, y_train_pred)
        r2_train = r2_score(self.y_train_strong, y_train_pred)
        mse_test = mean_squared_error(self.y_test_strong, y_test_pred)
        r2_test = r2_score(self.y_test_strong, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(self.y_train_strong[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"PCA- Elasticnet-strong - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"PCA- Elasticnet-strong - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"PCA- Elasticnet-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"PCA- Elasticnet-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            model = ElasticNet(**param, random_state=42)
            model.fit(self.X_train_scaled_df_strong, self.y_train_strong)

            preds = model.predict(self.X_test_scaled_df_strong)
            mse = mean_squared_error(self.y_test_strong, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        print("Best parameters:", best_params)

        model = ElasticNet(**best_params, random_state=42)
        model.fit(self.X_train_scaled_df_strong, self.y_train_strong)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_scaled_df_strong)
        y_test_pred = model.predict(self.X_test_scaled_df_strong)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_strong, y_train_pred)
        r2_train = r2_score(self.y_train_strong, y_train_pred)
        mse_test = mean_squared_error(self.y_test_strong, y_test_pred)
        r2_test = r2_score(self.y_test_strong, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_strong != 0
        train_error_ratio = np.abs(self.y_train_strong[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_strong[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_strong != 0
        test_error_ratio = np.abs(self.y_test_strong[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_strong[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"PCA- Elasticnet-strong - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"No PCA- Elasticnet-strong - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"No PCA- Elasticnet-strong - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"No PCA- Elasticnet-strong - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

    def XGB_base(self):
        def objective(trial):
            param = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'lambda': trial.suggest_float('lambda', 0, 5),
                'alpha': trial.suggest_float('alpha', 0, 5)
            }

            model = xgb.XGBRegressor(**param)
            model.fit(self.X_train_base, self.y_train_base, eval_set=[(self.X_test_base, self.y_test_base)],
                      early_stopping_rounds=10, verbose=False)

            preds = model.predict(self.X_test_base)
            mse = mean_squared_error(self.y_test_base, preds)
            return mse

        # 최적 하이퍼파라미터 출력
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        model = xgb.XGBRegressor(**best_params)
        model.fit(self.X_train_base, self.y_train_base)

        # 예측
        y_pred_train = model.predict(self.X_train_base)
        y_pred_test = model.predict(self.X_test_base)

        # 평가
        mse_train = mean_squared_error(self.y_train_base, y_pred_train)
        r2_train = r2_score(self.y_train_base, y_pred_train)
        mse_test = mean_squared_error(self.y_test_base, y_pred_test)
        r2_test = r2_score(self.y_test_base, y_pred_test)
        non_zero_indices_train = self.y_train_base != 0
        train_error_ratio = np.abs(self.y_train_base[non_zero_indices_train] - y_pred_train[non_zero_indices_train]) / \
                            self.y_train_base[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_base != 0
        test_error_ratio = np.abs(self.y_test_base[non_zero_indices_test] - y_pred_test[non_zero_indices_test]) / \
                           self.y_test_base[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"XGB-base - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"XGB-base - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"XGB-base - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"XGB-base - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")
        self.plot_predictions(self.y_test_base, y_pred_test, 'XGB-base Predictions vs True Values')


    def RF_reg_base(self):
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }

            model = RandomForestRegressor(**param, random_state=42)
            model.fit(self.X_train_base, self.y_train_base)

            preds = model.predict(self.X_test_base)
            mse = mean_squared_error(self.y_test_base, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적 하이퍼파라미터 출력
        print(f"Best parameters: {study.best_params}")

        # 최적 하이퍼파라미터로 모델 학습 및 평가
        best_params = study.best_params
        rf_model = RandomForestRegressor(**best_params, random_state=42)
        rf_model.fit(self.X_train_base, self.y_train_base)

        # 예측
        y_pred_train_rf = rf_model.predict(self.X_train_base)
        y_pred_test_rf = rf_model.predict(self.X_test_base)

        # 평가
        mse_train_rf = mean_squared_error(self.y_train_base, y_pred_train_rf)
        r2_train_rf = r2_score(self.y_train_base, y_pred_train_rf)
        mse_test_rf = mean_squared_error(self.y_test_base, y_pred_test_rf)
        r2_test_rf = r2_score(self.y_test_base, y_pred_test_rf)
        non_zero_indices_train = self.y_train_base != 0
        train_error_ratio = np.abs(
            self.y_train_base[non_zero_indices_train] - y_pred_train_rf[non_zero_indices_train]) / \
                            self.y_train_base[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_base != 0
        test_error_ratio = np.abs(self.y_test_base[non_zero_indices_test] - y_pred_test_rf[non_zero_indices_test]) / \
                           self.y_test_base[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"Random Forest-base - Train MSE: {mse_train_rf}, Train R^2: {r2_train_rf}")
        print(f" Random Forest-base - Test MSE: {mse_test_rf}, Test R^2: {r2_test_rf}")
        print(f"Random Forest-base - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"Random Forest-base - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

        self.plot_predictions(self.y_test_base, y_pred_test_rf, 'Random Forest-base Predictions vs True Values')

    def Ln_reg_base(self):
        def objective(trial):
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(self.X_train_base, self.y_train_base)
            y_pred = model.predict(self.X_train_base)
            mse = mean_squared_error(self.y_train_base, y_pred)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = LinearRegression(**best_params)
        model.fit(self.X_train_base, self.y_train_base)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_base)
        y_test_pred = model.predict(self.X_test_base)

        # MSE와 R^2 계산
        mse_train_lr = mean_squared_error(self.y_train_base, y_train_pred)
        r2_train_lr = r2_score(self.y_train_base, y_train_pred)
        mse_test_lr = mean_squared_error(self.y_test_base, y_test_pred)
        r2_test_lr = r2_score(self.y_test_base, y_test_pred)
        non_zero_indices_train = self.y_train_base != 0
        train_error_ratio = np.abs(self.y_train_base[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_base[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_base != 0
        test_error_ratio = np.abs(self.y_test_base[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_base[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"Linear Regression-base - Train MSE: {mse_train_lr}, Train R^2: {r2_train_lr}")
        print(f"Linear Regression-base - Test MSE: {mse_test_lr}, Test R^2: {r2_test_lr}")
        print(f"Linear Regression-base - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"Linear Regression-base - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")


    def Rd_reg_base(self):
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.1, 10.0)
            model = Ridge(alpha=alpha)
            model.fit(self.X_train_base, self.y_train_base)
            y_pred = model.predict(self.X_train_base)
            mse = mean_squared_error(self.y_train_base, y_pred)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        print("Best parameters:", best_params)

        # 최적 파라미터로 모델 훈련 및 평가
        best_model = Ridge(**best_params)
        best_model.fit(self.X_train_base, self.y_train_base)

        y_train_pred = best_model.predict(self.X_train_base)
        mse_train = mean_squared_error(self.y_train_base, y_train_pred)
        r2_train = r2_score(self.y_train_base, y_train_pred)

        y_test_pred = best_model.predict(self.X_test_base)
        mse_test = mean_squared_error(self.y_test_base, y_test_pred)
        r2_test = r2_score(self.y_test_base, y_test_pred)
        non_zero_indices_train = self.y_train_base != 0
        train_error_ratio = np.abs(self.y_train_base[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_base[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_base != 0
        test_error_ratio = np.abs(self.y_test_base[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_base[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)
        print(f"Ridge-base - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"Ridge-base - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"Ridge-base - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"Ridge-base - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

    def Rs_reg_base(self):
        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            from sklearn.linear_model import Lasso
            model = Lasso(**param, random_state=42)
            model.fit(self.X_train_base, self.y_train_base)

            preds = model.predict(self.X_test_base)
            mse = mean_squared_error(self.y_test_base, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        model = Lasso(**best_params, random_state=42)
        model.fit(self.X_train_base, self.y_train_base)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_base)
        y_test_pred = model.predict(self.X_test_base)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_base, y_train_pred)
        r2_train = r2_score(self.y_train_base, y_train_pred)
        mse_test = mean_squared_error(self.y_test_base, y_test_pred)
        r2_test = r2_score(self.y_test_base, y_test_pred)
        non_zero_indices_train = self.y_train_base != 0
        train_error_ratio = np.abs(self.y_train_base[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_base[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_base != 0
        test_error_ratio = np.abs(self.y_test_base[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_base[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"Rasso-base - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"Rasso-base - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"Rasso-base - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"Rasso-base - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")

    def Els_reg_base(self):
        def objective(trial):
            param = {
                'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }

            model = ElasticNet(**param, random_state=42)
            model.fit(self.X_train_base, self.y_train_base)

            preds = model.predict(self.X_test_base)
            mse = mean_squared_error(self.y_test_base, preds)
            return mse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        print("Best parameters:", best_params)

        model = ElasticNet(**best_params, random_state=42)
        model.fit(self.X_train_base, self.y_train_base)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train_base)
        y_test_pred = model.predict(self.X_test_base)

        # MSE와 R^2 계산
        mse_train = mean_squared_error(self.y_train_base, y_train_pred)
        r2_train = r2_score(self.y_train_base, y_train_pred)
        mse_test = mean_squared_error(self.y_test_base, y_test_pred)
        r2_test = r2_score(self.y_test_base, y_test_pred)
        # Train error ratio 계산
        non_zero_indices_train = self.y_train_base != 0
        train_error_ratio = np.abs(self.y_train_base[non_zero_indices_train] - y_train_pred[non_zero_indices_train]) / \
                            self.y_train_base[non_zero_indices_train]

        # Test error ratio 계산
        non_zero_indices_test = self.y_test_base != 0
        test_error_ratio = np.abs(self.y_test_base[non_zero_indices_test] - y_test_pred[non_zero_indices_test]) / \
                           self.y_test_base[non_zero_indices_test]

        # 평균 오차 비율
        mean_train_error_ratio = np.mean(train_error_ratio)
        mean_test_error_ratio = np.mean(test_error_ratio)

        print(f"Elasticnet-base - Train MSE: {mse_train}, Train R^2: {r2_train}")
        print(f"Elasticnet-base - Test MSE: {mse_test}, Test R^2: {r2_test}")
        print(f"Elasticnet-base - Mean Train Error Ratio: {mean_train_error_ratio:.2f}")
        print(f"Elasticnet-base - Mean Test Error Ratio: {mean_test_error_ratio:.2f}")


rul_cal= RULPredictionModel('train_FD001.csv','test_FD001.csv','RUL_FD001.csv')
rul_cal.load_data()
rul_cal.add_rul_column()
rul_cal.preprocess_data_weak()
rul_cal.preprocess_data_strong()
rul_cal.preprocess_data_base()
rul_cal.XGB_base()
rul_cal.XGB_weak()
rul_cal.XGB_strong()
rul_cal.RF_reg_base()
rul_cal.RF_reg_weak()
rul_cal.RF_reg_strong()
