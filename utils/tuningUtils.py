from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from functools import partial
from datetime import datetime
import optuna
from skopt import BayesSearchCV


def log(mensagem: str, inicio: datetime = None) -> None:
    """
    Registra mensagens de log com marcação de tempo e duração opcional.

    Parameters:
        mensagem (str): A mensagem a ser exibida.
        inicio (datetime, opcional): O tempo inicial para calcular a duração.
    """
    fim = datetime.now()
    if inicio:
        duracao = fim - inicio
        horas, resto = divmod(duracao.total_seconds(), 3600)
        minutos, segundos = divmod(resto, 60)
        tempo_formatado = f"{int(horas):02}:{int(minutos):02}:{int(segundos):02}"
        print(f"----- {mensagem} concluído! Tempo de execução: {tempo_formatado} ----- {fim.strftime('%H:%M:%S')}\n")
    else:
        print(f"----- {mensagem} iniciado ----- {fim.strftime('%H:%M:%S')}\n")


def objective_skopt(model_class, search_space, X, y, n_iter: int, scoring: str = "accuracy") -> dict:
    """
    Executa a busca bayesiana (BayesSearchCV) para encontrar os melhores hiperparâmetros.

    Parameters:
        model_class: Classe do modelo (ex: RandomForestClassifier).
        search_space (dict): Espaço de busca dos hiperparâmetros.
        X: Conjunto de atributos de treino.
        y: Conjunto de rótulos de treino.
        n_iter (int): Número de iterações da busca.
        scoring (str): Métrica de avaliação (default = "accuracy").

    Returns:
        dict: Dicionário com os melhores hiperparâmetros encontrados.
    """
    try:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        model = model_class()
        search = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        search.fit(X, y)
        melhores_params = dict(search.best_params_)
        print(f"Melhores Hiperparâmetros (skopt): {melhores_params}\n")
        return melhores_params
    except Exception as e:
        print(f"Erro durante a execução do BayesSearchCV: {e}")
        raise


def objective_optuna(trial, model_class, model_name: str, search_spaces_optuna: dict,
                     X_train, y_train, X_val, y_val) -> float:
    """
    Função de objetivo para a otimização com Optuna.

    Parameters:
        trial: Objeto de tentativa do Optuna.
        model_class: Classe do modelo (ex: RandomForestClassifier).
        model_name (str): Nome do modelo.
        search_spaces_optuna (dict): Espaços de busca definidos para o Optuna.
        X_train, y_train: Dados de treino.
        X_val, y_val: Dados de validação.

    Returns:
        float: A acurácia do modelo com os parâmetros testados.
    """
    try:
        params = {
            param: func(trial) for param, func in search_spaces_optuna[model_name].items()
        }
        model = model_class(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)
    except Exception as e:
        print(f"Erro durante a execução do Optuna: {e}")
        raise


def tunar_modelo(model_name: str,
                 model_class,
                 X_treino,
                 y_treino,
                 n_iter: int,
                 search_spaces_optuna: dict = None,
                 search_spaces_skopt: dict = None,
                 metodo: str = "optuna") -> dict:
    """
    Executa a otimização de hiperparâmetros utilizando Optuna ou skopt.

    Parameters:
        model_name (str): Nome do modelo.
        model_class: Classe do modelo (ex: RandomForestClassifier).
        X_treino: Conjunto de atributos de treino.
        y_treino: Conjunto de rótulos de treino.
        n_iter (int): Número de iterações ou trials da busca.
        search_spaces_optuna (dict, opcional): Espaço de busca de hiperparâmetros para Optuna.
        search_spaces_skopt (dict, opcional): Espaço de busca de hiperparâmetros para skopt.
        metodo (str): Método de tuning a ser utilizado ("optuna" ou "skopt").

    Returns:
        dict: Dicionário com os melhores hiperparâmetros encontrados.
    """
    try:
        inicio_tuning = datetime.now()
        log(f"Tuning para {model_class.__name__} ({metodo})")

        if metodo == "optuna":
            if search_spaces_optuna is None or model_name not in search_spaces_optuna:
                raise ValueError(f"Hiperparâmetros para '{model_name}' não encontrados no espaço de busca.")

            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            estudos = []

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            for train_idx, val_idx in kf.split(X_treino):
                X_train, X_val = X_treino.iloc[train_idx], X_treino.iloc[val_idx]
                y_train, y_val = y_treino.iloc[train_idx], y_treino.iloc[val_idx]

                study = optuna.create_study(direction="maximize")
                study.optimize(partial(objective_optuna, model_class=model_class,
                                       search_spaces_optuna=search_spaces_optuna,
                                       model_name=model_name,
                                       X_train=X_train, y_train=y_train,
                                       X_val=X_val, y_val=y_val), n_trials=n_iter)
                estudos.append(study)

            best_study = max(estudos, key=lambda s: s.best_value)
            melhores_parametros = best_study.best_params
            print(f"Melhores Hiperparâmetros (optuna): {melhores_parametros}\n")
            log(f"Tuning para {model_class.__name__} ({metodo})", inicio=inicio_tuning)
            return melhores_parametros

        elif metodo == "skopt":
            if search_spaces_skopt is None:
                raise ValueError("Um espaço de busca deve ser fornecido para o método 'skopt'.")

            melhores_parametros = objective_skopt(
                model_class=model_class,
                search_space=search_spaces_skopt,
                X=X_treino,
                y=y_treino,
                n_iter=n_iter
            )
            log(f"Tuning para {model_class.__name__} ({metodo})", inicio=inicio_tuning)
            return melhores_parametros

        else:
            raise ValueError("Método de tuning inválido. Escolha entre 'optuna' ou 'skopt'.")

    except ValueError as e:
        print(f"Erro de valor: {e}")
        raise
    except Exception as e:
        print(f"Erro inesperado: {e}")
        raise
