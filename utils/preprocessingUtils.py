import pandas as pd
import datetime as dt
import numpy as np
import warnings
from typing import Callable, Dict, Any, List, Tuple
from itables import init_notebook_mode
from pandas.tseries.holiday import AbstractHolidayCalendar, GoodFriday, Holiday, Easter, Day
from pandas.tseries.offsets import CustomBusinessDay


def configurar_ambiente() -> None:
    """
    Configura o ambiente do notebook definindo opções de exibição do pandas,
    desativando warnings e habilitando o modo interativo do itables.
    """
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    warnings.filterwarnings("ignore")
    init_notebook_mode(all_interactive=True)


class Feriados_Brasil(AbstractHolidayCalendar):
    """
    Classe que define os feriados nacionais brasileiros, incluindo feriados fixos
    e móveis com base na data da Páscoa.
    """
    
    rules = [
        Holiday('Confraternização Universal', month=1, day=1),
        Holiday('Segunda Feira de Carnaval', month=1, day=1, offset=[Easter(), Day(-48)]),
        Holiday('Terça Feira de Carnaval', month=1, day=1, offset=[Easter(), Day(-47)]),
        GoodFriday,
        Holiday('Tiradentes', month=4, day=21),
        Holiday('Dia do Trabalho', month=5, day=1),
        Holiday('Corpus Christi', month=1, day=1, offset=[Easter(), Day(60)]),
        Holiday('Independência do Brasil', month=9, day=7),
        Holiday('Nossa Senhora Aparecida', month=10, day=12),
        Holiday('Finados', month=11, day=2),
        Holiday('Proclamação da República', month=11, day=15),
        Holiday('Natal', month=12, day=25),
        Holiday('Dia de São Jorge', month=4, day=23)
    ]


br_feriados = CustomBusinessDay(calendar=Feriados_Brasil())
inst = Feriados_Brasil()

lista_feriados = inst.holidays(dt.datetime(2017, 1, 1), dt.datetime(2023, 12, 31))


def eh_feriado(row: pd.Series) -> int:
    """
    Verifica se a data contida na linha do DataFrame é um feriado.

    Parameters:
        row (pd.Series): Linha do DataFrame contendo a coluna 'data_inversa'.

    Returns:
        int: 1 se a data for feriado, 0 caso contrário.
    """
    
    data = pd.Timestamp(row['data_inversa'])
    return 1 if data.date() in lista_feriados.date else 0


def mapeador(dicionario: Dict[Tuple[Any, ...], Any]) -> Callable[[Any], Any]:
    """
    Cria uma função de mapeamento baseada em um dicionário de conjuntos.

    Parameters:
        dicionario (dict): Dicionário onde as chaves são tuplas com os valores a serem mapeados.

    Returns:
        function: Função que realiza o mapeamento de valores.
    """
    def mapper(valor: Any) -> Any:
        for chaves, resultado in dicionario.items():
            if valor in chaves:
                return resultado
        return None
    return mapper


def categorizar_tracado_via(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categoriza os tipos de traçado da via com base na descrição textual da coluna 'tracado_via'.

    Parameters:
        df (pd.DataFrame): DataFrame contendo a coluna 'tracado_via'.

    Returns:
        pd.DataFrame: DataFrame com novas colunas categorizadas.
    """
    
    def extrair_categorias(tracado: str) -> pd.Series:
        condicao_pista = 'Normal'
        tipo_inclinacao = 'Plano'
        tipo_superficie = 'Reta'
        tipo_manobra = 'Nenhum'
        tipo_estrutura = 'Nenhum'

        if 'Em Obras' in tracado or 'Desvio Temporário' in tracado:
            condicao_pista = 'Em obras'

        if 'Aclive' in tracado:
            tipo_inclinacao = 'Aclive'
        elif 'Declive' in tracado:
            tipo_inclinacao = 'Declive'

        if 'Curva' in tracado:
            tipo_superficie = 'Curva'

        if 'Rotatória' in tracado:
            tipo_manobra = 'Rotatória'
        elif 'Interseção' in tracado:
            tipo_manobra = 'Interseção'
        elif 'Retorno Regulamentado' in tracado:
            tipo_manobra = 'Retorno Regulamentado'

        if 'Ponte' in tracado:
            tipo_estrutura = 'Ponte'
        elif 'Túnel' in tracado:
            tipo_estrutura = 'Túnel'
        elif 'Viaduto' in tracado:
            tipo_estrutura = 'Viaduto'

        return pd.Series([condicao_pista, tipo_inclinacao, tipo_superficie, tipo_manobra, tipo_estrutura])

    df[['condicaoPista', 'tipoInclinacao', 'tipoSuperficie', 'tipoManobra', 'tipoEstrutura']] = df['tracado_via'].apply(extrair_categorias)

    return df


def conversorCiclico(df: pd.DataFrame, colunas: List[str], valoresMaximos: List[int]) -> pd.DataFrame:
    """
    Converte variáveis cíclicas em componentes senoidais e cossenoidais.

    Parameters:
        df (pd.DataFrame): DataFrame de entrada.
        colunas (list): Lista de colunas a serem transformadas.
        valoresMaximos (list): Lista de valores máximos correspondentes a cada coluna.

    Returns:
        pd.DataFrame: DataFrame com colunas transformadas e originais removidas.
    """
    
    for coluna, valorMaximo in zip(colunas, valoresMaximos):
        df[coluna + 'Sen'] = np.sin(2 * np.pi * df[coluna] / valorMaximo)
        df[coluna + 'Cos'] = np.cos(2 * np.pi * df[coluna] / valorMaximo)
        df.drop(coluna, axis=1, inplace=True)
    return df


def removeOutliers(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """
    Remove outliers de múltiplas colunas com base no intervalo interquartil (IQR).

    Parameters:
        df (pd.DataFrame): DataFrame de entrada.
        colunas (list): Lista de nomes de colunas das quais remover os outliers.

    Returns:
        pd.DataFrame: DataFrame sem os outliers.
    """
    
    for coluna in colunas:
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        df = df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]

    return df


def define_gravidade(row: pd.Series) -> int:
    """
    Define a gravidade de um acidente com base nas vítimas.

    Parameters:
        row (pd.Series): Linha do DataFrame contendo as colunas 'mortos', 'feridos_graves',
                         'feridos_leves' e 'ilesos'.

    Returns:
        int: 1 para acidentes com mortos ou feridos graves, 0 caso contrário.
    """
    
    if row['mortos'] > 0 or row['feridos_graves'] > 0:
        return 1
    elif row['feridos_leves'] > 0 or row['ilesos'] > 0:
        return 0
    else:
        return 0


def define_fase_do_dia(coluna: List[int]) -> List[str]:
    """
    Determina a fase do dia (Dia ou Noite) com base no horário.

    Parameters:
        coluna (List[int]): Lista de valores inteiros representando horas.

    Returns:
        List[str]: Lista de strings com os valores 'Dia' ou 'Noite'.
    """
    
    fase = []

    for hora in coluna:
        if 6 < hora < 18:
            fase.append('Dia')
        else:
            fase.append('Noite')

    return fase
