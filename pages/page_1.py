import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import requests
import pickle
import datetime
import lightgbm
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tqdm import tqdm
from scipy.special import comb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    def __init__(self):
        self.data = pd.DataFrame() # raw data
        self.data_p = pd.DataFrame() #after preprocessing
        self.data_h = pd.DataFrame() #after merging horse_results
        self.data_pe = pd.DataFrame() #after merging peds
        self.data_c = pd.DataFrame() #after processing categorical features

    def merge_horse_results(self, hr, n_samples_list=[5, 9, 'all']):
        self.data_h = self.data_p.copy()
        for n_samples in n_samples_list:
            self.data_h = hr.merge_all(self.data_h, n_samples=n_samples)
        self.data_h.drop(['開催'], axis=1, inplace=True)

    def merge_peds(self, peds):
        self.data_pe = self.data_h.merge(peds,left_on='horse_id',
        right_index=True, how='left')
        self.no_peds = self.data_pe[self.data_pe['peds_0'].isnull()]\
            ['horse_id'].unique()
        if len(self.no_peds) > 0:
            print('scrape peds at horse_id_list "no_peds"')
    
    # カテゴリ変数の処理
    def process_categorical(self, le_horse, le_jockey, results_m):
        df = self.data_pe.copy()
        
        # ラベルエンコーディング。horse_id, jockey_idを0始まりの整数に変換
        mask_horse = df['horse_id'].isin(le_horse.classes_)
        new_horse_id = df['horse_id'].mask(mask_horse).dropna().unique()
        le_horse.classes_ = np.concatenate([le_horse.classes_, new_horse_id])
        df['horse_id'] = le_horse.transform(df['horse_id'])
        mask_jockey = df['jockey_id'].isin(le_jockey.classes_)
        new_jockey_id = df['jockey_id'].mask(mask_jockey).dropna().unique()
        le_jockey.classes_ = np.concatenate([le_jockey.classes_, new_jockey_id])
        df['jockey_id'] = le_jockey.transform(df['jockey_id'])
        
        # horse_id, jockey_idをpandasのcategory型に変換
        df['horse_id'] = df['horse_id'].astype('category')
        df['jockey_id'] = df['jockey_id'].astype('category')
        
        # その他のカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        # 訓練データとテストデータの列を一定にするため
        weathers = results_m['weather'].unique()
        race_types = results_m['race_type'].unique()
        ground_states = results_m['ground_state'].unique()
        sexes = results_m['性'].unique()
        df['weather'] = pd.Categorical(df['weather'], weathers)
        df['race_type'] = pd.Categorical(df['race_type'], race_types)
        df['ground_state'] = pd.Categorical(df['ground_state'], ground_states)
        df['性'] = pd.Categorical(df['性'], sexes)

        # columsでダミー変数化する列を指定
        df = pd.get_dummies(df, columns=['weather', 'race_type', 'ground_state', '性'])
        
        self.data_c = df

class ShutubaTable(DataProcessor):
    def __init__(self, shutuba_tables):
        super(ShutubaTable, self).__init__() # 親クラスのinitを実行する
        self.data = shutuba_tables

    @classmethod
    def scrape(cls, race_id_list, date):
            data = pd.DataFrame()
            for race_id in tqdm(race_id_list):
                url = 'https://race.netkeiba.com/race/shutuba.html?race_id=' + race_id
                df = pd.read_html(url)[0]

                # 列名に半角スペースがあれば除去する
                df = df.rename(columns=lambda x: x.replace(' ', ''))
                # マルチインデックスを削除
                df = df.T.reset_index(level=0, drop=True).T

                html = requests.get(url)
                html.encoding = "EUC-JP"
                soup = BeautifulSoup(html.text, "html.parser")

                # race_info
                texts = soup.find('div', attrs={'class': 'RaceData01'}).text
                texts = re.findall(r'\w+', texts)

                for text in texts: # textsの中身を順に見ていき、条件に合致すれば、該当の列に格納
                    if 'm' in text:
                        df['course_len'] = [int(re.findall(r'\d+', text)[0])] * len(df)
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df["weather"] = [text] * len(df)
                    if text in ["良", "稍", "重"]:
                        df["ground_state"] = [text] * len(df)
                    if '不' in text:
                        df["ground_state"] = ['不良'] * len(df)
                    if '芝' in text:
                        df['race_type'] = ['芝'] * len(df)
                    if '障' in text:
                        df['race_type'] = ['障害'] * len(df)
                    if 'ダ' in text:
                        df['race_type'] = ['ダート'] * len(df)
                df['date'] = [date] * len(df)

                # horse_id
                horse_id_list = []
                horse_td_list = soup.find_all("td", attrs={'class': 'HorseInfo'})
                for td in horse_td_list:
                    horse_id = re.findall(r'\d+', td.find('a')['href'])[0]
                    horse_id_list.append(horse_id)
                # jockey_id
                jockey_id_list = []
                jockey_td_list = soup.find_all("td", attrs={'class': 'Jockey'})
                for td in jockey_td_list:
                    jockey_id = re.findall(r'\d+', td.find('a')['href'])[0]
                    jockey_id_list.append(jockey_id)
                df['horse_id'] = horse_id_list
                df['jockey_id'] = jockey_id_list

                df.index = [race_id] * len(df)
                data = pd.concat([data, df])
                time.sleep(1)
            return cls(data)

    def preprocessing(self):
        df = self.data.copy()
        
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        # 馬体重を体重と体重変化に分ける
        df = df[df["馬体重(増減)"] != '--']
        df["体重"] = df["馬体重(増減)"].str.split("(", expand=True)[0].astype(int)
        df["体重変化"] = df["馬体重(増減)"].str.split("(", expand=True)[1].str[:-1].astype(int)
        # 2020/12/13追加：増減が「前計不」などのとき欠損値にする
        df['体重変化'] = pd.to_numeric(df['体重変化'], errors='coerce')

        # 日付型に変換
        df["date"] = pd.to_datetime(df["date"])
        
        df['枠'] = df['枠'].astype(int)
        df['馬番'] = df['馬番'].astype(int)
        df['斤量'] = df['斤量'].astype(int)
        df['開催'] = df.index.map(lambda x:str(x)[4:6])

        #6/6出走数追加
        df['n_horses'] = df.index.map(df.index.value_counts())

        # 距離は10の位を切り捨てる
        df["course_len"] = df["course_len"].astype(float) // 100

        # 不要な列を削除（どの列を使用するかを指定）
        df = df[['枠', '馬番', '斤量', 'course_len', 'weather','race_type',
        'ground_state', 'date', 'horse_id', 'jockey_id', '性', '年齢',
       '体重', '体重変化', '開催']]
        
        self.data_p = df.rename(columns={'枠': '枠番'})

class Results(DataProcessor):
    def __init__(self, results):
        super(Results, self).__init__()
        self.data = results
    
    # pickleファイルの読み込み
    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    # レース結果のスクレイピング
    @staticmethod
    def scrape(race_id_list):
        #race_idをkeyにしてDataFrame型を格納
        race_results = {}
        for race_id in tqdm(race_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/race/" + race_id
                #メインとなるテーブルデータを取得
                df = pd.read_html(url)[0]

                # 列名に半角スペースがあれば除去する
                df = df.rename(columns=lambda x: x.replace(' ', ''))

                html = requests.get(url)
                html.encoding = "EUC-JP"
                soup = BeautifulSoup(html.text, "html.parser")

                #天候、レースの種類、コースの長さ、馬場の状態、日付をスクレイピング
                texts = (
                    soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
                    + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
                )
                info = re.findall(r'\w+', texts)
                for text in info:
                    if text in ["芝", "ダート"]:
                        df["race_type"] = [text] * len(df)
                    if "障" in text:
                        df["race_type"] = ["障害"] * len(df)
                    if "m" in text:
                        df["course_len"] = [int(re.findall(r"\d+", text)[0])] * len(df)
                    if text in ["良", "稍重", "重", "不良"]:
                        df["ground_state"] = [text] * len(df)
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df["weather"] = [text] * len(df)
                    if "年" in text:
                        df["date"] = [text] * len(df)

                #馬ID、騎手IDをスクレイピング
                horse_id_list = []
                horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/horse")}
                )
                for a in horse_a_list:
                    horse_id = re.findall(r"\d+", a["href"])
                    horse_id_list.append(horse_id[0])
                jockey_id_list = []
                jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/jockey")}
                )
                for a in jockey_a_list:
                    jockey_id = re.findall(r"\d+", a["href"])
                    jockey_id_list.append(jockey_id[0])
                df["horse_id"] = horse_id_list
                df["jockey_id"] = jockey_id_list

                #インデックスをrace_idにする
                df.index = [race_id] * len(df)

                race_results[race_id] = df
            #存在しないrace_idを飛ばす
            except IndexError:
                continue
            except AttributeError: #存在しないrace_idでAttributeErrorになるページもあるので追加
                continue
            #wifiの接続が切れた時などでも途中までのデータを返せるようにする
            except Exception as e:
                print(e)
                break
            #Jupyterで停止ボタンを押した時の対処
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        race_results_df = pd.concat([race_results[key] for key in race_results])

        return race_results_df

    # 前処理の関数
    def preprocessing(self):
        df = self.data.copy() 

        # 列名に半角スペースがあれば除去する
        df = df.rename(columns=lambda x: x.replace(' ', ''))

        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df["着順"] = df["着順"].astype(int)
        df['rank'] = df["着順"].map(lambda x:1 if x<4 else 0)

        # 性齢を性と年齢に分ける
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        # 馬体重を体重と体重変化に分ける
        df["体重"] = df["馬体重"].str.split("(", expand=True)[0].astype(int)
        df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1].astype(int)

        # データをint, floatに変換
        df["単勝"] = df["単勝"].astype(float)
        df["course_len"] = df["course_len"].astype(float) // 100

        # 不要な列を削除
        df.drop(["タイム", "着差", "調教師", "性齢", "馬体重", "馬名", "騎手", "人気", "着順"], axis=1, inplace=True)
        
        df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")

        #開催場所
        df['開催'] = df.index.map(lambda x:str(x)[4:6])

        self.data_p = df

    # カテゴリー変数の処理
    def process_categorical(self):
        self.le_horse = LabelEncoder().fit(self.data_pe['horse_id'])
        self.le_jockey = LabelEncoder().fit(self.data_pe['jockey_id'])
        super().process_categorical(self.le_horse, self.le_jockey, self.data_pe) # 親クラスのproces_cat関数を実行

class HorseResults:
    def __init__(self, horse_results):
        self.horse_results = horse_results[['日付', '着順', '賞金', '着差', '通過',
                                            '開催', '距離']]
        self.preprocessing() #絶対に実行するものだから含めておく

    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(horse_id_list,pre_horse_results = {}):
        #horse_idをkeyにしてDataFrame型を格納
        horse_results = pre_horse_results
        for horse_id in tqdm(horse_id_list):
            if horse_id in horse_results.keys():
                continue
            try:
                time.sleep(1)
                url = 'https://db.netkeiba.com/horse/' + horse_id
                df = pd.read_html(url)[3]
                #受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
                if df.columns[0]=='受賞歴':
                    df = pd.read_html(url)[4]
                df.index = [horse_id] * len(df)
                horse_results[horse_id] = df
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる        
        horse_results_df = pd.concat([horse_results[key] for key in horse_results])

        return horse_results_df

    def preprocessing(self):
        df = self.horse_results.copy()
        
        # 列名に半角スペースがあれば除去する
        df = df.rename(columns=lambda x: x.replace(' ', ''))

        #着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df["着順"] = df["着順"].astype(int)

        df["date"] = pd.to_datetime(df["日付"])
        df.drop(['日付'], axis=1, inplace=True)
        
        #賞金のNaNを0で埋める
        df['賞金'].fillna(0, inplace=True)

        #1着の着差を0にする
        df['着差'] = df['着差'].map(lambda x:0 if x<0 else x)

        #レース展開データ
        # n=1: 最初のコーナーの順位、n=4：最終コーナーの順位
        def corner(x, n):
            if type(x) != str:
                return x
            elif  n==4:
                return int(re.findall(r'\d+', x)[-1])
            elif  n==1:
                return int(re.findall(r'\d+', x)[0])

        df['first_corner'] = df['通過'].map(lambda x: corner(x, 1))
        df['final_corner'] = df['通過'].map(lambda x: corner(x, 4))

        df['final_to_rank'] = df['final_corner'] - df['着順']
        df['first_to_rank'] = df['first_corner'] - df['着順']
        df['first_to_final'] = df['first_corner'] - df['final_corner']

        #開催場所
        df['開催'] = df['開催'].str.extract(r'(\D+)')[0].map(place_dict).fillna('11')
        df['race_type'] = df['距離'].str.extract('(\D+)')[0].map(race_type_dict)
        df['course_len'] = df['距離'].str.extract('(\d+)').astype(int)
        df.drop(['距離'], axis=1, inplace=True)

        #インデックス名を与える
        df.index.name = 'horse_id'

        self.horse_results = df
        self.target_list = ['着順', '賞金', '着差', 'first_corner',
                            'first_to_rank', 'first_to_final','final_to_rank']
    
    # n_samplesレース分、馬ごとに平均する
    def average(self, horse_id_list, date, n_samples='all'):
        target_df = self.horse_results.query('index in @horse_id_list')
        
        #過去何着分取り出すか指定
        if n_samples == 'all':
            filtered_df = target_df[target_df['date']<date]
        elif n_samples > 0:
            filtered_df = target_df[target_df['date'] < date].\
                sort_values('date', ascending=False).groupby(level=0).head(n_samples)
        else:
            raise Exception('n_samples must be >0')
            
        # average1 = filtered_df.groupby(level=0)[self.target_list].mean()
        # average2 = filtered_df.groupby(['horse_id', '開催', '距離', 'race_type'])[self.target_list].mean().unstack(level=[1,2,3])
        # average2.columns = average2.columns.map(lambda x: '_'.join(x))
        # average = pd.concat([average1, average2], axis=1).add_suffix('_{}R'.format(n_samples))

        self.average_dict = {}
        self.average_dict['non_category'] = filtered_df.groupby(level=0)[self.target_list]\
            .mean().add_suffix('_{}R'.format(n_samples))
        for column in ['course_len', 'race_type', '開催']:
            self.average_dict[column] = filtered_df.groupby(['horse_id', column])\
                [self.target_list].mean().add_suffix('_{}_{}R'.format(column, n_samples))

    # 日付ごとに実行する関数
    def merge(self, results, date, n_samples='all'):
        df = results[results['date']==date] #日付で絞り込み
        horse_id_list = df['horse_id']
        self.average(horse_id_list, date, n_samples)
        merged_df = df.merge(self.average_dict['non_category'], left_on='horse_id',
                             right_index=True, how='left')
        for column in ['course_len','race_type', '開催']:
            merged_df = merged_df.merge(self.average_dict[column], 
                                        left_on=['horse_id', column],
                                        right_index=True, how='left')
        return merged_df
    
    def merge_all(self, results, n_samples='all'):
        date_list = results['date'].unique()
        merged_df = pd.concat([self.merge(results, date, n_samples) for date in tqdm(date_list)])
        return merged_df

class Peds:
    def __init__(self, peds):
        self.peds = peds
        self.peds_e = pd.DataFrame() #after label encoding and transforming into category
    
    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)

    @staticmethod
    def scrape(horse_id_list):
        peds_dict = {}
        for horse_id in tqdm(horse_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/horse/ped/" + horse_id
                df = pd.read_html(url)[0]

                #重複を削除して1列のSeries型データに直す
                generations = {}
                for i in reversed(range(5)):
                    generations[i] = df[i]
                    df.drop([i], axis=1, inplace=True)
                    df = df.drop_duplicates()
                ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)

                peds_dict[horse_id] = ped.reset_index(drop=True)
            except IndexError:
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #列名をpeds_0, ..., peds_61にする
        peds_df = pd.concat([peds_dict[key] for key in peds_dict],
                            axis=1).T.add_prefix('peds_')

        return peds_df

    def encode(self):
        df = self.peds.copy()
        for column in df.columns:
            df[column] = LabelEncoder().fit_transform(df[column].fillna('Na'))
        self.peds_e = df.astype('category')

class Return:
    def __init__(self, return_tables):
        self.return_tables = return_tables
    
    @classmethod
    def read_pickle(cls, path_list):
        df = pd.read_pickle(path_list[0])
        for path in path_list[1:]:
            df = update_data(df, pd.read_pickle(path))
        return cls(df)
    
    @staticmethod
    def scrape(race_id_list):
        return_tables = {}
        for race_id in tqdm(race_id_list):
            time.sleep(1)
            try:
                url = "https://db.netkeiba.com/race/" + race_id

                #普通にスクレイピングすると複勝やワイドなどが区切られないで繋がってしまう。
                #そのため、改行コードを文字列brに変換して後でsplitする
                f = urlopen(url)
                html = f.read()
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(html)

                #dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
                df = pd.concat([dfs[1], dfs[2]])

                df.index = [race_id] * len(df)
                return_tables[race_id] = df
            except IndexError:
                continue
            except AttributeError: #存在しないrace_idでAttributeErrorになるページもあるので追加
                continue
            except Exception as e:
                print(e)
                break
            except:
                break

        #pd.DataFrame型にして一つのデータにまとめる
        return_tables_df = pd.concat([return_tables[key] for key in return_tables])
        return return_tables_df

    @property
    def fukusho(self):
        fukusho = self.return_tables[self.return_tables[0]=='複勝'][[1,2]]
        #wins = fukusho[1].str.split('br', expand=True).drop([3], axis=1)
        #5列できてしまう場合があるので、現在はこちらを推奨
        wins = fukusho[1].str.split('br', expand=True)[[0,1,2]]
        
        wins.columns = ['win_0', 'win_1', 'win_2']
        #returns = fukusho[2].str.split('br', expand=True).drop([3], axis=1)
        #5列できてしまう場合があるので、現在はこちらを推奨
        returns = fukusho[2].str.split('br', expand=True)[[0,1,2]]
        returns.columns = ['return_0', 'return_1', 'return_2']
        
        df = pd.concat([wins, returns], axis=1)
        for column in df.columns:
            df[column] = df[column].str.replace(',', '')
        return df.fillna(0).astype(int)
    
    @property
    def tansho(self):
        tansho = self.return_tables[self.return_tables[0]=='単勝'][[1,2]]
        tansho.columns = ['win', 'return']
        
        for column in tansho.columns:
            tansho[column] = pd.to_numeric(tansho[column], errors='coerce')
            
        return tansho
    
    @property
    def umaren(self):
        umaren = self.return_tables[self.return_tables[0]=='馬連'][[1,2]]
        wins = umaren[1].str.split('-', expand=True)[[0,1]].add_prefix('win_')
        return_ = umaren[2].rename('return')  
        df = pd.concat([wins, return_], axis=1)        
        return df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        
class ModelEvaluator:
    def __init__(self, model, return_tables_path_list, std=True):
        self.model = model
        self.rt = Return.read_pickle(return_tables_path_list)
        self.fukusho = self.rt.fukusho
        self.tansho = self.rt.tansho
        self.umaren = self.rt.umaren
        self.std = std

    # 確率を返す
    def predict_proba(self, X):
        proba = pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)
        if self.std:
            standard_scaler = lambda x: (x - x.mean()) / x.std()
            proba = proba.groupby(level=0).transform(standard_scaler)
            proba = (proba - proba.min()) / (proba.max() - proba.min())
        return proba

    # 予測確率に対してしきい値で0か1を判定。初期値は0.5
    def predict(self, X, threshold=0.5):
        y_pred = self.predict_proba(X)
        return [0 if p<threshold else 1 for p in y_pred]

    # AUCスコアを返す
    def score(self, y_true, X):
        return roc_auc_score(y_true, self.predict_proba(X))
    
    # 特徴重要を返す。表示数の初期値は20
    def feature_importance(self, X, n_display=20):
        importances = pd.DataFrame({"features": X.columns, 
                                    "importance": self.model.feature_importances_})
        return importances.sort_values("importance", ascending=False)[:n_display]

    # 予測の結果、3着以内で賭けると判定した馬番を格納
    def pred_table(self, X, threshold=0.5, bet_only=True):
        pred_table = X.copy()[['馬番', '単勝']]
        pred_table['pred'] = self.predict(X, threshold)
        if bet_only:
            return pred_table[pred_table['pred']==1][['馬番', '単勝']]
        else:
            return pred_table
        
    def fukusho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)
        money = -100 * n_bets
        df = self.fukusho.copy()
        df = df.merge(pred_table, left_index=True, right_index=True, how='right')
        for i in range(3):
            money += df[df['win_{}'.format(i)]==df['馬番']]['return_{}'.format(i)].sum()
            return_rate = (n_bets*100 + money) / (n_bets*100)
        return n_bets, return_rate
    
    def tansho_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        n_bets = len(pred_table)
        n_races = pred_table.index.nunique()

        money = -100 * n_bets
        df = self.tansho.copy()
        df = df.merge(pred_table, left_index=True, right_index=True, how='right')

        std = ((df['win'] == df['馬番'])*df['return']).groupby(level=0).sum().std()\
              * np.sqrt(n_races) / (n_bets * 100)

        n_hits = len(df[df['win']==df['馬番']])
        money += df[df['win']==df['馬番']]['return'].sum()
        return_rate = (n_bets*100 + money) / (n_bets*100)
        return n_bets, return_rate, n_hits, std
    
    def tansho_return_proper(self, X, threshold=0.5):
        # モデルによって「賭ける」と判定された馬たち
        pred_table = self.pred_table(X, threshold)
        # 賭けた枚数
        n_bets = len(pred_table)
        n_races = pred_table.index.nunique()

        # 払い戻し表にpred_tableをマージ
        df = self.tansho.copy()
        df = df.merge(pred_table, left_index=True, right_index=True, how='right')

        bet_money = (1 / pred_table['単勝']).sum()
        
        std = ((df['win'] == df['馬番']).astype(int)).groupby(level=0).sum().std() \
            * np.sqrt(n_races) / bet_money

        # 単勝適正回収率を計算
        n_hits = len(df.query('win == 馬番'))
        return_rate = n_hits / bet_money

        return n_bets, return_rate, n_hits, std
        
    def umaren_return(self, X, threshold=0.5):
        pred_table = self.pred_table(X, threshold)
        hit = {}
        n_bets = 0
        for race_id, preds in pred_table.groupby(level=0):
            n_bets += comb(len(preds), 2)
            hit[race_id] = set(self.umaren.loc[race_id][['win_0', 'win_1']]).issubset(set(preds))
        return_rate = self.umaren.index.map(hit).values * self.umaren['return'].sum() / (n_bets * 100)
        return n_bets, return_rate


def update_data(old, new):
    filtered_old = old[~old.index.isin(new.index)]
    return pd.concat([filtered_old, new])

def split_data(df, test_size=0.3):
    sorted_id_list = df.sort_values("date").index.unique() #日付順に並び替え
    train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))] #古いデータから順に1-testsizeの割合に
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :] #古いデータから順に1-testsizeの割合に
    #dateは以降使わないのここで消しておく
    train = df.loc[train_id_list]#.drop(['date'], axis=1) #求めたid_listから訓練データを取り出す
    test = df.loc[test_id_list]#.drop(['date'], axis=1) #テストデータも同様
    return train, test

def gain(return_func, X, n_samples=100, lower=5, min_threshold=0.5):
    gain = {}
    for i in tqdm(range(n_samples)):
        threshold = 1 * i / n_samples + min_threshold * (1-(i/n_samples)) # たすき掛けの考え
        n_bets, return_rate, n_hits, std = return_func(X, threshold)
        if n_bets > lower:
            gain[n_bets] = {'return_rate': return_rate, 'n_hits': n_hits, 'std': std}
    return pd.DataFrame(gain).T

place_dict = {
    '札幌':'01', '函館':'02', '福島':'03', '新潟':'04', '東京':'05', \
    '中山':'06', '中京':'07', '京都':'08', '阪神':'09', '小倉':'10'
}
race_type_dict = {
    '芝':'芝', 'ダ':'ダート', '障':'障害'
}

st.title('データ入力')
st.caption('レース直前に指定した出馬表データを読み込んでテスト')

with st.form(key='profile_form'):
    
    # # セレクトボックス
    race_id = st.selectbox(
        'レースID',
        ('202308010401', '202308010402','202308010403','202308010404', '202308010405', '202308010406', \
         '202308010407', '202308010408', '202308010409', '202308010410', '202308010411', '202308010412')
    )
    # セレクトボックス

    # 日付
    date = st.date_input(
        '開始日',
        datetime.date(2023, 4, 30)
    )

    # ボタン
    submit_btn = st.form_submit_button('送信')
    cancel_btn = st.form_submit_button('キャンセル')

    if submit_btn:
        # 学習モデルmodel.pklの読み込み
        with open('./data/model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        st.text('～データ処理中～')
        # 訓練データの処理
        r = Results.read_pickle(['./data/results_2020.pickle', './data/results2021.pickle', \
                                 './data/results2022.pickle', './data/results_2023.pickle'])
        r.preprocessing()
        hr = HorseResults.read_pickle(['./data/horse_results.pickle'])
        r.merge_horse_results(hr)
        p = Peds.read_pickle(['./data/peds.pickle'])
        p.encode()
        r.merge_peds(p.peds_e)
        r.process_categorical()

        # テストデータの読み込み
        stb = ShutubaTable.scrape([race_id], date)
        stb.preprocessing()
        hr = HorseResults.read_pickle(['./data/horse_results.pickle'])
        stb.merge_horse_results(hr)
        p = Peds.read_pickle(['./data/peds.pickle'])
        p.encode()
        stb.merge_peds(p.peds_e)
        stb.process_categorical(r.le_horse, r.le_jockey, r.data_pe)

        # st.text('読み込み完了！')


        me = ModelEvaluator(model, ['./data/return_tables_2020.pickle', './data/return_tables_2021.pickle'\
                                    './data/return_tables_2022.pickle', './data/return_tables_2023.pickle'])
        X_test = stb.data_c.drop(['date'], axis=1)
        pred = me.predict_proba(X_test)
        df = pd.DataFrame()
        df['確率'] = pred
        df.index = stb.data_c['馬番']

        st.subheader('予測確率')
        st.dataframe(df)

        st.subheader('特徴重要度')
        st.dataframe(me.feature_importance(X_test))

        st.subheader('特徴量')
        st.dataframe(X_test)