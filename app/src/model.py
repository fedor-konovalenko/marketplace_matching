import pandas as pd
import numpy as np
import pickle
import joblib
import faiss
import logging
#from sklearn.preprocessing import MinMaxScaler


m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
handler_m = logging.StreamHandler()#FileHandler(f"{__name__}.log", mode='w')
formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
handler_m.setFormatter(formatter_m)
m_logger.addHandler(handler_m)


MODEL = joblib.load('model_LR_S.joblib')

with open('common_index_sh.pkl', 'rb') as f:
    DATA_INDEX = pickle.load(f)

SCALER = joblib.load('scaler.save')

#with open('features.npy', 'rb') as f:
  #DATA_FEATURES = np.load(f)
CONST_FEATURES = [70, 25, 21]
  
FAISS_INDEX = faiss.read_index("index_sh.bin")

K_RUDE = 20
K_FINE = 5

def rude_search(vector: np.array, kr) -> np.array:
    '''search K neibourghs with FAISS only for single feature LR usage'''
    dist, idx_found = FAISS_INDEX.search(np.ascontiguousarray(vector).astype('float32'), kr)
    #index_list = np.concatenate(idx_found).tolist()
    #cnd = []
    #for _ in index_list:
      #cnd.append(DATA_FEATURES[_].tolist())
    dist = dist.reshape(-1, 1)
    #data = np.repeat(vector, K, axis=0)
    cand_index = np.concatenate(idx_found).reshape(-1, 1)
    return np.hstack((dist, cand_index))

def fine_search(features: np.array, kf):
    '''fine search with pretraineg logistic regression'''
    X_short = features[:, 0].reshape(-1,1)
    prob_pred = MODEL.predict_proba(X_short)
    prob_pred_1 = prob_pred[:,1]
    answer = pd.DataFrame({'id':features[:, 1].astype('int').tolist(), 
                           'probability':prob_pred_1.tolist()})
    top_5 = answer.sort_values(by='probability', ascending=False)[:kf]['id']
    return top_5
    
def match_5(path: str):
    '''main process function'''
    with open(path, 'r') as f:
        query = f.read()
        query = query.split(' ')
    try:
        query = np.array(query, dtype='float')
    except:
        status = 'not valid data'
        m_logger.error(f'problems with file {status}')
        return [], status
    try:
        query = np.delete(query, CONST_FEATURES)
    except:
        status = 'not valid data'
        m_logger.error(f'problems with array shape {status}')
        return [], status
    #scale = max(abs(min(query)), max(query))
    pre_query = SCALER.transform(query.reshape(1, -1))
    matrix = rude_search(pre_query, K_RUDE)
    result = fine_search(matrix, K_FINE)
    if len(result) == 5:
        status = 'success. your query matches with:'
        result = result.to_json()
        m_logger.info(f'bingo!! {status}')
    else:
        status = 'matching error'
        m_logger.error(f'some problems {status}')
    return result, status

    
