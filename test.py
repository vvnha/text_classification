import pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
import gensim
# from gensim.summarization import keywords
import numpy as np
from tensorflow.python.keras import models
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from tensorflow.python.keras.optimizer_v2.adam import Adam
import sys

file_model = "data/model.json"

# loaded_model = models.load_model("data/model")

loaded_model = joblib.load("data/model.joblib")


def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines


test_doc_bongda = '''Tiếp đón ĐT Việt Nam trên sân nhà ở bán kết lượt đi AFF Cup 2018, những sai lầm nơi hàng thủ đã khiến ĐT Philippines nhận thất bại cay đắng 1-2. Sau trận, một số cầu thủ liên tục đăng đàn thể hiện sự tiếc nuối với kết quả này, thậm chí tuyên bố đội nhà xứng đáng giành chiến thắng hơn. 

Philippines thua Việt Nam: Nội bộ lục đục, báo châu Á khó tin phép màu - 1

Patrick Reichelt chỉ trích thái độ thi đấu thiếu quyết tâm của các đồng đội

Tuy nhiên khá bất ngờ khi Patrick Reichelt - tác giả bàn gỡ 1-1 lại lên tiếng chỉ trích thái độ thi đấu của các đồng đội giữa thời điểm nhạy cảm. Phản ứng này khiến dư luận nghi ngờ về tình trạng lục đục nội bộ ở Philippines.

"Các cầu thủ chỉ chơi với 80-90% phong độ, điều đó không đủ giúp Philippines chiến thắng. Tôi không hề muốn dừng bước trong lần thứ 3 lọt vào bán kết AFF Cup nhưng nếu toàn đội thi đấu hết mình, tôi sẽ không cảm thấy hối tiếc dù thất bại. Philippines đã có sự chuẩn bị rất tốt, vấn đề nằm ở thái độ thi đấu", trích lời Reichelt trên Fox Sport Asia.

Trong khi đó, chuyên gia bóng đá Đông Nam Á nổi tiếng Gabriel Tan cũng phân tích khá chi tiết những điểm mạnh, điểm yếu của Philppines ở bài viết: "AFF Cup: Philippines vẫn còn cơ hội sống sót hay Việt Nam đã đặt một chân vào chung kết?".

"Philippines phần nào tái hiện được tinh thần và lối chơi từng giúp họ cầm hòa ĐKVĐ Thái Lan 1-1 ở vòng bảng. Thầy trò Sven-Goran Eriksson cũng gây ra nhiều khó khăn cho Việt Nam suốt 90 phút, thậm chí trở thành đội đầu tiên chọc thủng lưới Đặng Văn Lâm ở AFF Cup 2018".

Tuy nhiên, Gabriel Tan lại bỏ ngỏ khả năng thầy trò Eriksson lội ngược dòng khi hành quân tới Hà Nội vào ngày 6/12 tới và chỉ gợi lại kỉ niệm đẹp tại SVĐ Mỹ Đình 8 năm trước - thời điểm Philippines đánh bại Việt Nam 2-0:

"Philippines có thể lội ngược dòng? Không có gì đảm bảo cả. Việt Nam vẫn còn nhiều phương án chiến thuật, nhân sự cho khả năng tấn công biên, trong khi The Azkals chỉ còn 18 cầu thủ. Tới Hà Nội, HLV Eriksson chỉ biết hy vọng các học trò thể hiện tinh thần quyết tâm như trận hòa Thái Lan và tái hiện phép màu Hà Nội cách đây 8 năm".

Philippines thua Việt Nam: Nội bộ lục đục, báo châu Á khó tin phép màu - 2
"Phép màu Hà Nội 2010" là yếu tố để giới chuyên môn lẫn các cầu thủ Philippines bấu víu ở trận bán kết lượt về

Về màn trình diễn của ĐT Việt Nam, Gabriel Tan đánh giá rất cao HLV Park Hang Seo với những điều chỉnh chiến thuật, nhân sự cực kì táo bạo, nhạy bén: 

"Chiến thắng của Việt Nam ấn tượng hơn cả bởi HLV Park Hang Seo thậm chí chưa tung ra Văn Quyết, Xuân Trường, trong khi Công Phượng chỉ vào sân 10 phút cuối. Thay vào đó, Đức Huy và Hùng Dũng - những cầu thủ mới đá chính ở lượt trận cuối vòng bảng gặp Campuchia - được lựa chọn cho vị trí tiền vệ trung tâm.

Nhiều người cho rằng họ vào sân chỉ để giúp Xuân Trường dưỡng sức, tạo điều kiện cho Quang Hải trở về vị trí đá cánh sở trường, nhưng chiến lược gia người Hàn Quốc lại nghĩ khác. Ông không e ngại đặt niềm tin vào những cầu thủ trẻ. Mặt khác, hàng thủ với bộ ba hậu vệ, Đặng Văn Lâm và đôi cánh Trọng Hoàng - Văn Hậu tiếp tục cho thấy sự ăn ý đáng kinh ngạc".

'''

test_doc_test = '''
Đề tài có thể ứng dụng AI trong việc phân loại nội dung của tài liệu từ đó kiểm tra, xem xét rằng tài liệu đó có được sao chép từ các tài liệu trước đó hay không và đồng thời hiển thị mức độ giống nhau giữa các tài liệu và báo về cho người dùng để nhận biết việc sao chép nội dung'''

test_ba = '''	
Tài trợ thương mại Quốc tế và một số giải pháp để nâng cao hiệu quả hoạt động tài trợ thương mại Quốc tế của ngân hàng công thương Việt Nam'''
test_doc = test_doc_test

X_data = pickle.load(open('data/X_data.pkl', 'rb'))
y_data = pickle.load(open('data/y_data.pkl', 'rb'))

tfidf_vect = pickle.load(open('data/tfidf_vect.pkl', 'rb'))
svd = joblib.load("data/svd.p")

def test(test_doc):

    test_doc = preprocessing_doc(test_doc)
    test_doc_tfidf = tfidf_vect.transform([test_doc])
    print(np.shape(test_doc_tfidf))
    test_doc_svd = svd.transform(test_doc_tfidf)
    probality = loaded_model.predict_proba(test_doc_tfidf)

    encoder = preprocessing.LabelEncoder()
    y_data_n = encoder.fit_transform(y_data)
    typeText = encoder.classes_

    rs = loaded_model.predict(test_doc_tfidf)
    rsArray = list(zip(probality.flatten(), typeText))


    def takeFirst(elem):
        return elem[0]


    rsArray.sort(key=takeFirst, reverse=True)
    return rsArray

# X_data = pickle.load(open('data/X_data.pkl', 'rb'))
# y_data = pickle.load(open('data/y_data.pkl', 'rb'))

# tfidf_vect = pickle.load(open('data/tfidf_vect.pkl', 'rb'))
# svd = joblib.load("data/svd.p")

# test_doc = preprocessing_doc(test_doc)
# test_doc_tfidf = tfidf_vect.transform([test_doc])
# print(np.shape(test_doc_tfidf))
# test_doc_svd = svd.transform(test_doc_tfidf)
# probality = loaded_model.predict_proba(test_doc_tfidf)

# encoder = preprocessing.LabelEncoder()
# y_data_n = encoder.fit_transform(y_data)
# typeText = encoder.classes_

# rs = loaded_model.predict(test_doc_tfidf)
# rsArray = list(zip(probality.flatten(), typeText))


# def takeFirst(elem):
#     return elem[0]


# rsArray.sort(key=takeFirst, reverse=True)

# print(rsArray)

# print(gensim.summarization.keywords(test_doc))

# # install https://github.com/trungtv/vivi_spacy/raw/master/vi/vi_core_news_md-2.0.1/dist/vi_core_news_md-2.0.1.tar.gz
# # to key extractor

sys.modules[__name__] = test