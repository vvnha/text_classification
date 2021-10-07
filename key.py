import yake

from langdetect import detect, detect_langs

textVi = 'Lập Trình Không Khó là blog chia sẻ kiến thức lập trình miễn phí . Do vậy , tệp khách hàng của chúng tôi chủ yếu là đối tượng học lập trình , độ tuổi từ 18 - 24 và hầu hết là người dùng đến từ Việt Nam . Lập Trình Không Khó có hơn 300.000 người dùng trung bình mỗi tháng , đóng góp lượt xem trang trung bình mỗi ngày trên 20.000 views . Trong đó , trên 80 % lượng truy cập đến từ công cụ tìm kiếm ( Google , Cốc Cốc , ... ) . Ngoài ra , nhóm Lập Trình Không Khó trên Facebook hoạt động sôi nổi có tới 30.000 thành viên .'

textEng = "Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning "\
    "competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud "\
    "Next conference in San Francisco this week, the official announcement could come as early as tomorrow. "\
    "Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. "\
    "Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, "\
    "was founded by Goldbloom  and Ben Hamner in 2010. "\
    "The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, "\
    "it has managed to stay well ahead of them by focusing on its specific niche. "\
    "The service is basically the de facto home for running data science and machine learning competitions. "\
    "With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, "\
    "it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow "\
    "and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, "\
    "Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. "\
    "That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google "\
    "will keep the service running - likely under its current name. While the acquisition is probably more about "\
    "Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition "\
    "and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can "\
    "share this code on the platform (the company previously called them 'scripts'). "\
    "Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with "\
    "that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) "\
    "since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, "\
    "Google chief economist Hal Varian, Khosla Ventures and Yuri Milner "

text = textVi
# lan = 'en'
# Ta cần custom stopword tiếng Việt do thư viện này không có sẵn stopword list cho tiếng Việt
stopwords = open('Keyword/stopwords.txt', encoding="utf8").read().splitlines()

lan = detect(text)

if(lan == 'vi'):
    kw_extractor = yake.KeywordExtractor(lan='vi', n=2, stopwords=stopwords)
else:
    kw_extractor = yake.KeywordExtractor()

# Khởi tạo YAKE với ngôn ngữ tiếng Việt (làm màu), sinh ứng viên 1-gram và 2-gram, với custom stopwrod
# kw_extractor = yake.KeywordExtractor(lan='vi', n=2, stopwords=stopwords)

# Truy vấn trích rút từ khóa
keywords = kw_extractor.extract_keywords(text)

for kw in keywords:
    print(kw)
