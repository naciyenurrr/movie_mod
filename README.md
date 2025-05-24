# Giriş
Bu projemde Kaggle üzerinden movies.csv veri seti kullanılarak, Visual Studio ortamında filmlerin açıklamalarına dayanarak her bir filme ait "mod" (ruh hali) etiketini tahmin etmeye yönelik bir metin sınıflandırma modeli geliştirildi. Veri setimde bulunan overview, genres, tagline, keywords alanlarını kullanrak filmlerin etiketlenmemiş moodlarını tahmin etmek. 
# Metrikler
İlk olarak veri ön işlemesi gerçekleştirildi. overview, genres, tagline ve keywords sütunlarının birleştirilmesiyle yeni bir text sütunu oluşturuldu. Modelleme kısmında TF-IDF ile metinlerin sayısal vektörlere dönüştürülmesi gerçekleştirildi.
3 tane sınıflandırıcı test edildi:
-Multinomial Naive Bayes
-Logistic Regression
-Linear Support Vector Classifier (SVC)
Model seçimi olarak en iyi sonuçları veren LinearSVC modeli seçildi.
Model karşılaştırma sonuçaları ise boxplot ile görselleştirildi.
# Sonuç ve Gelecek çalışmalar
web tabanlı arayüz geliştirilemsi ve yayınlanmsı ile kullanıcının o anki istediği moodu üzerinden ona film önerileri yapabiliriz. Bu sayede kullanıcı dostu bir uygulama gerçekleştirilmiş olur.
# Linkler
https://www.kaggle.com/code/naciyenur/notebookd221290481

